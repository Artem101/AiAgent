//! Headless Chromium driven over the DevTools protocol.
//!
//! The browser is started with `--remote-debugging-port=0`; the DevTools address is read from
//! its stderr, the page target is found through `/json/list`, and all interaction goes through
//! [`Cdp`]: navigation (`Page.navigate`, history), DOM snapshots (`Runtime.evaluate`), real
//! mouse clicks (`Input.dispatchMouseEvent`) and typing (`Input.insertText`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use candle_core::{bail, Error, Result};
use serde_json::{json, Value};

use super::cdp::{http_get, Cdp};
use super::{Browser, Element, PageSnapshot, Role};

/// Collects visible elements in document order and tags them with `data-aid` ids.
const SNAPSHOT_JS: &str = r#"(() => {
  for (const el of document.querySelectorAll('[data-aid]')) el.removeAttribute('data-aid');
  const out = [];
  const els = document.body ? document.body.querySelectorAll('h1,p,a,input,button,th,td') : [];
  for (const el of els) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const tag = el.tagName.toLowerCase();
    el.setAttribute('data-aid', String(out.length));
    out.push({ tag, text: tag === 'input' ? '' : el.innerText.trim(), value: tag === 'input' ? el.value : '' });
  }
  return JSON.stringify({ url: location.href, title: document.title, elements: out });
})()"#;

const NAVIGATION_STARTED: [&str; 3] =
    ["Page.frameRequestedNavigation", "Page.frameStartedLoading", "Page.frameNavigated"];

/// How to start the browser.
#[derive(Debug, Clone)]
pub struct ChromeOptions {
    /// Browser binary; `None` = [`Chrome::find_executable`].
    pub executable: Option<PathBuf>,
    /// Run without a window (the only option on a server).
    pub headless: bool,
    /// Pass `--no-sandbox` (needed when running as root, e.g. in containers).
    pub no_sandbox: bool,
    /// Time to wait for a page load.
    pub load_timeout: Duration,
}

impl Default for ChromeOptions {
    fn default() -> Self {
        Self { executable: None, headless: true, no_sandbox: true, load_timeout: Duration::from_secs(15) }
    }
}

/// A running Chromium with one tab under CDP control. Killed on drop.
pub struct Chrome {
    child: Child,
    cdp: Cdp,
    profile: PathBuf,
    load_timeout: Duration,
}

impl Chrome {
    /// `$COG_CHROME`, then Playwright's Chromium, then `chromium`/`google-chrome` on `PATH`.
    pub fn find_executable() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("COG_CHROME").map(PathBuf::from) {
            return Some(p);
        }
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(dir) = std::env::var_os("PLAYWRIGHT_BROWSERS_PATH") {
            candidates.push(Path::new(&dir).join("chromium"));
        }
        candidates.push("/opt/pw-browsers/chromium".into());
        let names = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome"];
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                candidates.extend(names.iter().map(|n| dir.join(n)));
            }
        }
        candidates.push("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into());
        candidates.into_iter().find(|p| p.is_file())
    }

    /// Starts the browser with the default options.
    pub fn launch_default() -> Result<Self> {
        Self::launch(&ChromeOptions::default())
    }

    pub fn launch(opts: &ChromeOptions) -> Result<Self> {
        let exe = match opts.executable.clone().or_else(Self::find_executable) {
            Some(p) => p,
            None => bail!("Chromium not found: install it or set COG_CHROME=/path/to/chrome"),
        };
        static LAUNCHES: AtomicUsize = AtomicUsize::new(0);
        let profile = std::env::temp_dir().join(format!(
            "cog_engine_chrome_{}_{}",
            std::process::id(),
            LAUNCHES.fetch_add(1, Ordering::Relaxed)
        ));
        let mut cmd = Command::new(&exe);
        if opts.headless {
            cmd.arg("--headless=new");
        }
        if opts.no_sandbox {
            cmd.arg("--no-sandbox");
        }
        cmd.args([
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-sync",
            "--mute-audio",
            // Back/forward cache would restore pages without a load event.
            "--disable-features=BackForwardCache,Translate,MediaRouter",
            "--window-size=1024,768",
        ])
        .arg(format!("--user-data-dir={}", profile.display()))
        .arg("about:blank")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| Error::Msg(format!("cannot start {}: {e}", exe.display())))?;

        // The DevTools address is printed on stderr; keep draining it afterwards so the
        // browser never blocks on a full pipe.
        let stderr = child.stderr.take().expect("stderr is piped");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(std::io::Result::ok) {
                if let Some(url) = line.strip_prefix("DevTools listening on ") {
                    let _ = tx.send(url.trim().to_string());
                }
            }
        });
        let browser_ws = match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(url) => url,
            Err(_) => {
                let _ = child.kill();
                bail!("{} did not open a DevTools endpoint", exe.display())
            }
        };
        let host = browser_ws.trim_start_matches("ws://").split('/').next().unwrap_or_default().to_string();

        let mut page_ws = None;
        for _ in 0..50 {
            let targets: Value = serde_json::from_str(&http_get(&host, "/json/list")?).map_err(Error::wrap)?;
            page_ws = targets.as_array().into_iter().flatten().find_map(|t| {
                (t["type"] == "page").then(|| t["webSocketDebuggerUrl"].as_str().map(str::to_string)).flatten()
            });
            if page_ws.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let Some(page_ws) = page_ws else {
            let _ = child.kill();
            bail!("no page target in {}", exe.display())
        };
        let mut cdp = Cdp::connect(&page_ws)?;
        cdp.call("Page.enable", json!({}))?;
        Ok(Self { child, cdp, profile, load_timeout: opts.load_timeout })
    }

    /// Raw access to the DevTools session.
    pub fn cdp(&mut self) -> &mut Cdp {
        &mut self.cdp
    }

    fn wait_load(&mut self) -> Result<()> {
        match self.cdp.wait_event(&["Page.loadEventFired"], self.load_timeout)? {
            Some(_) => Ok(()),
            None => bail!("page did not finish loading in {:?}", self.load_timeout),
        }
    }

    /// Checks that element `id` of the last snapshot exists and runs `body` on it (`el`).
    fn with_element(&mut self, id: usize, body: &str) -> Result<Value> {
        self.cdp.eval(&format!(
            "(() => {{ const el = document.querySelector('[data-aid=\"{id}\"]'); if (!el) return null; {body} }})()"
        ))
    }
}

impl Drop for Chrome {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.profile);
    }
}

impl Browser for Chrome {
    fn goto(&mut self, url: &str) -> Result<()> {
        self.cdp.clear_events();
        let r = self.cdp.call("Page.navigate", json!({ "url": url }))?;
        if let Some(err) = r.get("errorText").and_then(Value::as_str) {
            bail!("cannot open {url}: {err}")
        }
        self.wait_load()
    }

    fn snapshot(&mut self) -> Result<PageSnapshot> {
        let raw = self.cdp.eval(SNAPSHOT_JS)?;
        let v: Value = serde_json::from_str(raw.as_str().unwrap_or("{}")).map_err(Error::wrap)?;
        let s = |v: &Value| v.as_str().unwrap_or_default().to_string();
        let elements = v["elements"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let role = Role::from_tag(e["tag"].as_str()?)?;
                Some(Element { role, text: s(&e["text"]), value: s(&e["value"]) })
            })
            .collect();
        Ok(PageSnapshot { url: s(&v["url"]), title: s(&v["title"]), elements })
    }

    fn click(&mut self, id: usize) -> Result<()> {
        let p = self.with_element(
            id,
            "el.scrollIntoView({ block: 'center' }); const r = el.getBoundingClientRect(); \
             return { x: r.left + r.width / 2, y: r.top + r.height / 2 };",
        )?;
        let (Some(x), Some(y)) = (p["x"].as_f64(), p["y"].as_f64()) else { bail!("no element {id} to click") };
        self.cdp.clear_events();
        for kind in ["mouseMoved", "mousePressed", "mouseReleased"] {
            self.cdp.call(
                "Input.dispatchMouseEvent",
                json!({ "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1 }),
            )?;
        }
        // Links and submit buttons navigate; anything else returns after a short grace period.
        if self.cdp.wait_event(&NAVIGATION_STARTED, Duration::from_millis(1000))?.is_some() {
            self.wait_load()?;
        }
        Ok(())
    }

    fn type_text(&mut self, id: usize, text: &str) -> Result<()> {
        let ok =
            self.with_element(id, "if (el.tagName !== 'INPUT') return false; el.focus(); el.value = ''; return true;")?;
        if ok != Value::Bool(true) {
            bail!("element {id} is not a text field")
        }
        self.cdp.call("Input.insertText", json!({ "text": text }))?;
        Ok(())
    }

    fn back(&mut self) -> Result<()> {
        let h = self.cdp.call("Page.getNavigationHistory", json!({}))?;
        let current = h["currentIndex"].as_u64().unwrap_or(0) as usize;
        if current == 0 {
            return Ok(());
        }
        let Some(entry) = h["entries"][current - 1]["id"].as_u64() else { bail!("malformed navigation history: {h}") };
        self.cdp.clear_events();
        self.cdp.call("Page.navigateToHistoryEntry", json!({ "entryId": entry }))?;
        self.wait_load()
    }
}
