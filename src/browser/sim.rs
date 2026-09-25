//! In-process browser over the sandbox web: the same pages, links, form submission and
//! history as Chromium, without HTML or a network. Used to generate training data and in
//! tests; `tests/browser.rs` checks it element-for-element against Chromium.

use candle_core::{bail, Result};

use super::world::{self, Node, Target, World};
use super::{Browser, PageSnapshot, Role};

/// Origin of simulator URLs.
pub const SIM_ORIGIN: &str = "http://sim.local";

/// A browser tab over [`World`] pages.
#[derive(Debug, Clone)]
pub struct SimBrowser {
    origin: String,
    url: String,
    title: String,
    nodes: Vec<Node>,
    history: Vec<String>,
}

impl Default for SimBrowser {
    fn default() -> Self {
        Self::new()
    }
}

impl SimBrowser {
    pub fn new() -> Self {
        Self {
            origin: SIM_ORIGIN.into(),
            url: "about:blank".into(),
            title: String::new(),
            nodes: vec![],
            history: vec![],
        }
    }

    fn load(&mut self, url: &str) {
        self.url = url.to_string();
        if url == "about:blank" {
            self.title.clear();
            self.nodes.clear();
            return;
        }
        let (origin, path) = world::split_origin(url);
        let page = if origin == self.origin { World::page_at(path) } else { world::Page::NotFound };
        self.title = page.title();
        self.nodes = page.nodes();
    }

    fn node(&self, id: usize) -> Result<&Node> {
        match self.nodes.get(id) {
            Some(n) => Ok(n),
            None => bail!("no element {id} on {}", self.url),
        }
    }
}

impl Browser for SimBrowser {
    fn goto(&mut self, url: &str) -> Result<()> {
        // Like Chromium: the initial about:blank is a history entry, and navigating to the
        // current URL replaces the entry instead of adding one.
        if url != self.url {
            self.history.push(self.url.clone());
        }
        self.load(url);
        Ok(())
    }

    fn snapshot(&mut self) -> Result<PageSnapshot> {
        Ok(PageSnapshot {
            url: self.url.clone(),
            title: self.title.clone(),
            elements: self.nodes.iter().map(|n| n.element.clone()).collect(),
        })
    }

    fn click(&mut self, id: usize) -> Result<()> {
        let url = match &self.node(id)?.target {
            Target::None => return Ok(()),
            Target::Link(path) => format!("{}{path}", self.origin),
            Target::Submit(seed) => {
                // A GET form submits the value of its text field.
                let q = self.nodes.iter().find(|n| n.element.role == Role::Input).map(|n| n.element.value.as_str());
                format!("{}/w/{seed}/search?q={}", self.origin, world::url_encode(q.unwrap_or("")))
            }
        };
        self.goto(&url)
    }

    fn type_text(&mut self, id: usize, text: &str) -> Result<()> {
        let url = self.url.clone();
        let node = match self.nodes.get_mut(id) {
            Some(n) if n.element.role == Role::Input => n,
            _ => bail!("element {id} on {url} is not a text field"),
        };
        node.element.value = text.to_string();
        Ok(())
    }

    fn back(&mut self) -> Result<()> {
        if let Some(prev) = self.history.pop() {
            self.load(&prev);
        }
        Ok(())
    }
}
