//! `cog_engine` command-line interface.
//!
//! ```text
//! cog_engine demo  [--task sort] [--steps 1500]            train briefly, evaluate, benchmark
//! cog_engine train --out model.safetensors [--task sort|reverse|copy] [--steps N] [--preset tiny|small]
//!                  [--vocab 10] [--len 8] [--batch 64] [--lr 2e-3] [--seed 7]
//!                  [--device cpu|cuda] [--compute auto|f32|f16|bf16] [--eval-every 500]
//! cog_engine infer --ckpt model.safetensors --prompt "3 1 4 1 5 9 2 6" [--planner mppi|mppi+gd]
//!                  [--solver heun|midpoint|euler] [--ode-steps 16] [--seed 0]
//! cog_engine bench [--ckpt model.safetensors] [--iters 200]
//! cog_engine serve --ckpt model.safetensors [--addr 127.0.0.1:7878]   (tokio, one prompt per line)
//!
//! browsing agent (train with `--task browser`):
//! cog_engine agent --ckpt agent.safetensors --question "сколько стоит лампа" [--world 42]
//!                  [--browser chrome|sim] [--policy model|expert] [--max-steps 10] [--headed]
//! cog_engine agent-eval --ckpt agent.safetensors [--episodes 100] [--browser sim|chrome] [--policy model|expert]
//! cog_engine site  [--addr 127.0.0.1:8080]                  serve the sandbox web for a human browser
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{bail, DType, Device, Result};

use cog_engine::browser::agent::{self as browsing, EnginePolicy, ExpertPolicy, Policy};
use cog_engine::browser::chrome::{Chrome, ChromeOptions};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::vocab::{self, token_str};
use cog_engine::browser::{Browser, Goal, SimBrowser, SiteServer};
use cog_engine::config::PlannerKind;
use cog_engine::data::Task;
use cog_engine::flow::SolverKind;
use cog_engine::kernels::simd::simd_level;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, TrainConfig};

struct Args {
    cmd: String,
    opts: HashMap<String, String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut it = std::env::args().skip(1).peekable();
        let mut cmd = "demo".to_string();
        let mut opts = HashMap::new();
        let mut first = true;
        while let Some(a) = it.next() {
            if let Some(k) = a.strip_prefix("--") {
                // `--flag` without a value (e.g. `--headed`) when followed by another option.
                let v = it.next_if(|n| !n.starts_with("--")).unwrap_or_default();
                opts.insert(k.to_string(), v);
            } else if first {
                cmd = a;
            } else {
                bail!("unexpected argument '{a}'")
            }
            first = false;
        }
        Ok(Self { cmd, opts })
    }

    fn get(&self, k: &str, default: &str) -> String {
        self.opts.get(k).cloned().unwrap_or_else(|| default.to_string())
    }

    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T> {
        match self.opts.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| candle_core::Error::Msg(format!("--{k}: cannot parse '{v}'"))),
        }
    }
}

/// Architecture metadata stored next to a checkpoint (`<ckpt>.cfg`).
struct Meta {
    preset: String,
    vocab: usize,
    len: usize,
    task: Task,
    seed: u64,
    /// TTT causal window width (`ttt.conv_width`).
    conv: usize,
    /// Final TTT outputs in the readout (`ttt.readout_last`).
    readout_last: usize,
    /// Answer-probe loss weight (`jepa.probe_weight`; > 0 adds the probe head).
    probe: f64,
}

impl Meta {
    fn from_args(a: &Args) -> Result<Self> {
        let task = Task::parse(&a.get("task", "sort"))?;
        // The browsing task needs the causal window and the query readout (see docs/browser.md).
        let (conv, last, probe) = match task {
            Task::Browser => {
                (cog_engine::browser::CONV_WIDTH, cog_engine::browser::READOUT_LAST, cog_engine::browser::PROBE_WEIGHT)
            }
            _ => (1, 0, 0.0),
        };
        // Fixed-size tasks record their real dimensions (`Task::dims` ignores the flags).
        let (vocab, len, _) = task.dims(a.num("vocab", 10)?, a.num("len", 8)?);
        Ok(Self {
            preset: a.get("preset", "tiny"),
            vocab,
            len,
            task,
            seed: a.num("seed", 7)?,
            conv: a.num("conv", conv)?,
            readout_last: a.num("readout-last", last)?,
            probe: a.num("probe", probe)?,
        })
    }

    fn config(&self) -> Result<EngineConfig> {
        let (vocab, n, l) = self.task.dims(self.vocab, self.len);
        let mut cfg = EngineConfig::preset(&self.preset, vocab, n, l)?;
        cfg.seed = self.seed;
        cfg.ttt.conv_width = self.conv;
        cfg.ttt.readout_last = self.readout_last;
        cfg.jepa.probe_weight = self.probe;
        Ok(cfg)
    }

    fn save(&self, ckpt: &str) -> Result<()> {
        let s = format!(
            "preset={}\nvocab={}\nlen={}\ntask={}\nseed={}\nconv={}\nreadout_last={}\nprobe={}\n",
            self.preset,
            self.vocab,
            self.len,
            self.task.name(),
            self.seed,
            self.conv,
            self.readout_last,
            self.probe
        );
        std::fs::write(format!("{ckpt}.cfg"), s).map_err(candle_core::Error::wrap)
    }

    fn load(ckpt: &str) -> Result<Self> {
        let text = std::fs::read_to_string(format!("{ckpt}.cfg")).map_err(candle_core::Error::wrap)?;
        let kv: HashMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
        let get =
            |k: &str| kv.get(k).copied().ok_or_else(|| candle_core::Error::Msg(format!("{ckpt}.cfg: missing '{k}'")));
        let num = |k: &str| -> Result<u64> { get(k)?.parse().map_err(candle_core::Error::wrap) };
        Ok(Self {
            preset: get("preset")?.to_string(),
            vocab: num("vocab")? as usize,
            len: num("len")? as usize,
            task: Task::parse(get("task")?)?,
            seed: num("seed")?,
            // absent in checkpoints written before these options existed
            conv: if kv.contains_key("conv") { num("conv")? as usize } else { 1 },
            readout_last: if kv.contains_key("readout_last") { num("readout_last")? as usize } else { 0 },
            probe: match kv.get("probe") {
                Some(v) => v.parse().map_err(candle_core::Error::wrap)?,
                None => 0.0,
            },
        })
    }
}

fn parse_prompt(s: &str) -> Result<Vec<u32>> {
    s.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().map_err(|_| candle_core::Error::Msg(format!("bad token '{t}'"))))
        .collect()
}

fn fmt_tokens(t: &[u32]) -> String {
    t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

/// Token ids, followed by their words for the browsing task.
fn fmt_task_tokens(task: Task, t: &[u32]) -> String {
    match task {
        Task::Browser => format!("{}  ⟨{}⟩", fmt_tokens(t), vocab::describe(t)),
        _ => fmt_tokens(t),
    }
}

fn configure_engine(engine: &mut CognitiveEngine, a: &Args) -> Result<()> {
    match a.get("planner", "mppi").as_str() {
        "mppi" => engine.set_planner(PlannerKind::Mppi),
        "mppi+gd" | "gd" => engine.set_planner(PlannerKind::MppiThenGradient),
        other => bail!("unknown planner '{other}' (mppi | mppi+gd)"),
    }
    let steps = a.num("ode-steps", engine.config().flow.solver.steps)?;
    engine.set_solver(SolverKind::parse(&a.get("solver", "heun"))?, steps);
    Ok(())
}

fn load_engine(a: &Args) -> Result<(CognitiveEngine, Meta)> {
    let ckpt = a.get("ckpt", "model.safetensors");
    let meta = Meta::load(&ckpt)?;
    let model = CogModel::new(meta.config()?, &Device::Cpu)?;
    model.load(&ckpt)?;
    let mut engine = CognitiveEngine::from_model(&model)?;
    configure_engine(&mut engine, a)?;
    Ok((engine, meta))
}

fn train_config(a: &Args, task: Task) -> Result<TrainConfig> {
    let mut tc = TrainConfig::quick(task);
    tc.steps = a.num("steps", tc.steps)?;
    tc.batch_size = a.num("batch", tc.batch_size)?;
    tc.lr = a.num("lr", tc.lr)?;
    tc.eval_every = a.num("eval-every", tc.eval_every)?;
    tc.compute_dtype = match a.get("compute", "auto").as_str() {
        "auto" => None,
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        other => bail!("unknown compute dtype '{other}' (auto | f32 | f16 | bf16)"),
    };
    Ok(tc)
}

/// Training device: `cpu` (default) or `cuda` (needs the `cuda` cargo feature).
fn train_device(a: &Args) -> Result<Device> {
    match a.get("device", "cpu").as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0),
        other => bail!("unknown device '{other}' (cpu | cuda)"),
    }
}

fn cmd_train(a: &Args, demo: bool) -> Result<()> {
    let meta = Meta::from_args(a)?;
    let cfg = meta.config()?;
    let tc = train_config(a, meta.task)?;
    let device = train_device(a)?;
    let model = CogModel::new(cfg.clone(), &device)?;
    println!(
        "cog_engine | task={} vocab={} N={} L={} | preset={} | {} params | train {:?}/{:?} | simd={} | rayon threads={}",
        meta.task.name(),
        cfg.vocab_size,
        cfg.max_prompt_len,
        cfg.answer_len(),
        meta.preset,
        model.num_params(),
        device.location(),
        tc.resolved_compute_dtype(&device),
        simd_level(),
        rayon::current_num_threads()
    );
    let mut trainer = Trainer::new(model, tc)?;
    let report = trainer.run(|line| println!("{line}"))?;
    if let Some(r) = &report {
        for (p, t, y) in &r.examples {
            let mark = if t == y { "✓" } else { "✗" };
            match meta.task {
                Task::Browser => println!(
                    "  {mark} page ⟨{}⟩\n      → expert ⟨{}⟩ | engine ⟨{}⟩",
                    vocab::describe(p),
                    vocab::describe(t),
                    vocab::describe(y)
                ),
                _ => println!(
                    "  {mark} prompt [{}] → target [{}] | engine [{}]",
                    fmt_tokens(p),
                    fmt_tokens(t),
                    fmt_tokens(y)
                ),
            }
        }
    }
    let out = a.get("out", if demo { "" } else { "model.safetensors" });
    if !out.is_empty() {
        trainer.model.save(&out)?;
        meta.save(&out)?;
        println!("saved {out} (+ {out}.cfg)");
    }
    if meta.task == Task::Browser {
        // Closed-loop quality: whole episodes in the simulated browser.
        let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&trainer.model)?);
        println!("agent {}", browsing::step_accuracy(&mut policy, 1000, 99)?);
        let r = browsing::evaluate(&mut SimBrowser::new(), &mut policy, SIM_ORIGIN, 200, 99, 10)?;
        println!("agent (200 episodes, simulated browser): {r}");
    }
    if demo {
        bench(&mut CognitiveEngine::from_model(&trainer.model)?, 200)?;
    }
    Ok(())
}

fn cmd_infer(a: &Args) -> Result<()> {
    let (mut engine, meta) = load_engine(a)?;
    let prompt = parse_prompt(&a.get("prompt", ""))?;
    let n = engine.config().max_prompt_len;
    if prompt.len() != n {
        bail!("the checkpoint was trained on prompts of length {n} (got {})", prompt.len())
    }
    let (out, g) = engine.generate(&prompt, a.num("seed", 0)?)?;
    println!("prompt : {}", fmt_task_tokens(meta.task, &prompt));
    println!("output : {}", fmt_task_tokens(meta.task, &out));
    println!("target : {}   ({})", fmt_task_tokens(meta.task, &meta.task.apply(&prompt)), meta.task.name());
    println!(
        "plan   : energy {:.4} (warm start {:.4}), ESS {:.1}{}",
        g.plan.energy,
        g.plan.initial_energy,
        g.plan.effective_samples,
        if g.refined { ", refined by latent GD" } else { "" }
    );
    println!(
        "latency: encode {:?} | plan {:?} | decode {:?} | total {:?}",
        g.timings.encode,
        g.timings.plan,
        g.timings.decode,
        g.timings.total()
    );
    Ok(())
}

fn bench(engine: &mut CognitiveEngine, iters: usize) -> Result<()> {
    let (len, vocab) = (engine.config().max_prompt_len, engine.config().vocab_size);
    let m = engine.memory();
    println!(
        "memory : weights {:.1} KiB ({:?}) | arena {:.1} KiB in {} buffers | context state W_fast {} B (independent of N)",
        m.weight_bytes as f64 / 1024.0,
        engine.weight_dtype(),
        m.arena_bytes as f64 / 1024.0,
        m.arena_buffers,
        m.context_state_bytes
    );
    let prompt: Vec<u32> = (0..len).map(|i| ((i * 7 + 3) % vocab) as u32).collect();
    let mut out = vec![0u32; engine.config().answer_len()];
    engine.generate_into(&prompt, 0, &mut out)?; // warm-up
    let (mut enc, mut plan, mut dec) = (0f64, 0f64, 0f64);
    let t0 = Instant::now();
    for i in 0..iters {
        let g = engine.generate_into(&prompt, i as u64, &mut out)?;
        enc += g.timings.encode.as_secs_f64();
        plan += g.timings.plan.as_secs_f64();
        dec += g.timings.decode.as_secs_f64();
    }
    let n = iters as f64;
    let s = &engine.config().flow.solver;
    println!(
        "latency: encode {:.1} µs | plan {:.1} µs (MPPI {}×{}×H={}) | decode {:.1} µs ({} {:?} steps, {} NFE, L={} in parallel) | {:.0} queries/s",
        1e6 * enc / n,
        1e6 * plan / n,
        engine.config().planner.num_samples,
        engine.config().planner.iterations,
        engine.config().jepa.horizon,
        1e6 * dec / n,
        s.steps,
        s.solver,
        s.steps * s.solver.evals_per_step(),
        engine.config().answer_len(),
        n / t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_bench(a: &Args) -> Result<()> {
    let iters = a.num("iters", 200)?;
    if a.opts.contains_key("ckpt") {
        let (mut engine, _) = load_engine(a)?;
        bench(&mut engine, iters)
    } else {
        let meta = Meta::from_args(a)?;
        let model = CogModel::new(meta.config()?, &Device::Cpu)?;
        let mut engine = CognitiveEngine::from_model(&model)?;
        configure_engine(&mut engine, a)?;
        println!("(untrained weights — latency / memory only)");
        bench(&mut engine, iters)
    }
}

fn cmd_serve(a: &Args) -> Result<()> {
    let (engine, _) = load_engine(a)?;
    let len = engine.config().max_prompt_len;
    let addr = a.get("addr", "127.0.0.1:7878");
    let engine = Arc::new(Mutex::new(engine));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(candle_core::Error::wrap)?;
    rt.block_on(serve(engine, len, addr))
}

async fn serve(engine: Arc<Mutex<CognitiveEngine>>, len: usize, addr: String) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(candle_core::Error::wrap)?;
    println!("serving on {addr}: send {len} space-separated token ids per line");
    loop {
        let (sock, peer) = listener.accept().await.map_err(candle_core::Error::wrap)?;
        let engine = engine.clone();
        tokio::spawn(async move {
            let (rd, mut wr) = sock.into_split();
            let mut lines = BufReader::new(rd).lines();
            let mut seed = 0u64;
            while let Ok(Some(line)) = lines.next_line().await {
                seed += 1;
                let engine = engine.clone();
                // Inference is CPU-bound: run it off the async reactor.
                let reply = tokio::task::spawn_blocking(move || -> Result<String> {
                    let prompt = parse_prompt(&line)?;
                    if prompt.len() != len {
                        bail!("expected {len} tokens, got {}", prompt.len())
                    }
                    let mut engine = engine.lock().map_err(|_| candle_core::Error::Msg("engine poisoned".into()))?;
                    let (out, g) = engine.generate(&prompt, seed)?;
                    Ok(format!("{} | {:?}", fmt_tokens(&out), g.timings.total()))
                })
                .await;
                // One reply per line: keep only the first line of an error (candle errors can
                // carry a multi-line backtrace when RUST_BACKTRACE is set).
                let first_line = |e: String| e.lines().next().unwrap_or_default().to_string();
                let text = match reply {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => format!("error: {}", first_line(e.to_string())),
                    Err(e) => format!("error: {}", first_line(e.to_string())),
                };
                if wr.write_all(format!("{text}\n").as_bytes()).await.is_err() {
                    break;
                }
            }
            eprintln!("{peer} disconnected");
        });
    }
}

/// `--policy model` (default, needs `--ckpt`) or `--policy expert` (the scripted teacher).
fn agent_policy(a: &Args) -> Result<Box<dyn Policy>> {
    match a.get("policy", "model").as_str() {
        "expert" => Ok(Box::new(ExpertPolicy)),
        "model" => {
            let (engine, meta) = load_engine(a)?;
            if meta.task != Task::Browser {
                bail!("the checkpoint was trained on '{}', not 'browser' (train with --task browser)", meta.task.name())
            }
            Ok(Box::new(EnginePolicy::new(engine)))
        }
        other => bail!("unknown policy '{other}' (model | expert)"),
    }
}

/// The browser to act in and the origin of the sandbox web it sees. The site server (for
/// Chromium) lives as long as the returned guard.
fn agent_browser(a: &Args, default: &str) -> Result<(Box<dyn Browser>, String, Option<SiteServer>)> {
    match a.get("browser", default).as_str() {
        "sim" => Ok((Box::new(SimBrowser::new()), SIM_ORIGIN.to_string(), None)),
        "chrome" | "chromium" => {
            let server = SiteServer::start(&a.get("site-addr", "127.0.0.1:0"))?;
            let opts = ChromeOptions {
                executable: a.opts.get("chrome").map(Into::into),
                headless: !a.opts.contains_key("headed"),
                ..Default::default()
            };
            let chrome = Chrome::launch(&opts)?;
            Ok((Box::new(chrome), server.origin(), Some(server)))
        }
        other => bail!("unknown browser '{other}' (chrome | sim)"),
    }
}

fn cmd_agent(a: &Args) -> Result<()> {
    let question = a.get("question", "what is the price of the lamp?");
    let goal = Goal::parse(&question)?;
    let world: u64 = a.num("world", 42)?;
    let mut policy = agent_policy(a)?;
    let (mut browser, origin, _server) = agent_browser(a, "chrome")?;
    println!("question: {question}\ngoal    : {goal} | world {world} | site {origin}");
    let ep =
        browsing::run_episode(browser.as_mut(), policy.as_mut(), &origin, world, goal, a.num("max-steps", 10)?, |s| {
            let action = s.action.map_or_else(|| "?".to_string(), |x| x.to_string());
            println!("\n  {}", s.url);
            println!("    sees   : {}", vocab::describe(&s.observation));
            println!(
                "    does   : {action:<24} ({:.1} ms){}",
                1e3 * s.think_time.as_secs_f64(),
                s.error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default()
            );
        })?;
    println!();
    match ep.answer {
        Some(ans) => println!(
            "answer  : {} {} (page says {}) in {} steps",
            token_str(ans),
            if ep.success() { "✓" } else { "✗" },
            token_str(ep.truth),
            ep.steps.len()
        ),
        None => println!("answer  : — (no answer in {} steps; the page says {})", ep.steps.len(), token_str(ep.truth)),
    }
    Ok(())
}

fn cmd_agent_eval(a: &Args) -> Result<()> {
    let mut policy = agent_policy(a)?;
    let (mut browser, origin, _server) = agent_browser(a, "sim")?;
    let episodes = a.num("episodes", 100)?;
    if a.opts.contains_key("steps") {
        println!("{}", browsing::step_accuracy(policy.as_mut(), a.num("steps", 1000)?, a.num("seed", 1)?)?);
    }
    let r = browsing::evaluate(
        browser.as_mut(),
        policy.as_mut(),
        &origin,
        episodes,
        a.num("seed", 1)?,
        a.num("max-steps", 10)?,
    )?;
    println!("{} | {} | {r}", a.get("policy", "model"), a.get("browser", "sim"));
    Ok(())
}

fn cmd_site(a: &Args) -> Result<()> {
    let server = SiteServer::start(&a.get("addr", "127.0.0.1:8080"))?;
    println!("sandbox web on {}/w/<world>/ (e.g. {}/w/42/) — Ctrl-C to stop", server.origin(), server.origin());
    server.wait();
    Ok(())
}

fn main() -> Result<()> {
    let a = Args::parse()?;
    match a.cmd.as_str() {
        "demo" => cmd_train(&a, true),
        "train" => cmd_train(&a, false),
        "infer" => cmd_infer(&a),
        "bench" => cmd_bench(&a),
        "serve" => cmd_serve(&a),
        "agent" => cmd_agent(&a),
        "agent-eval" => cmd_agent_eval(&a),
        "site" => cmd_site(&a),
        other => bail!("unknown command '{other}' (demo | train | infer | bench | serve | agent | agent-eval | site)"),
    }
}
