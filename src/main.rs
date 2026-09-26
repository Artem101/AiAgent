//! `cog_engine` command-line interface.
//!
//! ```text
//! cog_engine demo  [--task sort] [--steps 1500]            train briefly, evaluate, benchmark
//! cog_engine train --out model.safetensors [--task sort|reverse|copy] [--steps N] [--preset tiny|small]
//!                  [--vocab 10] [--len 8] [--batch 64] [--lr 2e-3] [--seed 7] [--save-every N]
//!                  [--init ckpt [--start-step N]]
//!                  [--device cpu|cuda] [--compute auto|f32|f16|bf16] [--eval-every 500]
//! cog_engine infer --ckpt model.safetensors --prompt "3 1 4 1 5 9 2 6" [--planner mppi|mppi+gd]
//!                  [--solver heun|midpoint|euler] [--ode-steps 16] [--seed 0]
//! cog_engine bench [--ckpt model.safetensors] [--iters 200]
//! cog_engine serve --ckpt model.safetensors [--addr 127.0.0.1:7878]   (tokio, one prompt per line)
//!
//! Russian text and the browsing agent (BPE tokenizer `models/tokenizer_ru.bpe`):
//! cog_engine tokenizer --corpus data/ru/train.txt [--vocab 1024] [--out models/tokenizer_ru.bpe]
//! cog_engine train --task text|browser --corpus data/ru/train.txt [--valid data/ru/valid.txt] [--text-mix 0.25] …
//! cog_engine complete --ckpt model.safetensors --text "Москва — столица"   (next 8 tokens)
//! cog_engine text-eval --ckpt model.safetensors [--corpus data/ru/train.txt] [--valid data/ru/valid.txt] [--n 1000]
//! cog_engine agent --ckpt agent.safetensors --question "Сколько будет 5+5?" [--world 42]
//!                  [--browser chrome|sim] [--policy model|expert] [--calc python|rust] [--max-steps 12] [--headed]
//! cog_engine agent-eval --ckpt agent.safetensors [--episodes 100] [--browser sim|chrome] [--policy model|expert]
//!                  [--calc rust|python] [--split train|heldout] [--steps 1000]
//! cog_engine site  [--addr 127.0.0.1:8080]                  serve the sandbox web for a human browser
//!
//! The unified model — one network that browses, calculates and talks (docs/unified.md):
//! cog_engine train-unified [--preset base|tiny] [--steps 20000] [--batch 32] [--lr 1e-3] [--out models/agent.safetensors]
//!                  [--data data/ru20k|builtin] [--ud data/ru|none] [--save-every 500] [--eval-every N]
//!                  [--init ckpt [--start-step N]] [--d-model 256] [--layers 4] [--heads 4] [--copy 32]
//! cog_engine chat  [--ckpt models/agent.safetensors] [--say "Привет!|Сколько стоит лампа?"] [--think]
//!                  [--browser sim|chrome] [--temperature 0.7] [--search 1] [--world 42]
//! cog_engine unified-eval [--ckpt models/agent.safetensors] [--n 256] [--grammar 300] [--episodes 200]
//! cog_engine export-trajectories [--n 20000] [--out data/school/browser.jsonl]   (teacher episodes as JSONL chains)
//! cog_engine unified-params [--presets base,m,l]                                  (parameters by module)
//! (`agent` / `agent-eval` accept unified checkpoints as well)
//! ```

mod cli_unified;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{bail, DType, Device, Result};

use cog_engine::browser::agent::{self as browsing, EnginePolicy, ExpertPolicy, Policy};
use cog_engine::browser::chrome::{Chrome, ChromeOptions};
use cog_engine::browser::goal::{self, Goal, Split};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::{obs, Browser, SimBrowser, SiteServer};
use cog_engine::config::PlannerKind;
use cog_engine::data::Task;
use cog_engine::flow::SolverKind;
use cog_engine::kernels::simd::simd_level;
use cog_engine::text::{self, Bpe, Corpus};
use cog_engine::tools::{Calculator, PythonCalc, RustCalc};
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, TrainConfig};

/// Keeps freed tensor memory in the process: candle allocates a fresh buffer for every op, and
/// with glibc's defaults every large one is a new `mmap` whose pages fault in again (training
/// runs ~1.7× faster with this).
fn tune_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn mallopt(param: i32, value: i32) -> i32;
        }
        const M_TRIM_THRESHOLD: i32 = -1;
        const M_TOP_PAD: i32 = -2;
        const M_MMAP_THRESHOLD: i32 = -3;
        // SAFETY: plain configuration calls into glibc before any allocation-heavy work.
        unsafe {
            mallopt(M_MMAP_THRESHOLD, 32 << 20);
            mallopt(M_TRIM_THRESHOLD, i32::MAX);
            mallopt(M_TOP_PAD, 256 << 20);
        }
    }
}

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
    /// Planning horizon (`jepa.horizon`).
    horizon: usize,
    /// Gated readout pools (`ttt.readout_pools`).
    pools: usize,
    /// Copy-mechanism key width (`jepa.copy_dim`).
    copy: usize,
}

impl Meta {
    fn from_args(a: &Args) -> Result<Self> {
        let task = Task::parse(&a.get("task", "sort"))?;
        // Text tasks use the browser architecture: causal window, query readout, answer probe,
        // horizon 8 (see docs/browser.md and docs/reasoning.md).
        let defaults = EngineConfig::tiny(2, 1, 4);
        let c = if task.uses_text() { cog_engine::browser::engine_config("tiny")? } else { defaults };
        let (conv, last, probe, horizon, pools, copy) = (
            c.ttt.conv_width,
            c.ttt.readout_last,
            c.jepa.probe_weight,
            c.jepa.horizon,
            c.ttt.readout_pools,
            c.jepa.copy_dim,
        );
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
            horizon: a.num("horizon", horizon)?,
            pools: a.num("pools", pools)?,
            copy: a.num("copy", copy)?,
        })
    }

    fn config(&self) -> Result<EngineConfig> {
        let (vocab, n, l) = self.task.dims(self.vocab, self.len);
        let mut cfg = if self.task.uses_text() {
            cog_engine::browser::engine_config(&self.preset)? // incl. the latent reasoning settings
        } else {
            EngineConfig::preset(&self.preset, vocab, n, l)?
        };
        cfg.seed = self.seed;
        cfg.ttt.conv_width = self.conv;
        cfg.ttt.readout_last = self.readout_last;
        cfg.jepa.probe_weight = self.probe;
        cfg.jepa.horizon = self.horizon;
        cfg.ttt.readout_pools = self.pools;
        cfg.jepa.copy_dim = self.copy;
        Ok(cfg)
    }

    fn save(&self, ckpt: &str) -> Result<()> {
        let s = format!(
            "preset={}\nvocab={}\nlen={}\ntask={}\nseed={}\nconv={}\nreadout_last={}\nprobe={}\nhorizon={}\npools={}\ncopy={}\n",
            self.preset,
            self.vocab,
            self.len,
            self.task.name(),
            self.seed,
            self.conv,
            self.readout_last,
            self.probe,
            self.horizon,
            self.pools,
            self.copy
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
            horizon: if kv.contains_key("horizon") { num("horizon")? as usize } else { 4 },
            pools: if kv.contains_key("pools") { num("pools")? as usize } else { 0 },
            copy: if kv.contains_key("copy") { num("copy")? as usize } else { 0 },
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

/// Token ids, followed by their text for the text tasks.
fn fmt_task_tokens(task: Task, t: &[u32]) -> String {
    if task.uses_text() {
        format!("{}  ⟨{}⟩", fmt_tokens(t), obs::describe(t))
    } else {
        fmt_tokens(t)
    }
}

/// Inference overrides: `--planner mppi|mppi+gd`, `--tree <beam>` (0 = no tree search),
/// `--iters <MPPI iterations>`, `--solver`, `--ode-steps`. Without flags the checkpoint's
/// configuration is kept.
fn configure_engine(engine: &mut CognitiveEngine, a: &Args) -> Result<()> {
    if let Some(p) = a.opts.get("planner") {
        match p.as_str() {
            "mppi" => engine.set_planner(PlannerKind::Mppi),
            "mppi+gd" | "gd" => engine.set_planner(PlannerKind::MppiThenGradient),
            other => bail!("unknown planner '{other}' (mppi | mppi+gd)"),
        }
    }
    if a.opts.contains_key("tree") {
        engine.set_tree(a.num("tree", 0)?)?;
    }
    if a.opts.contains_key("iters") {
        engine.set_mppi_iterations(a.num("iters", 8)?);
    }
    if let Some(d) = a.opts.get("decoder") {
        engine.set_decoder(cog_engine::pipeline::ActionDecoder::parse(d)?)?;
    }
    let steps = a.num("ode-steps", engine.config().flow.solver.steps)?;
    engine.set_solver(SolverKind::parse(&a.get("solver", "heun"))?, steps);
    Ok(())
}

fn load_engine(a: &Args) -> Result<(CognitiveEngine, Meta)> {
    load_engine_from(a, &a.get("ckpt", "model.safetensors"))
}

/// The shipped agent, used by `agent` / `agent-eval` when no `--ckpt` is given.
const SHIPPED_AGENT: &str = cli_unified::SHIPPED;

fn load_engine_from(a: &Args, ckpt: &str) -> Result<(CognitiveEngine, Meta)> {
    let meta = Meta::load(ckpt)?;
    let model = CogModel::new(meta.config()?, &Device::Cpu)?;
    model.load(ckpt)?;
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
    tc.log_every = a.num("log-every", tc.log_every)?;
    tc.corpus = a.opts.get("corpus").map(Into::into);
    tc.text_mix = a.num("text-mix", tc.text_mix)?;
    tc.probe_goal = a.num::<u8>("probe-goal", tc.probe_goal as u8)? != 0;
    if task.uses_text() {
        tc.probe_states = cog_engine::browser::PROBE_STATES;
    }
    tc.probe_states = a.num("probe-states", tc.probe_states)?;
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
    if let Some(init) = a.opts.get("init") {
        model.load(init)?; // continue from a checkpoint of the same architecture
        println!("initialised from {init}");
    }
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
    if a.opts.contains_key("start-step") {
        // `--init <out>.step<N> --start-step N`: continue an interrupted run
        trainer.resume_at(a.num("start-step", 0)?);
        println!("resuming at step {}", trainer.step());
    }
    let out = a.get("out", if demo { "" } else { "model.safetensors" });
    // `--save-every N`: intermediate checkpoints `<out>.step<N>` for convergence studies.
    let every: usize = a.num("save-every", 0)?;
    let mut report = None;
    while trainer.step() < trainer.tc.steps {
        let until = if every > 0 { trainer.step() + every } else { trainer.tc.steps };
        report = trainer.run_until(until, |line| println!("{line}"))?.or(report);
        if every > 0 && !out.is_empty() && trainer.step() < trainer.tc.steps {
            let path = format!("{}.step{}", out.trim_end_matches(".safetensors"), trainer.step());
            trainer.model.save(&path)?;
            meta.save(&path)?;
            println!("saved {path}");
        }
    }
    if let Some(r) = &report {
        for (p, t, y) in &r.examples {
            let mark = if t == y { "✓" } else { "✗" };
            match meta.task {
                Task::Browser | Task::Text => println!(
                    "  {mark} ⟨{}⟩\n      → target ⟨{}⟩ | engine ⟨{}⟩",
                    obs::describe(p),
                    obs::describe(t),
                    obs::describe(y)
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
    if !out.is_empty() {
        trainer.model.save(&out)?;
        meta.save(&out)?;
        println!("saved {out} (+ {out}.cfg)");
    }
    if let (Some(train), Some(valid)) = (&trainer.sampler.corpus, a.opts.get("valid")) {
        let valid = Corpus::load(valid, text::ru())?;
        let mut engine = CognitiveEngine::from_model(&trainer.model)?;
        println!("{}", text::evaluate_lm(&mut engine, train, &valid, 1000, 99)?);
    }
    if meta.task == Task::Browser {
        // Closed-loop quality: whole episodes in the simulated browser, for training wordings
        // and for held-out wordings the model has never seen.
        let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&trainer.model)?);
        for (name, split) in [("train wordings", Split::Train), ("held-out wordings", Split::HeldOut)] {
            println!("agent, {name}: {}", browsing::step_accuracy(&mut policy, 1000, 99, split)?);
            let r =
                browsing::evaluate(&mut SimBrowser::new(), &mut policy, &mut RustCalc, SIM_ORIGIN, 200, 99, split, 12)?;
            println!("agent, {name} (200 episodes, simulated browser): {r}");
        }
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
    if let Some(target) = meta.task.apply(&prompt) {
        println!("target : {}   ({})", fmt_task_tokens(meta.task, &target), meta.task.name());
    }
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
/// `trace` records the decoded latent reasoning of every decision.
fn agent_policy(a: &Args, trace: bool) -> Result<Box<dyn Policy>> {
    match a.get("policy", "model").as_str() {
        "expert" => Ok(Box::new(ExpertPolicy)),
        "model" => {
            let ckpt = a.get("ckpt", SHIPPED_AGENT);
            if cog_engine::unified::UnifiedModel::is_checkpoint(&ckpt) {
                return cli_unified::policy(a, &ckpt, trace);
            }
            let (engine, meta) = load_engine_from(a, &ckpt)?;
            if meta.task != Task::Browser {
                bail!("the checkpoint was trained on '{}', not 'browser' (train with --task browser)", meta.task.name())
            }
            let policy = EnginePolicy::new(engine);
            Ok(Box::new(if trace { policy.with_trace() } else { policy }))
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
                local_only: !a.opts.contains_key("allow-internet"),
                ..Default::default()
            };
            let chrome = Chrome::launch(&opts)?;
            Ok((Box::new(chrome), server.origin(), Some(server)))
        }
        other => bail!("unknown browser '{other}' (chrome | sim)"),
    }
}

/// `--calc python` (a sandboxed `python3` worker) or `--calc rust` (its exact in-process
/// mirror).
fn agent_calculator(a: &Args, default: &str) -> Result<Box<dyn Calculator>> {
    match a.get("calc", default).as_str() {
        "python" | "py" => match PythonCalc::start() {
            Ok(p) => {
                println!("calculator: {} (sandboxed worker)", p.python().display());
                Ok(Box::new(p))
            }
            Err(e) => {
                println!("calculator: {e} — using the Rust mirror");
                Ok(Box::new(RustCalc))
            }
        },
        "rust" => Ok(Box::new(RustCalc)),
        other => bail!("unknown calculator '{other}' (python | rust)"),
    }
}

fn cmd_agent(a: &Args) -> Result<()> {
    let question = a.get("question", "Сколько стоит лампа?");
    let goal = Goal { spec: goal::recognize(&question), text: question.clone() };
    let world: u64 = a.num("world", 42)?;
    let mut policy = agent_policy(a, true)?;
    let mut calc = agent_calculator(a, "python")?;
    let (mut browser, origin, _server) = agent_browser(a, "chrome")?;
    println!("question: {question}\nworld   : {world} | site {origin}");
    let steps = a.num("max-steps", 12)?;
    let ep =
        browsing::run_episode(browser.as_mut(), policy.as_mut(), calc.as_mut(), &origin, world, &goal, steps, |s| {
            let action = s.action.as_ref().map_or_else(|| "?".to_string(), |x| x.to_string());
            println!("\n  {}", s.url);
            println!("    sees   : {}", obs::describe(&s.observation));
            if let Some(r) = &s.thoughts {
                print_reasoning(r);
            } else if let Some(g) = &s.plan {
                println!("    thinks : {}", g.plan);
            }
            println!(
                "    does   : {action:<28} ({:.1} ms){}",
                1e3 * s.think_time.as_secs_f64(),
                s.error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default()
            );
            if let Some((note, by)) = &s.tool {
                println!("    tool   : {by}: {} = {}", note.expr, note.result);
            }
        })?;
    println!();
    let verdict = match (ep.success(), ep.expected()) {
        (Some(true), _) => "✓".to_string(),
        (Some(false), Some(e)) => format!("✗ (expected {e})"),
        _ => "(question not recognised: cannot check)".to_string(),
    };
    match &ep.answer {
        Some(ans) => println!("answer  : {ans} {verdict} in {} steps", ep.steps.len()),
        None => println!("answer  : — (no answer in {} steps) {verdict}", ep.steps.len()),
    }
    Ok(())
}

/// A decoded thought: the action it stands for, or its raw tokens.
fn thought(tokens: &[u32]) -> String {
    match cog_engine::browser::Action::decode(tokens, text::ru()) {
        Some(a) => a.to_string(),
        None => format!("⟨{}⟩", obs::describe(tokens)),
    }
}

fn print_reasoning(r: &cog_engine::pipeline::Reasoning) {
    println!("    thinks : {}{}", r.stats, if r.refined { " (+ latent GD)" } else { "" });
    for (t, d) in r.depths.iter().enumerate() {
        let pruned = d.best_pruned.map_or(String::new(), |p| format!(", best pruned {p:.4}"));
        println!(
            "             depth {}: {} hypotheses, kept energy {:.4}…{:.4}{pruned}",
            t + 1,
            d.expanded,
            d.best_kept,
            d.worst_kept
        );
    }
    if !r.hypotheses.is_empty() {
        let hyps: Vec<String> = r.hypotheses.iter().map(|(e, t)| format!("{} (E {e:.4})", thought(t))).collect();
        println!("    leaves : {}", hyps.join(" | "));
    }
    if !r.chain.is_empty() {
        let mut chain: Vec<(usize, String)> = Vec::new();
        for (i, t) in r.chain.iter().enumerate() {
            let s = thought(t);
            if chain.last().map(|(_, l)| l) != Some(&s) {
                chain.push((i, s));
            }
        }
        let chain: Vec<String> = chain.into_iter().map(|(i, s)| format!("s{i}: {s}")).collect();
        println!("    chain  : {}", chain.join(" → "));
    }
}

fn cmd_agent_eval(a: &Args) -> Result<()> {
    let mut policy = agent_policy(a, false)?;
    let mut calc = agent_calculator(a, "rust")?;
    let (mut browser, origin, _server) = agent_browser(a, "sim")?;
    let episodes = a.num("episodes", 100)?;
    let seed = a.num("seed", 1)?;
    let splits: Vec<(&str, Split)> = match a.get("split", "both").as_str() {
        "train" => vec![("train wordings", Split::Train)],
        "heldout" => vec![("held-out wordings", Split::HeldOut)],
        "both" => vec![("train wordings", Split::Train), ("held-out wordings", Split::HeldOut)],
        other => bail!("unknown split '{other}' (train | heldout | both)"),
    };
    for (name, split) in splits {
        if a.opts.contains_key("steps") {
            println!("{name}: {}", browsing::step_accuracy(policy.as_mut(), a.num("steps", 1000)?, seed, split)?);
        }
        let r = browsing::evaluate(
            browser.as_mut(),
            policy.as_mut(),
            calc.as_mut(),
            &origin,
            episodes,
            seed,
            split,
            a.num("max-steps", 12)?,
        )?;
        println!("{} | {} | {name}: {r}", a.get("policy", "model"), a.get("browser", "sim"));
    }
    Ok(())
}

fn cmd_site(a: &Args) -> Result<()> {
    let server = SiteServer::start(&a.get("addr", "127.0.0.1:8080"))?;
    println!("sandbox web on {}/w/<world>/ (e.g. {}/w/42/) — Ctrl-C to stop", server.origin(), server.origin());
    server.wait();
    Ok(())
}

/// Trains the byte-level BPE on a corpus plus the browsing task's texts (pages, wordings,
/// actions and tool results), every text as a fragment with a leading space.
fn cmd_tokenizer(a: &Args) -> Result<()> {
    let corpus_path = a.get("corpus", "data/ru/train.txt");
    let corpus = std::fs::read_to_string(&corpus_path)
        .map_err(|e| candle_core::Error::Msg(format!("{corpus_path}: {e} (run scripts/fetch_ru_corpus.sh)")))?;
    let vocab: usize = a.num("vocab", 1024)?;
    let extra: usize = a.num("browser-texts", 20000)?;
    let mut rng = cog_engine::kernels::rng::Rng::new(a.num("seed", 1)?);
    let mut texts: Vec<String> = corpus.lines().map(String::from).collect();
    for _ in 0..extra {
        use cog_engine::browser::Action;
        let (goal, snapshot, note, action) = cog_engine::browser::data::raw(&mut rng, Split::Train);
        texts.push(goal.text);
        texts.extend(snapshot.elements.into_iter().flat_map(|e| [e.text, e.value]));
        if let Some(n) = note {
            texts.push(format!("{} = {}", n.expr, n.result));
        }
        if let Action::Calc { text } | Action::Answer { text } = action {
            texts.push(text);
        }
    }
    let texts: Vec<String> =
        texts.into_iter().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).map(|t| format!(" {t}")).collect();
    let t0 = Instant::now();
    let bpe = Bpe::train(texts.iter().map(String::as_str), &text::SPECIALS, vocab)?;
    let out = a.get("out", "models/tokenizer_ru.bpe");
    bpe.save(&out)?;
    let (bytes, tokens) =
        corpus.lines().take(2000).fold((0, 0), |(b, t), l| (b + l.chars().count(), t + text::fragment(&bpe, l).len()));
    println!(
        "{out}: {} tokens ({} specials + 256 bytes + {} merges) in {:.1?}; corpus: {:.2} characters per token",
        bpe.vocab_size(),
        bpe.num_specials(),
        bpe.num_merges(),
        t0.elapsed(),
        bytes as f64 / tokens.max(1) as f64
    );
    for s in [
        "Сколько стоит лампа?",
        "Что дешевле: клавиатура или мышь?",
        "Найди товар дешевле 12 ₽.",
        "Сколько будет 12 умножить на 3?",
        "Привет! Чем помочь?",
    ] {
        println!("  {s:<36} → {}", bpe.describe(&text::fragment(&bpe, s)));
    }
    Ok(())
}

/// Text continuation accuracy of a checkpoint on held-out sentences, next to baselines.
fn cmd_text_eval(a: &Args) -> Result<()> {
    let (mut engine, meta) = load_engine(a)?;
    if !meta.task.uses_text() {
        bail!("the checkpoint was trained on '{}', not on text", meta.task.name())
    }
    let train = Corpus::load(a.get("corpus", "data/ru/train.txt"), text::ru())?;
    let valid = Corpus::load(a.get("valid", "data/ru/valid.txt"), text::ru())?;
    println!("{}", text::evaluate_lm(&mut engine, &train, &valid, a.num("n", 1000)?, a.num("seed", 99)?)?);
    Ok(())
}

/// Text continuation: the next `L` tokens after `--text`.
fn cmd_complete(a: &Args) -> Result<()> {
    let (mut engine, meta) = load_engine(a)?;
    if !meta.task.uses_text() {
        bail!("the checkpoint was trained on '{}', not on text", meta.task.name())
    }
    let bpe = text::ru();
    let n = engine.config().max_prompt_len;
    let mut ctx = text::fragment(bpe, &a.get("text", "Москва — столица"));
    ctx.truncate(n - 1);
    let mut prompt = vec![text::PAD; n - 1 - ctx.len()];
    prompt.push(text::TEXT);
    prompt.extend(&ctx);
    let (out, g) = engine.generate(&prompt, a.num("seed", 0)?)?;
    let end = out.iter().position(|&t| bpe.is_special(t)).unwrap_or(out.len());
    println!("{}⟦{}⟧", a.get("text", "Москва — столица"), bpe.decode(&out[..end]));
    println!("tokens : {}\nplan   : {}", obs::describe(&out), g.plan);
    Ok(())
}

fn main() -> Result<()> {
    tune_allocator();
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
        "tokenizer" => cmd_tokenizer(&a),
        "complete" => cmd_complete(&a),
        "text-eval" => cmd_text_eval(&a),
        "train-unified" => cli_unified::cmd_train(&a),
        "chat" => cli_unified::cmd_chat(&a),
        "unified-eval" => cli_unified::cmd_eval(&a),
        "export-trajectories" => cli_unified::cmd_export(&a),
        "unified-params" => cli_unified::cmd_params(&a),
        other => bail!(
            "unknown command '{other}' (demo | train | infer | bench | serve | agent | agent-eval | site | tokenizer | complete | \
             text-eval | train-unified | chat | unified-eval | export-trajectories | unified-params)"
        ),
    }
}
