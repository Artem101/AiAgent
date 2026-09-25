//! Browsing agent from code: load (or briefly train) a browser checkpoint, then let the engine
//! answer a Russian question by searching the sandbox web in headless Chromium (or the
//! simulator when no Chromium is installed), printing its latent reasoning at every step and
//! every call of the calculator tool.
//!
//! ```text
//! cargo run --release --example browser_agent -- [checkpoint] ["вопрос"]
//! cargo run --release --example browser_agent -- models/browser_agent.safetensors "Что дешевле: лампа или стул?"
//! cargo run --release --example browser_agent -- models/browser_agent.safetensors "Сколько будет 5+5?"
//! ```

use candle_core::{Device, Result};
use cog_engine::browser::agent::{self, EnginePolicy};
use cog_engine::browser::goal::{self, Goal};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::{self, Browser, Chrome, SimBrowser, SiteServer};
use cog_engine::data::Task;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, TrainConfig};

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1);
    let question = std::env::args().nth(2).unwrap_or_else(|| "Сколько стоит лампа?".into());

    // The architecture of a checkpoint is recorded next to it (`<ckpt>.cfg`, written by the CLI).
    let meta = ckpt.as_ref().and_then(|p| std::fs::read_to_string(format!("{p}.cfg")).ok()).unwrap_or_default();
    let field = |k: &str| meta.lines().find_map(|l| l.strip_prefix(&format!("{k}=")).map(str::to_string));
    let mut cfg = browser::engine_config(&field("preset").unwrap_or_else(|| "tiny".into()))?;
    if let Some(copy) = field("copy").and_then(|c| c.parse().ok()) {
        cfg.jepa.copy_dim = copy;
    }
    let model = CogModel::new(cfg, &Device::Cpu)?;
    let model = match ckpt {
        Some(path) => {
            model.load(&path)?;
            model
        }
        None => {
            println!("no checkpoint given: training for 300 steps (the agent will be weak)");
            let mut tc = TrainConfig::quick(Task::Browser);
            tc.steps = 300;
            tc.eval_every = 0;
            let mut trainer = Trainer::new(model, tc)?;
            trainer.run(|line| println!("{line}"))?;
            trainer.model
        }
    };
    let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&model)?).with_trace();

    // Real Chromium if available, otherwise the DOM-identical simulator.
    let server = SiteServer::start("127.0.0.1:0")?;
    let (mut browser, origin): (Box<dyn Browser>, String) = match Chrome::find_executable() {
        Some(_) => (Box::new(Chrome::launch_default()?), server.origin()),
        None => (Box::new(SimBrowser::new()), SIM_ORIGIN.to_string()),
    };

    // The model reads the raw text; the recognised spec is only used to check the answer.
    let goal = Goal { spec: goal::recognize(&question), text: question.clone() };
    println!("\n{question}");
    // `CALC` actions go to a sandboxed python3 (or its exact Rust mirror without Python).
    let mut calc = cog_engine::tools::default_calculator();
    let ep = agent::run_episode(browser.as_mut(), &mut policy, calc.as_mut(), &origin, 42, &goal, 12, |s| {
        let action = s.action.as_ref().map_or("?".into(), |a| a.to_string());
        println!("  {:<56} {action}", s.url);
        if let Some((note, by)) = &s.tool {
            println!("      {by}: {} = {}", note.expr, note.result);
        }
        if let Some(r) = &s.thoughts {
            println!("      {}", r.stats);
        }
    })?;
    match (&ep.answer, ep.success()) {
        (Some(a), Some(ok)) => println!("answer: {a} ({})", if ok { "correct" } else { "wrong" }),
        (Some(a), None) => println!("answer: {a}"),
        (None, _) => println!("no answer within 12 steps"),
    }
    Ok(())
}
