//! The agent from code: load (or briefly train) a unified checkpoint, then talk to it. Every line
//! is one episode in the sandbox web — headless Chromium, or the simulator when no Chromium is
//! installed — where the model may search, calculate or just answer; the conversation so far is
//! part of what it sees. Its latent reasoning and every calculator call are printed.
//!
//! ```text
//! cargo run --release --example browser_agent -- [checkpoint] ["реплика|реплика|…"]
//! cargo run --release --example browser_agent -- models/agent.safetensors "Привет!|Сколько стоит лампа?|Спасибо!"
//! cargo run --release --example browser_agent -- models/agent.safetensors "Сколько будет 5+5?"
//! ```

use candle_core::{Device, Result};
use cog_engine::browser::agent;
use cog_engine::browser::goal::{self, Goal};
use cog_engine::browser::obs::Turn;
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::{Browser, Chrome, SimBrowser, SiteServer};
use cog_engine::dialog::LanguageData;
use cog_engine::unified::{
    UnifiedConfig, UnifiedEngine, UnifiedModel, UnifiedPolicy, UnifiedTrainConfig, UnifiedTrainer,
};

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1);
    let lines = std::env::args().nth(2).unwrap_or_else(|| "Привет!|Сколько стоит лампа?|Спасибо!".into());

    // The architecture of a checkpoint is recorded next to it (`<ckpt>.cfg`).
    let model = match ckpt {
        Some(path) => UnifiedModel::load(&path, &Device::Cpu)?,
        None => {
            println!("no checkpoint given: training a tiny model for 200 steps on built-in data (it will be weak)");
            let model = UnifiedModel::new(UnifiedConfig::preset("tiny")?, &Device::Cpu)?;
            let tc = UnifiedTrainConfig { steps: 200, batch_size: 16, log_every: 50, ..Default::default() };
            let mut trainer = UnifiedTrainer::new(model, tc, LanguageData::builtin())?;
            trainer.run_until(200, |line| println!("{line}"))?;
            trainer.model
        }
    };
    let mut policy = UnifiedPolicy::new(UnifiedEngine::new(model)?).with_trace();

    // Real Chromium if available, otherwise the DOM-identical simulator.
    let server = SiteServer::start("127.0.0.1:0")?;
    let (mut browser, origin): (Box<dyn Browser>, String) = match Chrome::find_executable() {
        Some(_) => (Box::new(Chrome::launch_default()?), server.origin()),
        None => (Box::new(SimBrowser::new()), SIM_ORIGIN.to_string()),
    };
    // `CALC` actions go to a sandboxed python3 (or its exact Rust mirror without Python).
    let mut calc = cog_engine::tools::default_calculator();
    let mut history: Vec<Turn> = Vec::new();
    for line in lines.split('|').map(str::trim).filter(|l| !l.is_empty()) {
        // The model reads the raw text; the recognised spec is only used to check the answer.
        let goal = Goal { spec: goal::recognize(line), text: line.to_string() };
        println!("\n> {line}");
        let ep = agent::run_dialog_episode(
            browser.as_mut(),
            &mut policy,
            calc.as_mut(),
            &origin,
            42,
            &history,
            &goal,
            12,
            |s| {
                let action = s.action.as_ref().map_or("?".into(), |a| a.to_string());
                println!("  {:<48} {action}", s.url);
                if let Some((note, by)) = &s.tool {
                    println!("      {by}: {} = {}", note.expr, note.result);
                }
                if let Some(r) = &s.thoughts {
                    println!("      {}", r.stats);
                }
            },
        )?;
        let answer = ep.answer.clone().unwrap_or_else(|| "…".into());
        match ep.success() {
            Some(ok) => println!("< {answer} ({})", if ok { "correct" } else { "wrong" }),
            None => println!("< {answer}"),
        }
        history.push(Turn::user(line));
        history.push(Turn::bot(answer));
    }
    Ok(())
}
