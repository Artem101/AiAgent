//! Browsing agent from code: load (or briefly train) a browser checkpoint, then let the engine
//! answer a question by searching the sandbox web in headless Chromium (or the simulator when
//! no Chromium is installed).
//!
//! ```text
//! cargo run --release --example browser_agent -- [checkpoint] ["question"]
//! cargo run --release --example browser_agent -- models/browser_agent.safetensors "какого цвета велосипед"
//! ```

use candle_core::{Device, Result};
use cog_engine::browser::agent::{self, EnginePolicy};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::vocab::token_str;
use cog_engine::browser::{self, Browser, Chrome, Goal, SimBrowser, SiteServer};
use cog_engine::data::Task;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, TrainConfig};

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1);
    let question = std::env::args().nth(2).unwrap_or_else(|| "what is the price of the lamp?".into());

    let model = CogModel::new(browser::engine_config("tiny")?, &Device::Cpu)?;
    let model = match ckpt {
        Some(path) => {
            model.load(&path)?;
            model
        }
        None => {
            println!("no checkpoint given: training for 1000 steps (~2.5 min; the agent will still be weak)");
            let mut tc = TrainConfig::quick(Task::Browser);
            tc.steps = 1000;
            tc.eval_every = 0;
            let mut trainer = Trainer::new(model, tc)?;
            trainer.run(|line| println!("{line}"))?;
            trainer.model
        }
    };
    let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&model)?);

    // Real Chromium if available, otherwise the DOM-identical simulator.
    let server = SiteServer::start("127.0.0.1:0")?;
    let (mut browser, origin): (Box<dyn Browser>, String) = match Chrome::find_executable() {
        Some(_) => (Box::new(Chrome::launch_default()?), server.origin()),
        None => (Box::new(SimBrowser::new()), SIM_ORIGIN.to_string()),
    };

    let goal = Goal::parse(&question)?;
    println!("\n{question}  →  goal: {goal}");
    let ep = agent::run_episode(browser.as_mut(), &mut policy, &origin, 42, goal, 10, |s| {
        println!("  {:<48} {}", s.url, s.action.map_or("?".into(), |a| a.to_string()));
    })?;
    match ep.answer {
        Some(a) => println!("answer: {} ({})", token_str(a), if ep.success() { "correct" } else { "wrong" }),
        None => println!("no answer within 10 steps"),
    }
    Ok(())
}
