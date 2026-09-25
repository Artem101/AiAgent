//! Train a tiny engine on `sort`, evaluate it and answer a few prompts.
//!
//! ```text
//! cargo run --release --example quickstart -- [steps]
//! ```

use candle_core::{Device, Result};
use cog_engine::data::Task;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, TrainConfig};

fn main() -> Result<()> {
    let steps = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(600);

    // vocabulary 10, prompts of 8 tokens, answers of 8 tokens
    let model = CogModel::new(EngineConfig::tiny(10, 8, 8), &Device::Cpu)?;
    let mut tc = TrainConfig::quick(Task::Sort);
    tc.steps = steps;
    tc.eval_every = 0; // evaluate once at the end instead
    let mut trainer = Trainer::new(model, tc)?;
    trainer.run(|line| println!("{line}"))?;
    println!("{}", trainer.evaluate(64, 1234)?);

    // Pack the weights (bf16) and pre-allocate every buffer once.
    let mut engine = CognitiveEngine::from_model(&trainer.model)?;
    let mut out = [0u32; 8];
    for prompt in [[3, 1, 4, 1, 5, 9, 2, 6], [9, 8, 7, 6, 5, 4, 3, 2]] {
        let g = engine.generate_into(&prompt, 0, &mut out)?; // no heap allocation
        println!("{prompt:?} → {out:?} | plan energy {:.4} | {:?}", g.plan.energy, g.timings.total());
    }
    Ok(())
}
