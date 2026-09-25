//! End-to-end: training reduces the joint loss, checkpoints round-trip, generation is
//! deterministic for a given seed.

use candle_core::{Device, Result};
use cog_engine::data::Task;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, TrainConfig};

fn tiny() -> EngineConfig {
    let mut cfg = EngineConfig::tiny(6, 4, 4);
    cfg.planner.num_samples = 32;
    cfg.planner.iterations = 3;
    cfg.flow.solver.steps = 4;
    cfg
}

#[test]
fn train_save_load_generate() -> Result<()> {
    let dev = Device::Cpu;
    let mut tc = TrainConfig::quick(Task::Sort);
    tc.steps = 60;
    tc.batch_size = 32;
    tc.warmup = 10;
    tc.eval_every = 0;
    let mut trainer = Trainer::new(CogModel::new(tiny(), &dev)?, tc)?;

    let mut losses = Vec::new();
    for _ in 0..60 {
        losses.push(trainer.train_step()?.0.total);
    }
    let head: f32 = losses[..10].iter().sum::<f32>() / 10.0;
    let tail: f32 = losses[50..].iter().sum::<f32>() / 10.0;
    assert!(losses.iter().all(|l| l.is_finite()));
    assert!(tail < 0.7 * head, "joint loss did not decrease: {head} → {tail}");

    let report = trainer.evaluate(8, 1)?;
    assert_eq!(report.engine.len(), 3);
    assert!(report.engine.iter().all(|e| e.mean_plan_energy <= e.mean_initial_energy + 1e-6));

    // checkpoint round-trip → identical packed engine outputs
    let dir = std::env::temp_dir().join(format!("cog_engine_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(candle_core::Error::wrap)?;
    let path = dir.join("model.safetensors");
    trainer.model.save(&path)?;
    let restored = CogModel::new(tiny(), &dev)?;
    restored.load(&path)?;
    std::fs::remove_dir_all(&dir).ok();

    let prompt = [3u32, 1, 5, 0];
    let mut a = CognitiveEngine::from_model(&trainer.model)?;
    let mut b = CognitiveEngine::from_model(&restored)?;
    let (out_a, ga) = a.generate(&prompt, 42)?;
    let (out_b, gb) = b.generate(&prompt, 42)?;
    assert_eq!(out_a, out_b);
    assert_eq!(ga.plan.energy, gb.plan.energy);
    assert_eq!(a.last_plan()?, b.last_plan()?);

    // determinism: same seed → same tokens and plan; outputs are valid token ids
    let (again, _) = a.generate(&prompt, 42)?;
    assert_eq!(out_a, again);
    assert_eq!(out_a.len(), 4);
    assert!(out_a.iter().all(|&t| t < 6));

    // the context state does not grow with the prompt: longer prompts reuse the same W_fast
    let mem = a.memory();
    let long: Vec<u32> = (0..500).map(|i| i % 6).collect();
    let s = a.encode(&long)?;
    assert_eq!(s.dim(), a.config().ttt.d_ctx);
    assert_eq!(a.memory().context_state_bytes, mem.context_state_bytes);
    Ok(())
}

/// Every trainable parameter of the full browsing model (TTT with window, pools and copy keys,
/// JEPA with probe and copy head, the DiT decoder) receives a gradient once adaLN-Zero has
/// opened up. Catches ops without a backward pass (e.g. `candle_nn::ops::softmax_last_dim`),
/// which silently freeze everything before them.
#[test]
fn every_parameter_receives_a_gradient() -> Result<()> {
    let dev = Device::Cpu;
    let mut cfg = cog_engine::browser::engine_config("tiny")?;
    cfg.planner.num_samples = 8;
    let mut tc = TrainConfig::quick(Task::Browser);
    tc.batch_size = 8;
    tc.warmup = 1;
    tc.eval_every = 0;
    tc.probe_states = 3;
    let mut trainer = Trainer::new(CogModel::new(cfg, &dev)?, tc.clone())?;
    for _ in 0..4 {
        trainer.train_step()?;
    }
    let model = &trainer.model;
    let mut rng = cog_engine::kernels::rng::Rng::new(3);
    let batch = trainer.sampler.batch(&mut rng, 8, &dev)?;
    let grads = model.loss(&batch, &mut rng, &tc)?.0.backward()?;
    let mut frozen = Vec::new();
    for name in model.online.names() {
        // a key bias shifts every score of a query equally: softmax ignores it
        if name.ends_with(".k.bias") {
            continue;
        }
        let v = model.online.var(&name).expect("listed");
        let norm = match grads.get(v.as_tensor()) {
            Some(g) => g.sqr()?.sum_all()?.to_scalar::<f32>()?,
            None => 0.0,
        };
        if norm == 0.0 {
            frozen.push(name);
        }
    }
    assert!(frozen.is_empty(), "parameters without a gradient: {frozen:?}");
    Ok(())
}
