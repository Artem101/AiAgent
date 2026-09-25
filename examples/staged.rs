//! Tour of the public API below `CognitiveEngine::generate`: the three stages called one by
//! one, TTT fast weights, the standalone MPPI planner, and the ODE sampler with a custom
//! vector field. Weights are untrained — this shows the API, not accuracy.
//!
//! ```text
//! cargo run --release --example staged
//! ```

use candle_core::{Device, Result, Tensor};
use cog_engine::arena::Arena;
use cog_engine::flow::{FlowMatchingSampler, ODESolverConfig, SamplerBuffers, SolverKind, VectorFieldEstimator};
use cog_engine::jepa::JEPAPlanner;
use cog_engine::kernels::inplace::{host_read, host_write};
use cog_engine::kernels::rng::Rng;
use cog_engine::ttt::FastWeightsState;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, LatentPlan, LatentState};

/// Hand-written field `v(x, t) = c − x`: every coordinate relaxes towards `c`.
/// Exact solution at `t = 1`: `x_1 = c + (x_0 − c)·e^{−1}`.
struct Relax {
    c: f32,
}

impl VectorFieldEstimator for Relax {
    fn estimate_velocity_into(&mut self, x: &Tensor, _t: f32, _plan: &LatentPlan, out: &mut Tensor) -> Result<()> {
        let c = self.c;
        host_read(x, |xs| {
            host_write(out, |o| {
                for (o, &x) in o.iter_mut().zip(xs) {
                    *o = c - x;
                }
                Ok(())
            })
        })?
    }
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = EngineConfig::tiny(10, 8, 8);
    let model = CogModel::new(cfg.clone(), &dev)?;
    let mut engine = CognitiveEngine::from_model(&model)?;

    // 1. The pipeline, stage by stage.
    let prompt = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let s_prompt = engine.encode(&prompt)?; // PromptState [d_ctx]
    let (plan, stats, refined) = engine.think(&s_prompt, 7)?; // LatentPlan [H+1, d_s]
    let mut out = [0u32; 8];
    engine.decode(&plan, 7, &mut out)?; // K ODE steps + argmax
    println!(
        "[1] S_prompt {:?} → plan {:?} (energy {:.4} ← {:.4}, refined: {refined}) → tokens {out:?}",
        s_prompt.tensor().dims(),
        plan.trajectory().dims(),
        stats.energy,
        stats.initial_energy
    );

    // Context memory does not grow with the prompt length.
    let long: Vec<u32> = (0..10_000).map(|i| (i * 7 % 10) as u32).collect();
    engine.encode(&long)?;
    println!("[1] after 10 000 tokens the context state is still {} bytes", engine.memory().context_state_bytes);

    // 2. TTT fast weights: online gradient steps on ½‖W k − v‖².
    let mut fast = FastWeightsState::new(4, 0.5, &dev)?;
    let k = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], &dev)?; // unit norm
    let v = Tensor::new(&[1f32, -1.0, 0.0, 2.0], &dev)?;
    for _ in 0..10 {
        fast.step_update(&k, &v)?;
    }
    println!("[2] W·k after 10 steps = {:?} (target {:?})", fast.forward(&k)?.to_vec1::<f32>()?, v.to_vec1::<f32>()?);

    // 3. Standalone MPPI planner over the (packed) world model.
    let planner = JEPAPlanner::new(model.jepa.world.pack(cfg.weight_dtype)?, cfg.jepa.horizon, &cfg.planner);
    let s0 = Tensor::zeros(cfg.jepa.d_state, candle_core::DType::F32, &dev)?;
    let goal = (Tensor::ones(cfg.jepa.d_state, candle_core::DType::F32, &dev)? * 0.1)?;
    let mut arena = Arena::new(&dev);
    let mut ws = planner.workspace(&mut arena)?;
    let st = planner.plan_into(
        &LatentState::new(s0, cfg.jepa.d_state)?,
        &LatentState::new(goal, cfg.jepa.d_state)?,
        &mut ws,
        42,
    )?;
    println!("[3] MPPI: energy {:.5} → {:.5}, ESS {:.1}", st.initial_energy, st.energy, st.effective_samples);

    // 4. ODE sampler with a custom vector field.
    let mut field = Relax { c: 2.0 };
    let mut bufs = SamplerBuffers::new(&mut arena, 2, 3)?;
    let dummy_plan = LatentPlan::new(arena.tensor((1, 1))?, 1, 1)?; // this field ignores the plan
    let mut sampler =
        FlowMatchingSampler::new(&mut field, ODESolverConfig { steps: 8, sigma_min: 0.0, solver: SolverKind::Heun });
    sampler.sample_into(&dummy_plan, &mut bufs, &mut Rng::new(0))?;
    println!("[4] X_1 = {:?}", bufs.x.to_vec2::<f32>()?);
    Ok(())
}
