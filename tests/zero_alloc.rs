//! Enforces the "zero allocations in the hot loop" policy with a counting global allocator.
//!
//! This file intentionally contains a single `#[test]` so no other test thread allocates
//! concurrently while we measure.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{Device, Result, Tensor};
use cog_engine::arena::Arena;
use cog_engine::flow::{FlowMatchingSampler, SamplerBuffers, VectorFieldEstimator};
use cog_engine::jepa::JEPAPlanner;
use cog_engine::kernels::{self, rng::Rng};
use cog_engine::ttt::FastWeightsState;
use cog_engine::{CogModel, CognitiveEngine, EngineConfig, LatentPlan, LatentState};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.load(Ordering::SeqCst);
    let r = f();
    (r, ALLOCS.load(Ordering::SeqCst) - before)
}

fn check_hot_loops(model: &CogModel, label: &str) -> Result<()> {
    let cfg = &model.cfg;
    let dev = Device::Cpu;
    let mut arena = Arena::new(&dev);

    // --- TTT: FastWeightsState::step_update over a long stream (tensor API) ---
    let d = cfg.ttt.d_fast;
    let mut fast = FastWeightsState::in_arena(&mut arena, d, 0.5)?;
    let k = arena.tensor(d)?;
    let v = arena.tensor(d)?;
    kernels::inplace::copy_from_slice(&k, &vec![1.0 / (d as f32).sqrt(); d])?;
    kernels::inplace::copy_from_slice(&v, &vec![0.25; d])?;
    fast.step_update(&k, &v)?; // warm-up
    let (r, n) = count(|| -> Result<()> {
        for _ in 0..10_000 {
            fast.step_update(&k, &v)?;
        }
        Ok(())
    });
    r?;
    assert_eq!(n, 0, "[{label}] FastWeightsState::step_update allocated {n} times");

    // --- CFM: FlowMatchingSampler integration loop (Heun, 2K NFE) ---
    let mut vf = model.flow.pack(cfg.weight_dtype, &mut arena)?;
    let mut bufs = SamplerBuffers::new(&mut arena, cfg.answer_len(), cfg.flow.d_token)?;
    let plan_t = arena.tensor((cfg.plan_len(), cfg.jepa.d_state))?;
    let plan = LatentPlan::new(plan_t, cfg.plan_len(), cfg.jepa.d_state)?;
    let mut rng = Rng::new(1);
    vf.prepare(&plan)?;
    let mut sampler = FlowMatchingSampler::new(&mut vf, cfg.flow.solver.clone());
    sampler.sample_into(&plan, &mut bufs, &mut rng)?; // warm-up
    let (r, n) = count(|| sampler.sample_into(&plan, &mut bufs, &mut rng));
    r?;
    assert_eq!(n, 0, "[{label}] FlowMatchingSampler::sample_into allocated {n} times");

    // --- JEPA: MPPI trajectory search ---
    let planner = JEPAPlanner::new(model.jepa.world.pack(cfg.weight_dtype)?, cfg.jepa.horizon, &cfg.planner);
    let mut ws = planner.workspace(&mut arena)?;
    let s0 = LatentState::new(arena.tensor(cfg.jepa.d_state)?, cfg.jepa.d_state)?;
    let goal_t = arena.tensor(cfg.jepa.d_state)?;
    kernels::inplace::copy_from_slice(&goal_t, &vec![0.3; cfg.jepa.d_state])?;
    let goal = LatentState::new(goal_t, cfg.jepa.d_state)?;
    planner.plan_into(&s0, &goal, &mut ws, 3)?; // warm-up
    let (r, n) = count(|| planner.plan_into(&s0, &goal, &mut ws, 4));
    r?;
    assert_eq!(n, 0, "[{label}] JEPAPlanner::plan_into allocated {n} times");

    // --- Whole pipeline: tokens → TTT → MPPI → CFM → tokens ---
    let mut engine = CognitiveEngine::from_model(model)?;
    let prompt: Vec<u32> = (0..cfg.max_prompt_len as u32).map(|i| i % cfg.vocab_size as u32).collect();
    let mut out = vec![0u32; cfg.answer_len()];
    engine.generate_into(&prompt, 0, &mut out)?; // warm-up
    let (r, n) = count(|| engine.generate_into(&prompt, 1, &mut out));
    r?;
    assert_eq!(n, 0, "[{label}] CognitiveEngine::generate_into allocated {n} times");
    Ok(())
}

#[test]
fn hot_loops_do_not_allocate() -> Result<()> {
    let model = CogModel::new(EngineConfig::tiny(10, 8, 8), &Device::Cpu)?;
    // The browsing configuration: TTT causal window, last-output readout, answer probe.
    let browser = CogModel::new(cog_engine::browser::engine_config("tiny")?, &Device::Cpu)?;

    // Sequential kernels.
    kernels::set_parallel(false);
    check_hot_loops(&model, "sequential")?;
    check_hot_loops(&browser, "sequential, browser config")?;

    // rayon fan-out inside a pool: nested parallel iterators use worker-local deques only.
    kernels::set_parallel(true);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().expect("pool");
    // Make sure every worker thread has started (thread start-up allocates) before measuring.
    pool.broadcast(|_| ());
    pool.install(|| check_hot_loops(&model, "rayon"))?;
    pool.install(|| check_hot_loops(&browser, "rayon, browser config"))?;

    // Sanity: the counter does observe allocations.
    let (_, n) = count(|| Tensor::zeros(16, candle_core::DType::F32, &Device::Cpu));
    assert!(n > 0);
    Ok(())
}
