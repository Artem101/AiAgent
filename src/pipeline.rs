//! End-to-end inference: `Tokens → TTT → JEPA (MPPI) → CFM (ODE) → Tokens`.
//!
//! [`CognitiveEngine`] owns packed weights and an [`Arena`] with every buffer the three
//! stages need. After construction, [`CognitiveEngine::generate_into`] performs no heap
//! allocation (the optional latent-GD refinement is the one documented exception: it builds
//! a small candle autograd graph).

use std::time::{Duration, Instant};

use candle_core::{bail, DType, Device, Result, Tensor};

use crate::arena::Arena;
use crate::config::{EngineConfig, PlannerKind};
use crate::flow::{FlowMatchingSampler, ODESolverConfig, PackedHead, PackedVectorField, SamplerBuffers, SolverKind};
use crate::jepa::{GradientPlanner, JEPAPlanner, MppiWorkspace, PlanStats, WorldModel};
use crate::kernels::inplace::{copy_from_slice, host_read};
use crate::kernels::{rng::Rng, PackedMlp};
use crate::model::CogModel;
use crate::ttt::{PackedTttEncoder, TttWorkspace};
use crate::types::{LatentPlan, LatentState, PromptState};

/// Wall-clock time of the three stages of one generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    pub encode: Duration,
    pub plan: Duration,
    pub decode: Duration,
}

impl StageTimings {
    /// Sum of the three stages.
    pub fn total(&self) -> Duration {
        self.encode + self.plan + self.decode
    }
}

/// Diagnostics of one generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Generation {
    pub plan: PlanStats,
    pub timings: StageTimings,
    /// Whether latent GD improved on the MPPI plan.
    pub refined: bool,
}

/// Memory held by an engine (see [`CognitiveEngine::memory`]).
#[derive(Debug, Clone, Copy)]
pub struct MemoryReport {
    /// Packed weights (bf16 by default).
    pub weight_bytes: usize,
    /// All pre-allocated activations / states / scratch.
    pub arena_bytes: usize,
    pub arena_buffers: usize,
    /// `W_fast` — the entire context memory, independent of the prompt length.
    pub context_state_bytes: usize,
}

/// Inference engine: packed weights plus every pre-allocated buffer for one request.
///
/// Build it once with [`CognitiveEngine::from_model`]; afterwards
/// [`CognitiveEngine::generate_into`] runs without heap allocation. Methods take `&mut self`
/// because the buffers are reused — use one engine per thread or a `Mutex`.
pub struct CognitiveEngine {
    cfg: EngineConfig,
    ttt: PackedTttEncoder,
    ttt_ws: TttWorkspace,
    ctx_enc: PackedMlp,
    goal_head: PackedMlp,
    jepa_hidden: Box<[f32]>,
    s0_host: Box<[f32]>,
    goal_host: Box<[f32]>,
    s0: Tensor,
    goal: Tensor,
    planner: JEPAPlanner,
    mppi_ws: MppiWorkspace,
    gradient: GradientPlanner,
    graph_world: WorldModel,
    vf: PackedVectorField,
    bufs: SamplerBuffers,
    solver: ODESolverConfig,
    head: PackedHead,
    logits: Box<[f32]>,
    tokens: Box<[u32]>,
    memory: MemoryReport,
}

impl CognitiveEngine {
    /// Packs a trained model (weights in `cfg.weight_dtype`) and pre-allocates every buffer.
    pub fn from_model(model: &CogModel) -> Result<Self> {
        let cfg = model.cfg.clone();
        let dtype = cfg.weight_dtype;
        let host = Device::Cpu;
        let mut arena = Arena::new(&host);

        let ttt = model.ttt.pack(dtype)?;
        let ttt_ws = ttt.workspace(&mut arena)?;
        let jepa = model.jepa.pack(dtype)?;
        let (ds, dh) = (cfg.jepa.d_state, cfg.jepa.d_hidden);
        let planner =
            JEPAPlanner::new(jepa.world.clone(), cfg.jepa.horizon, &cfg.planner).with_policy(jepa.policy.clone());
        let mppi_ws = planner.workspace(&mut arena)?;
        let graph_world = WorldModel::from_packed(&jepa.world, &host)?;
        let vf = model.flow.pack(dtype, &mut arena)?;
        let bufs = SamplerBuffers::new(&mut arena, cfg.answer_len(), cfg.flow.d_token)?;
        let head = model.head.pack(dtype)?;

        let weight_bytes = ttt.bytes()
            + jepa.ctx_enc.bytes()
            + jepa.goal.bytes()
            + jepa.world.bytes()
            + jepa.policy.bytes()
            + vf.bytes()
            + head.lin.bytes();
        let mut engine = Self {
            jepa_hidden: arena.host(dh),
            s0_host: arena.host(ds),
            goal_host: arena.host(ds),
            s0: arena.tensor(ds)?,
            goal: arena.tensor(ds)?,
            logits: arena.host(cfg.answer_len() * cfg.vocab_size),
            tokens: arena.host_u32(cfg.answer_len()),
            gradient: GradientPlanner {
                steps: cfg.planner.gd_steps,
                lr: cfg.planner.gd_lr,
                action_cost: cfg.planner.action_cost,
            },
            solver: cfg.flow.solver.clone(),
            ctx_enc: jepa.ctx_enc,
            goal_head: jepa.goal,
            memory: MemoryReport { weight_bytes, arena_bytes: 0, arena_buffers: 0, context_state_bytes: 0 },
            cfg,
            ttt,
            ttt_ws,
            planner,
            mppi_ws,
            graph_world,
            vf,
            bufs,
            head,
        };
        engine.memory.arena_bytes = arena.bytes();
        engine.memory.arena_buffers = arena.buffers();
        engine.memory.context_state_bytes = engine.ttt_ws.state.weights.elem_count() * 4;
        Ok(engine)
    }

    /// Configuration the engine was packed with (including runtime overrides).
    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// Bytes held in packed weights, in the arena, and in `W_fast`.
    pub fn memory(&self) -> MemoryReport {
        self.memory
    }

    /// Selects MPPI alone or MPPI followed by latent gradient descent.
    pub fn set_planner(&mut self, kind: PlannerKind) {
        self.cfg.planner.kind = kind;
    }

    /// MPPI iterations (`0` = use the policy rollout as the plan).
    pub fn set_mppi_iterations(&mut self, iterations: usize) {
        self.cfg.planner.iterations = iterations;
        self.planner.config.iterations = iterations;
    }

    /// Enables/disables the policy warm start (zero-action warm start when off).
    pub fn set_policy_prior(&mut self, on: bool) {
        self.cfg.planner.policy_prior = on;
        self.planner.config.policy_prior = on;
    }

    /// Changes the ODE solver and its number of steps (no retraining needed).
    pub fn set_solver(&mut self, solver: SolverKind, steps: usize) {
        self.solver.solver = solver;
        self.solver.steps = steps.max(1);
    }

    /// Stage 1: compress the prompt into `W_fast` and read out `S_prompt`.
    pub fn encode(&mut self, prompt: &[u32]) -> Result<PromptState> {
        self.ttt.encode(prompt, &mut self.ttt_ws)
    }

    /// Stage 2: `s_0 = E(S_prompt)`, `ĝ = s_0 + G(s_0)`, then trajectory search.
    pub fn think(&mut self, prompt_state: &PromptState, seed: u64) -> Result<(LatentPlan, PlanStats, bool)> {
        let ds = self.cfg.jepa.d_state;
        host_read(prompt_state.tensor(), |p| self.ctx_enc.forward(p, &mut self.jepa_hidden, &mut self.s0_host))?;
        self.goal_host.copy_from_slice(&self.s0_host);
        self.goal_head.forward_acc(&self.s0_host, &mut self.jepa_hidden, &mut self.goal_host);
        copy_from_slice(&self.s0, &self.s0_host)?;
        copy_from_slice(&self.goal, &self.goal_host)?;
        let s0 = LatentState::new(self.s0.clone(), ds)?;
        let goal = LatentState::new(self.goal.clone(), ds)?;
        let mut stats = self.planner.plan_into(&s0, &goal, &mut self.mppi_ws, seed)?;

        let mut refined = false;
        if self.cfg.planner.kind == PlannerKind::MppiThenGradient && self.gradient.steps > 0 {
            let (h, da) = (self.cfg.jepa.horizon, self.cfg.jepa.d_action);
            let init = Tensor::from_slice(self.mppi_ws.nominal_actions(), (h, da), &Device::Cpu)?;
            let gd = self.gradient.refine(&self.graph_world, s0.tensor(), goal.tensor(), &init)?;
            if gd.energy < stats.energy {
                let traj = gd.trajectory.flatten_all()?.to_vec1::<f32>()?;
                copy_from_slice(&self.mppi_ws.plan, &traj)?;
                stats.energy = gd.energy;
                stats.terminal_error =
                    traj[h * ds..].iter().zip(self.goal_host.iter()).map(|(s, g)| (s - g) * (s - g)).sum::<f32>()
                        / ds as f32;
                refined = true;
            }
        }
        Ok((LatentPlan::new(self.mppi_ws.plan.clone(), self.cfg.plan_len(), ds)?, stats, refined))
    }

    /// Stage 3: integrate the flow from noise to `X_1` and unembed all positions at once.
    pub fn decode(&mut self, plan: &LatentPlan, seed: u64, out: &mut [u32]) -> Result<()> {
        if out.len() != self.cfg.answer_len() {
            bail!("output buffer has {} slots, expected {}", out.len(), self.cfg.answer_len())
        }
        let mut rng = Rng::stream(seed, 0xF10, 0);
        let mut sampler = FlowMatchingSampler::new(&mut self.vf, self.solver.clone());
        sampler.sample_into(plan, &mut self.bufs, &mut rng)?;
        let (head, logits) = (&self.head, &mut self.logits);
        host_read(&self.bufs.x, |x| head.decode_into(x, logits, out))
    }

    /// Full pipeline into a caller-provided buffer (`out.len() == answer_len`).
    pub fn generate_into(&mut self, prompt: &[u32], seed: u64, out: &mut [u32]) -> Result<Generation> {
        let t0 = Instant::now();
        let s_prompt = self.encode(prompt)?;
        let t1 = Instant::now();
        let (plan, stats, refined) = self.think(&s_prompt, seed)?;
        let t2 = Instant::now();
        self.decode(&plan, seed, out)?;
        let t3 = Instant::now();
        Ok(Generation {
            plan: stats,
            timings: StageTimings { encode: t1 - t0, plan: t2 - t1, decode: t3 - t2 },
            refined,
        })
    }

    /// Convenience wrapper returning an owned token vector.
    pub fn generate(&mut self, prompt: &[u32], seed: u64) -> Result<(Vec<u32>, Generation)> {
        let mut tokens = std::mem::take(&mut self.tokens);
        let res = self.generate_into(prompt, seed, &mut tokens);
        let out = tokens.to_vec();
        self.tokens = tokens;
        res.map(|g| (out, g))
    }

    /// The latent plan of the last [`CognitiveEngine::think`] call, `[H + 1, d_s]`.
    pub fn last_plan(&self) -> Result<Vec<Vec<f32>>> {
        self.mppi_ws.plan.to_vec2::<f32>()
    }

    /// Continuous output `X_1` of the last decode, `[L, d_token]`.
    pub fn last_flow_state(&self) -> Result<Tensor> {
        self.bufs.x.copy()
    }

    /// Storage precision of the packed weights.
    pub fn weight_dtype(&self) -> DType {
        self.cfg.weight_dtype
    }
}
