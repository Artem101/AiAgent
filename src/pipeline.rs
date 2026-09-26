//! End-to-end inference: `Tokens → TTT → JEPA (MPPI) → CFM (ODE) → Tokens`.
//!
//! [`CognitiveEngine`] owns packed weights and an [`Arena`] with every buffer the three
//! stages need. After construction, [`CognitiveEngine::generate_into`] performs no heap
//! allocation (the optional latent-GD refinement is the one documented exception: it builds
//! a small candle autograd graph).

use std::time::{Duration, Instant};

use candle_core::{bail, DType, Device, Result, Tensor};

use crate::arena::Arena;
pub use crate::config::ActionDecoder;
use crate::config::{EngineConfig, PlannerKind};
use crate::copy::{CopyScratch, PackedCopyHead};
use crate::flow::{FlowMatchingSampler, ODESolverConfig, PackedHead, PackedVectorField, SamplerBuffers, SolverKind};
use crate::jepa::planner::DepthStats;
use crate::jepa::{GradientPlanner, JEPAPlanner, MppiWorkspace, PlanStats, WorldModel};
use crate::kernels::inplace::{copy_from_slice, host_read};
use crate::kernels::{rng::Rng, PackedLinear, PackedMlp};
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

/// Readable trace of the last latent reasoning ([`CognitiveEngine::last_reasoning`]).
///
/// Thoughts are latent states; with a trained thought probe (`JepaConfig::probe_weight > 0`)
/// each one is also decoded into the action tokens it "has in mind".
#[derive(Debug, Clone)]
pub struct Reasoning {
    pub stats: PlanStats,
    /// Whether latent GD improved on MPPI.
    pub refined: bool,
    /// Per-depth statistics of the tree search (empty when it is off).
    pub depths: Vec<DepthStats>,
    /// Surviving hypotheses, best first: energy and the decoded terminal thought.
    pub hypotheses: Vec<(f32, Vec<u32>)>,
    /// The chosen plan `s_0 … s_H`, each thought decoded (empty without a probe).
    pub chain: Vec<Vec<u32>>,
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
    /// Thought probe `s → [L, d_token]` (decodes latent states for [`Reasoning`]).
    probe: Option<PackedLinear>,
    /// Copy mechanism of the probe (points into the prompt's copy memory).
    copy: Option<PackedCopyHead>,
    copy_scratch: CopyScratch,
    logits: Box<[f32]>,
    /// `[L, d_token]` scratch of the probe decoder and `[L, vocab]` vote accumulator.
    probe_emb: Box<[f32]>,
    vote: Box<[f32]>,
    /// Proposals `[thoughts, L]` and their scores for [`ActionDecoder::ProbeConsensus`].
    proposals: Box<[u32]>,
    proposal_scores: Box<[f32]>,
    decoder: ActionDecoder,
    tokens: Box<[u32]>,
    memory: MemoryReport,
    refined: bool,
}

impl CognitiveEngine {
    /// Packs a trained model (weights in `cfg.weight_dtype`) and pre-allocates every buffer.
    pub fn from_model(model: &CogModel) -> Result<Self> {
        let cfg = model.cfg.clone();
        let dtype = cfg.weight_dtype;
        let host = Device::Cpu;
        let mut arena = Arena::new(&host);

        let ttt = model.ttt.pack(dtype, cfg.max_prompt_len)?;
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
        let probe = model.jepa.probe.as_ref().map(|p| p.pack(dtype)).transpose()?;
        let copy = model.jepa.copy.as_ref().map(|c| c.pack(dtype)).transpose()?;
        let copy_scratch = CopyScratch::new(
            &mut arena,
            copy.as_ref().map_or(0, |c| c.d_key()),
            if copy.is_some() { cfg.max_prompt_len } else { 0 },
        );
        if cfg.decoder != ActionDecoder::Flow && probe.is_none() {
            bail!("decoder {:?} needs a model trained with a thought probe (jepa.probe_weight > 0)", cfg.decoder)
        }

        let weight_bytes = ttt.bytes()
            + jepa.ctx_enc.bytes()
            + jepa.goal.bytes()
            + jepa.world.bytes()
            + jepa.policy.bytes()
            + vf.bytes()
            + head.lin.bytes()
            + probe.as_ref().map_or(0, |p| p.bytes())
            + copy.as_ref().map_or(0, |c| c.bytes());
        let mut engine = Self {
            jepa_hidden: arena.host(dh),
            s0_host: arena.host(ds),
            goal_host: arena.host(ds),
            s0: arena.tensor(ds)?,
            goal: arena.tensor(ds)?,
            logits: arena.host(cfg.answer_len() * cfg.vocab_size),
            probe_emb: arena.host(if probe.is_some() { cfg.answer_len() * cfg.flow.d_token } else { 0 }),
            vote: arena.host(if probe.is_some() { cfg.answer_len() * cfg.vocab_size } else { 0 }),
            proposals: arena.host_u32(Self::max_thoughts(&cfg) * cfg.answer_len()),
            proposal_scores: arena.host(Self::max_thoughts(&cfg)),
            decoder: cfg.decoder,
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
            probe,
            copy,
            copy_scratch,
            refined: false,
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

    /// Beam width of the latent tree search (`0` = off; re-allocates the planner buffers).
    pub fn set_tree(&mut self, beam: usize) -> Result<()> {
        self.cfg.planner.tree_beam = beam;
        self.planner.config.tree_beam = beam;
        let mut arena = Arena::new(&Device::Cpu);
        self.mppi_ws = self.planner.workspace(&mut arena)?;
        self.proposals = arena.host_u32(Self::max_thoughts(&self.cfg) * self.cfg.answer_len());
        self.proposal_scores = arena.host(Self::max_thoughts(&self.cfg));
        Ok(())
    }

    /// Thoughts a decision can consult: the plan `s_0 … s_H` and the surviving tree leaves.
    fn max_thoughts(cfg: &EngineConfig) -> usize {
        cfg.plan_len() + cfg.planner.tree_beam
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
        self.refined = refined;
        Ok((LatentPlan::new(self.mppi_ws.plan.clone(), self.cfg.plan_len(), ds)?, stats, refined))
    }

    /// Log-probabilities `[L, vocab]` of the tokens a latent `state` stands for: the probe's
    /// embedding unembedded by the shared head, mixed with pointers into the prompt when the
    /// model has a copy mechanism. No allocation.
    #[allow(clippy::too_many_arguments)]
    fn probe_logp(
        probe: &PackedLinear,
        head: &PackedHead,
        copy: Option<&PackedCopyHead>,
        memory: (&[f32], &[u32]),
        scratch: &mut CopyScratch,
        state: &[f32],
        emb: &mut [f32],
        rows: &mut [f32],
    ) {
        probe.forward(state, emb);
        head.lin.forward(emb, rows);
        match copy {
            Some(c) => {
                c.mix(emb, rows, memory.0, memory.1, scratch);
                rows.iter_mut().for_each(|x| *x = (*x + 1e-12).ln());
            }
            None => {
                for row in rows.chunks_exact_mut(head.vocab()) {
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
                    row.iter_mut().for_each(|x| *x -= lse);
                }
            }
        }
    }

    fn argmax_rows(rows: &[f32], vocab: usize, out: &mut [u32]) {
        for (o, row) in out.iter_mut().zip(rows.chunks_exact(vocab)) {
            *o = row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
                as u32;
        }
    }

    /// Decodes a latent state with the thought probe (`None` without a probe; allocates).
    pub fn decode_thought(&self, state: &[f32]) -> Option<Vec<u32>> {
        let probe = self.probe.as_ref()?;
        let (l, dt, v) = (self.cfg.answer_len(), self.cfg.flow.d_token, self.head.vocab());
        let (mut emb, mut rows, mut out) = (vec![0f32; l * dt], vec![0f32; l * v], vec![0u32; l]);
        let mut scratch = CopyScratch::new(
            &mut Arena::new(&Device::Cpu),
            self.copy.as_ref().map_or(0, |c| c.d_key()),
            self.cfg.max_prompt_len,
        );
        let memory = self.ttt_ws.copy_memory();
        Self::probe_logp(probe, &self.head, self.copy.as_ref(), memory, &mut scratch, state, &mut emb, &mut rows);
        Self::argmax_rows(&rows, v, &mut out);
        Some(out)
    }

    /// The latent reasoning of the last [`CognitiveEngine::think`]: tree statistics, surviving
    /// hypotheses and the chosen chain of thoughts, decoded by the probe (allocates; call it
    /// outside hot loops).
    pub fn last_reasoning(&self, stats: PlanStats) -> Result<Reasoning> {
        let ds = self.cfg.jepa.d_state;
        let h = self.cfg.jepa.horizon;
        let hypotheses = self
            .mppi_ws
            .hypotheses()
            .map(|(e, traj)| (e, self.decode_thought(&traj[h * ds..]).unwrap_or_default()))
            .collect();
        let plan = self.mppi_ws.plan.flatten_all()?.to_vec1::<f32>()?;
        let chain = plan.chunks_exact(ds).filter_map(|s| self.decode_thought(s)).collect();
        Ok(Reasoning { stats, refined: self.refined, depths: self.mppi_ws.tree_depths().collect(), hypotheses, chain })
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

    /// Selects how actions are decoded from the plan (see [`ActionDecoder`]).
    pub fn set_decoder(&mut self, decoder: ActionDecoder) -> Result<()> {
        if decoder != ActionDecoder::Flow && self.probe.is_none() {
            bail!("the probe decoder needs a model trained with a thought probe (jepa.probe_weight > 0)")
        }
        self.decoder = decoder;
        Ok(())
    }

    pub fn decoder(&self) -> ActionDecoder {
        self.decoder
    }

    /// Stage 3 with the probe: `out = argmax P(· | Θ_probe s_t)` for plan state `t` (no
    /// allocation).
    fn decode_probe(&mut self, t: usize, out: &mut [u32]) -> Result<()> {
        let Some(probe) = &self.probe else { bail!("no thought probe") };
        let ds = self.cfg.jepa.d_state;
        let Self { probe_emb, head, logits, copy, copy_scratch, ttt_ws, mppi_ws, .. } = self;
        host_read(&mppi_ws.plan, |plan| {
            let state = &plan[t * ds..(t + 1) * ds];
            let memory = ttt_ws.copy_memory();
            Self::probe_logp(probe, head, copy.as_ref(), memory, copy_scratch, state, probe_emb, logits);
        })?;
        Self::argmax_rows(logits, head.vocab(), out);
        Ok(())
    }

    /// Stage 3 by self-consistency (see [`ActionDecoder::ProbeVote`]); no allocation.
    fn decode_vote(&mut self, out: &mut [u32]) -> Result<()> {
        let Some(probe) = &self.probe else { bail!("no thought probe") };
        let (ds, h) = (self.cfg.jepa.d_state, self.cfg.jepa.horizon);
        let Self { probe_emb, head, logits, vote, mppi_ws, copy, copy_scratch, ttt_ws, .. } = self;
        let memory = ttt_ws.copy_memory();
        vote.fill(0.0);
        {
            let mut add = |state: &[f32]| {
                Self::probe_logp(probe, head, copy.as_ref(), memory, copy_scratch, state, probe_emb, logits);
                vote.iter_mut().zip(logits.iter()).for_each(|(a, &x)| *a += x);
            };
            host_read(&mppi_ws.plan, |plan| plan.chunks_exact(ds).for_each(&mut add))?;
            for (_, traj) in mppi_ws.hypotheses() {
                add(&traj[h * ds..]);
            }
        }
        Self::argmax_rows(vote, head.vocab(), out);
        Ok(())
    }

    /// Stage 3 by sequence-level self-consistency (see [`ActionDecoder::ProbeConsensus`]); no
    /// allocation.
    fn decode_consensus(&mut self, out: &mut [u32]) -> Result<()> {
        let Some(probe) = &self.probe else { bail!("no thought probe") };
        let (ds, h, l) = (self.cfg.jepa.d_state, self.cfg.jepa.horizon, self.cfg.answer_len());
        let Self { probe_emb, head, logits, mppi_ws, copy, copy_scratch, ttt_ws, proposals, proposal_scores, .. } =
            self;
        let memory = ttt_ws.copy_memory();
        let v = head.vocab();
        // pass 1: every thought proposes its action; pass 2: every thought scores every proposal
        proposal_scores.fill(0.0);
        for pass in 0..2 {
            let mut n = 0;
            let mut visit = |state: &[f32]| {
                Self::probe_logp(probe, head, copy.as_ref(), memory, copy_scratch, state, probe_emb, logits);
                if pass == 0 {
                    Self::argmax_rows(logits, v, &mut proposals[n * l..(n + 1) * l]);
                    n += 1;
                } else {
                    for (score, prop) in proposal_scores.iter_mut().zip(proposals.chunks_exact(l)) {
                        *score += prop.iter().enumerate().map(|(i, &t)| logits[i * v + t as usize]).sum::<f32>();
                    }
                }
            };
            host_read(&mppi_ws.plan, |plan| plan.chunks_exact(ds).for_each(&mut visit))?;
            for (_, traj) in mppi_ws.hypotheses() {
                visit(&traj[h * ds..]);
            }
            if pass == 0 {
                // proposals that were not made this time (fewer leaves) must not win
                proposal_scores[n..].fill(f32::NEG_INFINITY);
            }
        }
        let best = proposal_scores
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
            .0;
        out.copy_from_slice(&proposals[best * l..(best + 1) * l]);
        Ok(())
    }

    /// Proposals of the last [`ActionDecoder::ProbeConsensus`] decision, best first and without
    /// duplicates: `(summed log-probability over all thoughts, tokens)` (allocates). A caller
    /// that knows its output grammar can take the best *valid* proposal instead of the best one.
    pub fn ranked_proposals(&self) -> Vec<(f32, Vec<u32>)> {
        let l = self.cfg.answer_len();
        let mut out: Vec<(f32, Vec<u32>)> = Vec::new();
        for (&score, prop) in self.proposal_scores.iter().zip(self.proposals.chunks_exact(l)) {
            if score.is_finite() && !out.iter().any(|(_, p)| p == prop) {
                out.push((score, prop.to_vec()));
            }
        }
        out.sort_by(|a, b| b.0.total_cmp(&a.0));
        out
    }

    /// Full pipeline into a caller-provided buffer (`out.len() == answer_len`).
    pub fn generate_into(&mut self, prompt: &[u32], seed: u64, out: &mut [u32]) -> Result<Generation> {
        let t0 = Instant::now();
        let s_prompt = self.encode(prompt)?;
        let t1 = Instant::now();
        let (plan, stats, refined) = self.think(&s_prompt, seed)?;
        let t2 = Instant::now();
        match self.decoder {
            ActionDecoder::Flow => self.decode(&plan, seed, out)?,
            ActionDecoder::Probe => self.decode_probe(self.cfg.jepa.horizon, out)?,
            ActionDecoder::ProbeStart => self.decode_probe(0, out)?,
            ActionDecoder::ProbeVote => self.decode_vote(out)?,
            ActionDecoder::ProbeConsensus => self.decode_consensus(out)?,
            ActionDecoder::ProbeFirst => {
                self.decode_consensus(out)?;
                let l = self.cfg.answer_len();
                out.copy_from_slice(&self.proposals[..l]); // the proposal of s_0
            }
        }
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
