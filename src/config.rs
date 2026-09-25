//! Model / planner / training configuration and presets.

use candle_core::{bail, DType, Device, Result};

use crate::data::Task;
use crate::flow::ode_solver::{ODESolverConfig, SolverKind};

/// Test-Time-Training encoder (module 1).
#[derive(Debug, Clone)]
pub struct TTTConfig {
    /// Token embedding width fed into the TTT layer.
    pub d_model: usize,
    /// Side of the fast-weight matrix `W_fast ∈ ℝ^{d_fast×d_fast}`.
    pub d_fast: usize,
    /// Base inner-loop learning rate η (upper bound when `adaptive_lr` is on).
    pub learning_rate: f64,
    /// Per-token η_t = η · σ(w_η·x_t + b_η) instead of a constant η.
    pub adaptive_lr: bool,
    /// Number of learned probe vectors used to read `W_fast` out (`R = W_fast · P`).
    pub readout_probes: usize,
    /// Width of the prompt state `S_prompt` produced by the readout MLP.
    pub d_ctx: usize,
    /// Width `w` of the causal convolution in front of the projections: `k, v, q, η` of token
    /// `t` are computed from `[LN(x_t); LN(x_{t−1}); …; LN(x_{t−w+1})]` (`1` = the token
    /// alone). With `w > 1` a single rank-1 update can bind neighbouring tokens, e.g. a
    /// table label (key) to the value next to it.
    pub conv_width: usize,
    /// Number of final layer outputs `z_N, z_{N−1}, …` appended to the readout (`0` = none).
    /// A question at the end of the prompt then acts as an associative lookup into `W_fast`.
    pub readout_last: usize,
    /// Number of gated pools `Σ_t g_t z_t / Σ_t g_t`, `g_t = σ(w·x̃_t + b)`, appended to the
    /// readout (`0` = none): a streaming, position-independent attention pooling that lets
    /// the readout pick informative tokens anywhere in the context.
    pub readout_pools: usize,
}

/// VICReg anti-collapse regulariser weights.
#[derive(Debug, Clone)]
pub struct VicRegConfig {
    pub inv_weight: f64,
    pub var_weight: f64,
    pub cov_weight: f64,
    /// Target standard deviation γ per latent dimension.
    pub gamma: f64,
    pub eps: f64,
}

/// JEPA latent world model (module 2, learned parts).
#[derive(Debug, Clone)]
pub struct JepaConfig {
    /// Latent state width `d_s`.
    pub d_state: usize,
    /// Latent action ("thought step") width `d_a`.
    pub d_action: usize,
    /// Hidden width of every JEPA MLP.
    pub d_hidden: usize,
    /// Planning horizon `H` (the plan has `H + 1` states, `s_0 … s_H`).
    pub horizon: usize,
    /// EMA decay τ of the target encoder: `θ̄ ← τ θ̄ + (1 − τ) θ`.
    pub ema_decay: f64,
    pub vicreg: VicRegConfig,
    /// Weight of the goal-predictor loss `‖G(s_0) − sg(s̄_H)‖²`.
    pub goal_weight: f64,
    /// Weight of the behaviour-cloning loss of the proposal policy `π(s, ĝ)`.
    pub policy_weight: f64,
    /// Weight of the thought-probe loss `CE(head(Θ s) → answer tokens)` over the teacher plan
    /// states `s_0 … s_H`: every latent thought must linearly decode into the answer through
    /// the decoder's unembedding head. It gives the encoder a direct signal to extract the
    /// answer from the prompt and makes latent hypotheses readable at inference.
    /// `0` = no probe head.
    pub probe_weight: f64,
    /// Key width of the probe's copy mechanism ([`crate::copy`]): thoughts can point at
    /// context tokens instead of spelling them out. `0` = off; needs the probe.
    pub copy_dim: usize,
    /// Token ids below this are never copied (the special tokens of the vocabulary).
    pub copy_min_token: u32,
}

/// Which trajectory optimiser runs at inference time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerKind {
    /// Sampling-based Model Predictive Path Integral control (host, rayon, zero-alloc).
    Mppi,
    /// MPPI followed by gradient descent on the actions through the world model.
    MppiThenGradient,
}

/// Inference-time latent trajectory search (module 2, MPPI / latent GD).
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    pub kind: PlannerKind,
    /// Warm-start MPPI from the learned proposal policy instead of zero actions.
    pub policy_prior: bool,
    /// Number of sampled action sequences `M` per MPPI iteration.
    pub num_samples: usize,
    /// MPPI iterations; `0` returns the warm-start (policy) rollout unchanged.
    pub iterations: usize,
    /// Softmax temperature λ in `w_m ∝ exp(−E_m / λ)`.
    pub temperature: f64,
    /// When set, λ is relative to the cost spread (`λ · (mean E − min E)`), which makes
    /// the temperature invariant to the scale of the latent space.
    pub normalize_costs: bool,
    /// Standard deviation of the action perturbations at iteration 0.
    pub noise_std: f64,
    /// Multiplicative decay of `noise_std` per iteration.
    pub noise_decay: f64,
    /// Running cost weight on `‖a_t‖²` (keeps actions in the training distribution).
    pub action_cost: f64,
    /// Gradient refinement steps / learning rate (for [`PlannerKind::MppiThenGradient`]).
    pub gd_steps: usize,
    pub gd_lr: f64,
    /// Latent tree search before MPPI ("chain of latent thought"): beam width `B`, `0` = off
    /// (MPPI then starts from the plain policy rollout). Needs the policy prior.
    pub tree_beam: usize,
    /// Children per node `K`: the policy's proposal plus `K − 1` perturbed ones.
    pub tree_branch: usize,
    /// Std of the perturbations of the proposals (actions lie in `[−1, 1]`).
    pub tree_noise: f64,
}

/// Continuous-flow-matching decoder (module 3).
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Width of the continuous token space `X ∈ ℝ^{L×d_token}` the flow lives in.
    pub d_token: usize,
    /// Transformer width of the vector field.
    pub d_hidden: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub mlp_ratio: usize,
    /// Width of the sinusoidal time embedding.
    pub d_time: usize,
    /// Output length `L` (tokens produced in parallel).
    pub seq_len: usize,
    pub solver: ODESolverConfig,
}

/// How the action tokens are produced from the latent plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDecoder {
    /// Continuous flow matching conditioned on the whole plan (the default).
    Flow,
    /// The thought probe applied to the last plan state `s_H` — the answer the latent
    /// reasoning arrived at, read out in one linear step (needs a probe).
    Probe,
    /// The thought probe applied to `s_0`: the first impression, before any reasoning
    /// (an ablation of the latent reasoning).
    ProbeStart,
    /// Self-consistency over the reasoning: per position, the token with the highest summed
    /// log-probability over every thought of the chosen plan `s_0 … s_H` and the terminal
    /// thoughts of the surviving tree hypotheses.
    ProbeVote,
    /// Sequence-level self-consistency over the same thoughts: every thought proposes its whole
    /// action (per-position argmax), and the proposal with the highest log-probability summed
    /// over all thoughts wins. Unlike the per-position vote it never splices the verb of one
    /// thought with the text of another.
    ProbeConsensus,
    /// The first thought `s_0` decides; the other thoughts and the tree hypotheses are ranked
    /// as in [`ActionDecoder::ProbeConsensus`] and serve as alternatives when a caller rejects
    /// the first choice (see `CognitiveEngine::ranked_proposals`).
    ProbeFirst,
}

impl ActionDecoder {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "flow" | "cfm" => Ok(Self::Flow),
            "probe" => Ok(Self::Probe),
            "probe-start" => Ok(Self::ProbeStart),
            "probe-vote" => Ok(Self::ProbeVote),
            "probe-consensus" => Ok(Self::ProbeConsensus),
            "probe-first" => Ok(Self::ProbeFirst),
            other => {
                bail!("unknown decoder '{other}' (flow | probe | probe-start | probe-vote | probe-consensus | probe-first)")
            }
        }
    }
}

/// Full engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub vocab_size: usize,
    /// Maximum prompt length (positional table size is `max_prompt_len + seq_len`).
    pub max_prompt_len: usize,
    pub ttt: TTTConfig,
    pub jepa: JepaConfig,
    pub planner: PlannerConfig,
    pub flow: FlowConfig,
    /// Storage precision of packed inference weights (accumulation is always f32).
    pub weight_dtype: DType,
    /// How actions / answers are decoded from the latent plan (inference only).
    pub decoder: ActionDecoder,
    pub seed: u64,
}

impl EngineConfig {
    /// Small configuration that trains in about a minute on a laptop CPU.
    pub fn tiny(vocab_size: usize, prompt_len: usize, answer_len: usize) -> Self {
        Self {
            vocab_size,
            max_prompt_len: prompt_len,
            ttt: TTTConfig {
                d_model: 64,
                d_fast: 32,
                learning_rate: 1.0,
                adaptive_lr: true,
                readout_probes: 8,
                d_ctx: 96,
                conv_width: 1,
                readout_last: 0,
                readout_pools: 0,
            },
            jepa: JepaConfig {
                d_state: 32,
                d_action: 8,
                d_hidden: 128,
                horizon: 4,
                ema_decay: 0.99,
                vicreg: VicRegConfig { inv_weight: 1.0, var_weight: 0.5, cov_weight: 0.04, gamma: 1.0, eps: 1e-4 },
                goal_weight: 1.0,
                policy_weight: 1.0,
                probe_weight: 0.0,
                copy_dim: 0,
                copy_min_token: 0,
            },
            planner: PlannerConfig {
                kind: PlannerKind::Mppi,
                policy_prior: true,
                num_samples: 128,
                iterations: 8,
                temperature: 0.1,
                normalize_costs: true,
                noise_std: 0.6,
                noise_decay: 0.75,
                action_cost: 0.01,
                gd_steps: 20,
                gd_lr: 0.05,
                tree_beam: 0,
                tree_branch: 4,
                tree_noise: 0.5,
            },
            flow: FlowConfig {
                d_token: 32,
                d_hidden: 64,
                n_heads: 4,
                n_layers: 2,
                mlp_ratio: 4,
                d_time: 64,
                seq_len: answer_len,
                solver: ODESolverConfig { steps: 16, sigma_min: 1e-4, solver: SolverKind::Heun },
            },
            weight_dtype: DType::BF16,
            decoder: ActionDecoder::Flow,
            seed: 7,
        }
    }

    /// Wider configuration (still CPU-trainable, but slower).
    pub fn small(vocab_size: usize, prompt_len: usize, answer_len: usize) -> Self {
        let mut c = Self::tiny(vocab_size, prompt_len, answer_len);
        c.ttt.d_model = 128;
        c.ttt.d_fast = 64;
        c.ttt.d_ctx = 192;
        c.jepa.d_state = 64;
        c.jepa.d_action = 16;
        c.jepa.d_hidden = 256;
        c.planner.num_samples = 256;
        c.flow.d_token = 64;
        c.flow.d_hidden = 128;
        c.flow.n_layers = 3;
        c.flow.d_time = 128;
        c
    }

    pub fn preset(name: &str, vocab_size: usize, prompt_len: usize, answer_len: usize) -> Result<Self> {
        match name {
            "tiny" => Ok(Self::tiny(vocab_size, prompt_len, answer_len)),
            "small" => Ok(Self::small(vocab_size, prompt_len, answer_len)),
            other => bail!("unknown preset '{other}' (expected tiny | small)"),
        }
    }

    /// Output length `L`.
    pub fn answer_len(&self) -> usize {
        self.flow.seq_len
    }

    /// Number of latent states in a plan (`H + 1`).
    pub fn plan_len(&self) -> usize {
        self.jepa.horizon + 1
    }

    /// Positional-table length: prompt followed by the answer continuation used in training.
    pub fn max_positions(&self) -> usize {
        self.max_prompt_len + self.flow.seq_len
    }

    pub fn validate(&self) -> Result<()> {
        let f = &self.flow;
        if !f.d_hidden.is_multiple_of(f.n_heads) {
            bail!("flow.d_hidden ({}) must be divisible by n_heads ({})", f.d_hidden, f.n_heads)
        }
        if !f.d_time.is_multiple_of(2) {
            bail!("flow.d_time must be even")
        }
        if !f.seq_len.is_multiple_of(self.jepa.horizon) {
            bail!(
                "answer length ({}) must be divisible by the planning horizon ({}): each plan step \
                 covers one answer chunk during training",
                f.seq_len,
                self.jepa.horizon
            )
        }
        if self.ttt.conv_width == 0 {
            bail!("ttt.conv_width must be >= 1")
        }
        if f.solver.steps == 0 {
            bail!("flow.solver.steps must be > 0")
        }
        if self.planner.num_samples == 0 {
            bail!("planner needs at least one sample")
        }
        if !(0.0..1.0).contains(&self.jepa.ema_decay) {
            bail!("ema_decay must be in [0, 1)")
        }
        if self.vocab_size < 2 {
            bail!("vocab_size must be >= 2")
        }
        if self.jepa.copy_dim > 0 && self.jepa.probe_weight <= 0.0 {
            bail!("the copy mechanism (jepa.copy_dim) needs the thought probe (jepa.probe_weight > 0)")
        }
        Ok(())
    }
}

/// Training hyper-parameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub task: Task,
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    /// Weight of the cross-entropy on the one-step estimate `x̂_1 = x_t + (1 − t) v_θ`.
    pub ce_weight: f64,
    /// Weight of the cross-entropy that trains the unembedding head on noisy clean targets.
    pub head_weight: f64,
    /// Std of Gaussian noise added to the conditioning plan (robustness to planner error).
    pub plan_noise: f64,
    /// Text corpus (one sentence per line) for `Task::Text`, or to mix into `Task::Browser`.
    pub corpus: Option<std::path::PathBuf>,
    /// Share of text-continuation examples in browsing batches when a corpus is given.
    pub text_mix: f64,
    /// Also train the thought probe on the goal `ĝ = s_0 + G(s_0)`, so that the state the
    /// latent search steers towards decodes into the answer.
    pub probe_goal: bool,
    /// Plan states the probe loss sees per step: `s_0`, `s_H` and `probe_states − 2` random
    /// intermediate thoughts (a fresh draw each step; `0` = all `H + 1`). The probe dominates
    /// the step time with long answers and a large vocabulary.
    pub probe_states: usize,
    /// `None` = auto: f32 on CPU (candle has no CPU bf16 matmul), bf16 on accelerators.
    /// Master weights and optimiser moments always stay in f32.
    pub compute_dtype: Option<DType>,
    pub log_every: usize,
    pub eval_every: usize,
    pub eval_samples: usize,
}

impl TrainConfig {
    pub fn quick(task: Task) -> Self {
        Self {
            task,
            batch_size: 64,
            steps: 1500,
            lr: 2e-3,
            min_lr: 1e-4,
            warmup: 100,
            weight_decay: 0.01,
            grad_clip: 1.0,
            ce_weight: 0.2,
            head_weight: 0.2,
            plan_noise: 0.05,
            corpus: None,
            text_mix: 0.25,
            probe_goal: false,
            probe_states: 0,
            compute_dtype: None,
            log_every: 100,
            eval_every: 500,
            eval_samples: 128,
        }
    }

    pub fn resolved_compute_dtype(&self, device: &Device) -> DType {
        self.compute_dtype.unwrap_or(if device.is_cpu() { DType::F32 } else { DType::BF16 })
    }
}
