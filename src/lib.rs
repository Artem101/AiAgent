//! # cog_engine
//!
//! Hybrid **non-autoregressive** cognitive engine:
//!
//! ```text
//!  tokens ──► TTT-Encoder ──► S_prompt ──► JEPA planner (MPPI in ℝ^{d_s}) ──► S_plan ──► CFM decoder ──► X_1 ──► argmax ──► tokens
//!             O(d²) state         E_θ, goal G        world model P_φ(s,a)          K ODE steps (Heun)
//! ```
//!
//! * [`ttt`] — Test-Time-Training encoder: the context is compressed into fast weights
//!   `W_fast ∈ ℝ^{d×d}` by online gradient descent on a self-supervised reconstruction loss.
//!   Memory is O(d²) and independent of the context length (no KV cache).
//! * [`jepa`] — latent world model trained JEPA-style (EMA target encoder + VICReg) and an
//!   inference-time trajectory optimiser (MPPI on the host with rayon, optional latent GD).
//!   Reasoning never leaves the continuous latent space.
//! * [`flow`] — conditional flow-matching decoder: a DiT vector field with cross-attention
//!   to the latent plan, integrated from Gaussian noise to token embeddings in `K` parallel
//!   (all positions at once) ODE steps.
//!
//! Two execution paths share one set of parameters:
//!
//! * **graph path** — candle ops with autograd; used for training, device generic
//!   (CPU / CUDA with the `cuda` feature);
//! * **kernel path** — packed bf16/f16/f32 weights, f32 accumulators, SIMD dot products,
//!   rayon, and every buffer pre-allocated in an [`arena::Arena`]. The hot loops
//!   ([`flow::FlowMatchingSampler::integrate`], [`ttt::FastWeightsState::step_update`],
//!   [`jepa::JEPAPlanner::plan_into`]) perform **zero heap allocations** (enforced by
//!   `tests/zero_alloc.rs`).
//!
//! Guides (in Russian) live in the repository's `docs/` directory: architecture, math,
//! training, library API, CLI, configuration, runtime internals and development. Runnable
//! examples: `examples/quickstart.rs` and `examples/staged.rs`.

pub mod arena;
pub mod config;
pub mod data;
pub mod flow;
pub mod jepa;
pub mod kernels;
pub mod model;
pub mod nn;
pub mod pipeline;
pub mod train;
pub mod ttt;
pub mod types;

pub use config::{EngineConfig, FlowConfig, JepaConfig, PlannerConfig, TTTConfig, TrainConfig};
pub use model::CogModel;
pub use pipeline::CognitiveEngine;
pub use types::{FlowState, LatentPlan, LatentState, PromptState};
