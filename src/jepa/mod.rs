//! Module 2 — JEPA latent planner ("System 2").
//!
//! Learned components (all in `ℝ^{d_s}`, never in token space):
//!
//! * context encoder `E_θ(S_prompt) → s_0` and its EMA target `Ē_θ`
//!   (`θ̄ ← τ θ̄ + (1 − τ) θ`);
//! * world model `P_φ(s_t, a_t) → ŝ_{t+1}`;
//! * latent-action (inverse) model `a_t = tanh(A_ψ(ŝ_t, s̄_{t+1}))` — infers, during
//!   training only, which "thought step" moves the state towards the next target;
//! * goal head `G_ω(s_0) → ĝ ≈ s̄_H`, the energy target used by the planner at inference;
//! * proposal policy `π(s, ĝ) → a`, behaviour-cloned from the latent-action model along the
//!   teacher trajectory; it warm-starts MPPI (amortised System-2 search).
//!
//! Training targets come from the TTT encoder itself: `s̄_h = Ē(S_h)` where `S_h` is the
//! readout after the prompt **plus the first `h` answer chunks**. The reasoning trajectory is
//! therefore a path in latent space from "question" to "question + answer". The loss is
//! VICReg over `(ŝ_h, s̄_h)` plus the goal regression.

pub mod planner;
pub mod vicreg;
pub mod world_model;

use candle_core::{DType, Result, Tensor};

pub use planner::{GradientPlan, GradientPlanner, JEPAPlanner, MppiWorkspace, PlanStats};
pub use world_model::{PackedWorldModel, WorldModel};

use crate::config::{EngineConfig, JepaConfig};
use crate::kernels::PackedMlp;
use crate::nn::{self, Mlp, ParamStore};
use vicreg::VicRegLoss;

#[derive(Debug, Clone)]
pub struct Jepa {
    pub ctx_enc: Mlp,
    /// EMA copy of `ctx_enc` (lives in the non-trainable parameter store).
    pub target_enc: Mlp,
    pub world: WorldModel,
    pub inverse: Mlp,
    pub goal: Mlp,
    pub policy: Mlp,
}

/// Individual (unweighted) JEPA loss terms.
pub struct JepaTerms {
    pub invariance: Tensor,
    pub variance: Tensor,
    pub covariance: Tensor,
    pub goal: Tensor,
    pub policy: Tensor,
}

impl JepaTerms {
    pub fn weighted(&self, cfg: &JepaConfig) -> Result<Tensor> {
        let v = &cfg.vicreg;
        let vic = (((&self.invariance * v.inv_weight)? + (&self.variance * v.var_weight)?)?
            + (&self.covariance * v.cov_weight)?)?;
        (vic + (&self.goal * cfg.goal_weight)?)? + (&self.policy * cfg.policy_weight)?
    }
}

const CTX_ENC: &str = "jepa.ctx_enc";

impl Jepa {
    pub fn new(online: &mut ParamStore, target: &mut ParamStore, cfg: &EngineConfig) -> Result<Self> {
        let j = &cfg.jepa;
        let ctx_enc = online.mlp(CTX_ENC, cfg.ttt.d_ctx, j.d_hidden, j.d_state)?;
        let target_enc = target.mlp(CTX_ENC, cfg.ttt.d_ctx, j.d_hidden, j.d_state)?;
        let jepa = Self {
            ctx_enc,
            target_enc,
            world: WorldModel::new(online, "jepa.world", j.d_state, j.d_action, j.d_hidden)?,
            inverse: online.mlp("jepa.inverse", 2 * j.d_state, j.d_hidden, j.d_action)?,
            goal: online.mlp("jepa.goal", j.d_state, j.d_hidden, j.d_state)?,
            policy: online.mlp("jepa.policy", 2 * j.d_state, j.d_hidden, j.d_action)?,
        };
        Self::ema(online, target, 0.0)?; // Ē ← E
        Ok(jepa)
    }

    /// `s_0 = E_θ(S_prompt)`.
    pub fn encode(&self, s_prompt: &Tensor) -> Result<Tensor> {
        self.ctx_enc.forward(s_prompt)
    }

    /// Energy target `ĝ = s_0 + G_ω(s_0)`.
    pub fn goal_of(&self, s0: &Tensor) -> Result<Tensor> {
        s0 + self.goal.forward(s0)?
    }

    /// `a = tanh(A_ψ([s ; s_next]))`.
    pub fn infer_action(&self, s: &Tensor, s_next: &Tensor) -> Result<Tensor> {
        let s_next = s_next.to_dtype(s.dtype())?;
        self.inverse.forward(&Tensor::cat(&[s, &s_next], 1)?)?.tanh()
    }

    /// Proposal `a = tanh(π([s ; ĝ]))`.
    pub fn policy_action(&self, s: &Tensor, goal: &Tensor) -> Result<Tensor> {
        let goal = goal.to_dtype(s.dtype())?;
        self.policy.forward(&Tensor::cat(&[s, &goal], 1)?)?.tanh()
    }

    /// Training forward. `readouts[h]` is the TTT readout after the prompt plus `h` answer
    /// chunks (`h = 0..=H`), each `[B, d_ctx]`. Returns the teacher-forced plan
    /// `[B, H + 1, d_s]` (used to condition the decoder) and the unweighted loss terms.
    pub fn forward_train(&self, readouts: &[Tensor], cfg: &JepaConfig) -> Result<(Tensor, JepaTerms)> {
        let horizon = readouts.len() - 1;
        let targets =
            readouts.iter().map(|r| Ok(self.target_enc.forward(&r.detach())?.detach())).collect::<Result<Vec<_>>>()?;
        let s0 = self.encode(&readouts[0])?;
        let g_hat = self.goal_of(&s0)?;
        let g_detached = g_hat.detach();
        let mut states = Vec::with_capacity(horizon + 1);
        let mut bc = Vec::with_capacity(horizon);
        states.push(s0.clone());
        for t in 0..horizon {
            let a = self.infer_action(&states[t], &targets[t + 1])?;
            // behaviour cloning: π(sg(ŝ_t), sg(ĝ)) ≈ sg(a_t)
            bc.push(nn::mse(&self.policy_action(&states[t].detach(), &g_detached)?, &a.detach())?);
            let next = self.world.forward(&states[t], &a)?;
            states.push(next);
        }
        let (mut inv, mut var, mut cov) = (Vec::new(), Vec::new(), Vec::new());
        for (s, target) in states.iter().zip(&targets) {
            let v = VicRegLoss::compute(s, target, &cfg.vicreg)?;
            inv.push(v.invariance);
            var.push(v.variance);
            cov.push(v.covariance);
        }
        let mean = |v: Vec<Tensor>| -> Result<Tensor> {
            let n = v.len() as f64;
            Tensor::stack(&v, 0)?.sum_all()? / n
        };
        let goal = nn::mse(&g_hat, &targets[horizon])?;
        let terms =
            JepaTerms { invariance: mean(inv)?, variance: mean(var)?, covariance: mean(cov)?, goal, policy: mean(bc)? };
        Ok((Tensor::stack(&states, 1)?, terms))
    }

    /// EMA update of the target encoder: `θ̄ ← τ θ̄ + (1 − τ) θ`.
    pub fn ema(online: &ParamStore, target: &ParamStore, decay: f64) -> Result<()> {
        for name in target.names().iter().filter(|n| n.starts_with(CTX_ENC)) {
            let (Some(t), Some(o)) = (target.var(name), online.var(name)) else {
                candle_core::bail!("EMA: parameter '{name}' missing from the online store")
            };
            let blended = ((t.as_tensor() * decay)? + (o.as_tensor().detach() * (1.0 - decay))?)?;
            t.set(&blended)?;
        }
        Ok(())
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedJepa> {
        Ok(PackedJepa {
            ctx_enc: self.ctx_enc.pack(dtype)?,
            goal: self.goal.pack(dtype)?,
            world: self.world.pack(dtype)?,
            policy: self.policy.pack(dtype)?,
        })
    }
}

/// Inference-side JEPA weights.
#[derive(Debug, Clone)]
pub struct PackedJepa {
    pub ctx_enc: PackedMlp,
    pub goal: PackedMlp,
    pub world: PackedWorldModel,
    pub policy: PackedMlp,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn ema_moves_target_towards_online() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EngineConfig::tiny(10, 8, 8);
        let (mut on, mut tg) = (ParamStore::new(&dev, 1), ParamStore::new(&dev, 2));
        let jepa = Jepa::new(&mut on, &mut tg, &cfg)?;
        let w = |m: &Mlp| m.l1.w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(w(&jepa.ctx_enc), w(&jepa.target_enc), "target starts as a copy");
        let v = on.var("jepa.ctx_enc.fc1.weight").unwrap();
        v.set(&(v.as_tensor() + 1.0)?)?;
        Jepa::ema(&on, &tg, 0.75)?;
        let (o, t) = (w(&jepa.ctx_enc), w(&jepa.target_enc));
        for (o, t) in o.iter().zip(&t) {
            assert!((t - (o - 0.75)).abs() < 1e-5, "θ̄ = 0.75 θ̄ + 0.25 θ");
        }
        Ok(())
    }

    #[test]
    fn forward_train_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EngineConfig::tiny(10, 8, 8);
        let (mut on, mut tg) = (ParamStore::new(&dev, 1), ParamStore::new(&dev, 2));
        let jepa = Jepa::new(&mut on, &mut tg, &cfg)?;
        let readouts: Vec<Tensor> = (0..=cfg.jepa.horizon)
            .map(|_| Tensor::randn(0f32, 1.0, (16, cfg.ttt.d_ctx), &dev))
            .collect::<Result<_>>()?;
        let (plan, terms) = jepa.forward_train(&readouts, &cfg.jepa)?;
        assert_eq!(plan.dims(), &[16, cfg.plan_len(), cfg.jepa.d_state]);
        let total = terms.weighted(&cfg.jepa)?.to_scalar::<f32>()?;
        assert!(total.is_finite());
        Ok(())
    }
}
