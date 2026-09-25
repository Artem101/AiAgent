//! The full trainable model: TTT encoder + JEPA + CFM decoder + unembedding head.
//!
//! Joint objective per batch (graph path, autograd):
//!
//! ```text
//!   L = L_CFM + λ_ce CE(head(x̂_1), y) + λ_head CE(head(X_1 + 0.1ε), y)
//!       + λ_inv Inv + λ_var Var + λ_cov Cov + λ_goal ‖G(s_0) − s̄_H‖² + λ_π ‖π(ŝ_t, ĝ) − a_t‖²
//!       [+ λ_probe mean_{h ∈ S} −log P(y | Θ_probe ŝ_h)]
//! ```
//!
//! where `x̂_1 = X_t + (1 − t) v_θ` is the one-step estimate of the clean sample and the
//! decoder is conditioned on the teacher-forced latent plan `[s_0, ŝ_1, …, ŝ_H]`. The thought
//! probe is optional (`JepaConfig::probe_weight > 0`); `S` holds `s_0`, `s_H` and random
//! intermediate thoughts (all of them by default, see `TrainConfig::probe_states`); `P` is the softmax of the shared head,
//! or its mixture with pointers into the prompt when the copy mechanism is on
//! (`JepaConfig::copy_dim > 0`, see [`crate::copy`]; the pointer's own loss is then added).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, Var};

use crate::config::{EngineConfig, TrainConfig};
use crate::data::Batch;
use crate::flow::{self, UnembedHead, VectorField};
use crate::jepa::Jepa;
use crate::kernels::rng::Rng;
use crate::nn::{self, Init, ParamStore};
use crate::ttt::{TttEncoder, SEGMENT_ANSWER, SEGMENT_PROMPT};

/// The complete trainable system (graph path): TTT encoder, JEPA, CFM vector field and
/// unembedding head, with their parameter stores. Pack it into a [`crate::CognitiveEngine`]
/// for inference.
pub struct CogModel {
    /// Architecture this model was built with (needed to load checkpoints).
    pub cfg: EngineConfig,
    /// Trainable parameters (optimised by AdamW).
    pub online: ParamStore,
    /// Non-trainable state: EMA target encoder and frozen buffers.
    pub target: ParamStore,
    pub ttt: TttEncoder,
    pub jepa: Jepa,
    pub flow: VectorField,
    pub head: UnembedHead,
    /// Frozen token embeddings `[vocab, d_token]` — the data points `X_1` of the flow.
    pub out_emb: Tensor,
    device: Device,
}

/// Scalar values of every loss term (for logging).
#[derive(Debug, Clone, Copy, Default)]
pub struct LossReport {
    pub total: f32,
    pub cfm: f32,
    pub ce: f32,
    pub head: f32,
    pub inv: f32,
    pub var: f32,
    pub cov: f32,
    pub goal: f32,
    pub policy: f32,
    /// Thought-probe cross-entropy (0 when the probe is disabled), including the pointer loss.
    pub probe: f32,
    /// The copy pointer's own loss (0 without a copy mechanism).
    pub pointer: f32,
}

impl LossReport {
    /// Adds another report term by term (for running averages).
    pub fn accumulate(&mut self, o: &Self) {
        self.total += o.total;
        self.cfm += o.cfm;
        self.ce += o.ce;
        self.head += o.head;
        self.inv += o.inv;
        self.var += o.var;
        self.cov += o.cov;
        self.goal += o.goal;
        self.policy += o.policy;
        self.probe += o.probe;
        self.pointer += o.pointer;
    }
    /// Multiplies every term by `s`.
    pub fn scaled(&self, s: f32) -> Self {
        Self {
            total: self.total * s,
            cfm: self.cfm * s,
            ce: self.ce * s,
            head: self.head * s,
            inv: self.inv * s,
            var: self.var * s,
            cov: self.cov * s,
            goal: self.goal * s,
            policy: self.policy * s,
            probe: self.probe * s,
            pointer: self.pointer * s,
        }
    }
}

impl std::fmt::Display for LossReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "loss {:.4} | cfm {:.4} ce {:.4} head {:.4} | inv {:.4} var {:.4} cov {:.4} goal {:.4} π {:.4}",
            self.total, self.cfm, self.ce, self.head, self.inv, self.var, self.cov, self.goal, self.policy
        )?;
        if self.probe != 0.0 {
            write!(f, " probe {:.4}", self.probe)?;
        }
        if self.pointer != 0.0 {
            write!(f, " ptr {:.4}", self.pointer)?;
        }
        Ok(())
    }
}

fn randn(rng: &mut Rng, shape: &[usize], std: f32, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; shape.iter().product()];
    rng.fill_normal(&mut v, std);
    Tensor::from_vec(v, shape, device)
}

impl CogModel {
    /// Builds a deterministically initialised model on `device` (validates `cfg` first).
    pub fn new(cfg: EngineConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let mut online = ParamStore::new(device, cfg.seed);
        let mut target = ParamStore::new(device, cfg.seed ^ 0x7A11);
        let ttt = TttEncoder::new(&mut online, &cfg)?;
        let jepa = Jepa::new(&mut online, &mut target, &cfg)?;
        let flow = VectorField::new(&mut online, &cfg)?;
        let head = UnembedHead::new(&mut online, cfg.flow.d_token, cfg.vocab_size)?;
        let out_emb = target.tensor("buffers.out_emb", &[cfg.vocab_size, cfg.flow.d_token], Init::Normal(1.0))?;
        Ok(Self { cfg, online, target, ttt, jepa, flow, head, out_emb, device: device.clone() })
    }

    /// Device of the graph-path parameters.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Parameters the optimiser updates (excludes the EMA target encoder and frozen buffers).
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.online.vars()
    }

    /// Number of trainable scalars.
    pub fn num_params(&self) -> usize {
        self.online.num_params()
    }

    /// TTT readouts after the prompt and after each answer chunk (`H + 1` tensors `[B, d_ctx]`),
    /// and the copy keys of the prompt tokens (`[B, N, d_key]`, with a copy mechanism).
    fn readouts(&self, batch: &Batch, dtype: DType) -> Result<(Vec<Tensor>, Option<Tensor>)> {
        let n = batch.prompt.dims2()?.1;
        let (l, h) = (self.cfg.answer_len(), self.cfg.jepa.horizon);
        let chunk = l / h;
        let x = Tensor::cat(
            &[self.ttt.embed(&batch.prompt, 0, SEGMENT_PROMPT)?, self.ttt.embed(&batch.answer, n, SEGMENT_ANSWER)?],
            1,
        )?
        .to_dtype(dtype)?;
        let snaps: Vec<usize> = (0..=h).map(|i| n + i * chunk).collect();
        self.ttt.encode_with_keys(&x, &snaps, n)
    }

    /// Teacher-forced latent plan `[B, H + 1, d_s]` (uses the answer — evaluation oracle).
    pub fn teacher_plan(&self, batch: &Batch, dtype: DType) -> Result<Tensor> {
        let (readouts, _) = self.readouts(batch, dtype)?;
        Ok(self.jepa.forward_train(&readouts, &self.cfg.jepa)?.0)
    }

    /// Computes the joint loss. Returns the differentiable total and a scalar report.
    pub fn loss(&self, batch: &Batch, rng: &mut Rng, tc: &TrainConfig) -> Result<(Tensor, LossReport)> {
        let cd = tc.resolved_compute_dtype(&self.device);
        let (b, l) = batch.answer.dims2()?;
        let (dt, vocab) = (self.cfg.flow.d_token, self.cfg.vocab_size);

        // TTT → JEPA
        let (readouts, keys) = self.readouts(batch, cd)?;
        let (mut plan, terms) = self.jepa.forward_train(&readouts, &self.cfg.jepa)?;
        let targets = batch.answer.flatten_all()?;
        let mut probe_ce = None;
        let mut copy_aux = None;
        if let Some(probe) = &self.jepa.probe {
            // every thought s_0 … s_H must decode into the answer through the shared head —
            // and so must the goal ĝ the latent search steers towards (`probe_goal`)
            let h = self.cfg.jepa.horizon;
            let mut states = if tc.probe_states >= 2 && tc.probe_states < h + 1 {
                // s_0, s_H and a random subset of the intermediate thoughts (a new one each step)
                let mut idx = vec![0u32, h as u32];
                while idx.len() < tc.probe_states {
                    let i = 1 + rng.below(h - 1) as u32;
                    if !idx.contains(&i) {
                        idx.push(i);
                    }
                }
                let idx = Tensor::from_vec(idx, tc.probe_states, &self.device)?;
                plan.index_select(&idx, 1)?
            } else {
                plan.clone()
            };
            if tc.probe_goal {
                let s0 = plan.narrow(1, 0, 1)?.squeeze(1)?;
                states = Tensor::cat(&[states, self.jepa.goal_of(&s0)?.unsqueeze(1)?], 1)?;
            }
            let n = states.dims()[1];
            let states = states.reshape((b * n, self.cfg.jepa.d_state))?;
            let emb = probe.forward(&states)?.reshape((b * n, l, dt))?;
            let logits = self.head.forward(&emb)?.to_dtype(DType::F32)?;
            let per_state = |t: &Tensor| -> Result<Tensor> {
                // [B, …] → [B·n, …], row b·n + j for thought j of example b
                let mut dims = t.dims().to_vec();
                let mut rep = vec![1; dims.len() + 1];
                rep[1] = n;
                let r = t.unsqueeze(1)?.repeat(rep)?;
                dims[0] *= n;
                r.reshape(dims)
            };
            let tgt = per_state(&batch.answer)?;
            probe_ce = Some(match (&self.jepa.copy, &keys) {
                (Some(copy), Some(keys)) => {
                    let (ll, aux) = copy.losses(&emb, &logits, &per_state(keys)?, &tgt, &per_state(&batch.prompt)?)?;
                    copy_aux = Some(aux.clone());
                    (ll.mean_all()?.neg()? + (aux * crate::copy::POINTER_WEIGHT)?)?
                }
                _ => candle_nn::loss::cross_entropy(&logits.reshape((b * n * l, vocab))?, &tgt.flatten_all()?)?,
            });
        }
        if tc.plan_noise > 0.0 {
            let noise = randn(rng, plan.dims(), tc.plan_noise as f32, &self.device)?.to_dtype(cd)?;
            plan = (plan + noise)?;
        }

        // CFM on the conditional OT path
        let x1 = self.out_emb.index_select(&targets, 0)?.reshape((b, l, dt))?.to_dtype(cd)?;
        let x0 = randn(rng, &[b, l, dt], 1.0, &self.device)?.to_dtype(cd)?;
        let mut tv = vec![0f32; b];
        rng.fill_uniform(&mut tv, 0.0, 1.0);
        let t = Tensor::from_vec(tv, b, &self.device)?;
        let (xt, ut) = flow::ot_path(&x0, &x1, &t, self.cfg.flow.solver.sigma_min as f64)?;
        let v = self.flow.forward(&xt, &t, &plan)?;
        let cfm = nn::mse(&v, &ut)?;

        // one-step clean estimate → token CE; head CE on noisy clean embeddings
        let remaining = (t.neg()? + 1.0)?.reshape((b, 1, 1))?.to_dtype(cd)?;
        let x1_hat = (&xt + v.broadcast_mul(&remaining)?)?;
        let logits = self.head.forward(&x1_hat)?.to_dtype(DType::F32)?.reshape((b * l, vocab))?;
        let ce = candle_nn::loss::cross_entropy(&logits, &targets)?;
        let noisy = (&x1 + randn(rng, &[b, l, dt], 0.1, &self.device)?.to_dtype(cd)?)?;
        let head_logits = self.head.forward(&noisy)?.to_dtype(DType::F32)?.reshape((b * l, vocab))?;
        let head_ce = candle_nn::loss::cross_entropy(&head_logits, &targets)?;

        let mut total =
            ((((&cfm + (&ce * tc.ce_weight)?)? + (&head_ce * tc.head_weight)?)?) + terms.weighted(&self.cfg.jepa)?)?;
        if let Some(p) = &probe_ce {
            total = (total + (p * self.cfg.jepa.probe_weight)?)?;
        }
        let s = |t: &Tensor| t.to_dtype(DType::F32)?.to_scalar::<f32>();
        let report = LossReport {
            total: s(&total)?,
            cfm: s(&cfm)?,
            ce: s(&ce)?,
            head: s(&head_ce)?,
            inv: s(&terms.invariance)?,
            var: s(&terms.variance)?,
            cov: s(&terms.covariance)?,
            goal: s(&terms.goal)?,
            policy: s(&terms.policy)?,
            probe: probe_ce.as_ref().map(s).transpose()?.unwrap_or(0.0),
            pointer: copy_aux.as_ref().map(s).transpose()?.unwrap_or(0.0),
        };
        Ok((total, report))
    }

    /// EMA step of the JEPA target encoder.
    pub fn ema_update(&self) -> Result<()> {
        Jepa::ema(&self.online, &self.target, self.cfg.jepa.ema_decay)
    }

    /// Decodes `plan: [B, H+1, d_s]` with the batched graph sampler → token ids.
    pub fn decode_graph(&self, plan: &Tensor, rng: &mut Rng) -> Result<Vec<Vec<u32>>> {
        let b = plan.dims()[0];
        let (l, dt) = (self.cfg.answer_len(), self.cfg.flow.d_token);
        let x0 = randn(rng, &[b, l, dt], 1.0, &self.device)?.to_dtype(plan.dtype())?;
        let x1 = flow::sample_graph(&self.flow, plan, &x0, &self.cfg.flow.solver)?;
        self.head.forward(&x1)?.argmax(candle_core::D::Minus1)?.to_dtype(DType::U32)?.to_vec2::<u32>()
    }

    /// Writes all parameters (`online.*`) and non-trainable state (`target.*`) to one
    /// safetensors file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut map = HashMap::new();
        self.online.export("online", &mut map);
        self.target.export("target", &mut map);
        candle_core::safetensors::save(&map, path)
    }

    /// Loads a checkpoint written by [`CogModel::save`] into a model of the same architecture.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let map = candle_core::safetensors::load(path, &self.device)?;
        self.online.import("online", &map)?;
        self.target.import("target", &map)
    }
}
