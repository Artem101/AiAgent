//! Module 3 — Continuous Flow Matching decoder.
//!
//! Probability path from noise `X_0 ~ N(0, I)` to data `X_1` (token embeddings):
//!
//! ```text
//!   X_t = (1 − (1 − σ_min) t) X_0 + t X_1          u_t = X_1 − (1 − σ_min) X_0
//!   L_CFM = E_{t, X_0, X_1} ‖v_θ(X_t, t, S_plan) − u_t‖²
//! ```
//!
//! After integration, `Tokens = argmax(W_head X_1)` in a single parallel step.

pub mod ode_solver;
pub mod vector_field;

use candle_core::{DType, Result, Tensor};

pub use ode_solver::{FlowMatchingSampler, ODESolverConfig, SamplerBuffers, SolverKind, VectorFieldEstimator};
pub use vector_field::{GraphVectorField, PackedVectorField, VectorField};

use crate::kernels::{argmax, PackedLinear};
use crate::nn::{Lin, ParamStore};

/// Conditional OT path: returns `(X_t, u_t)` for `x0, x1: [B, L, d]`, `t: [B]`.
pub fn ot_path(x0: &Tensor, x1: &Tensor, t: &Tensor, sigma_min: f64) -> Result<(Tensor, Tensor)> {
    let b = t.dims1()?;
    let t3 = t.reshape((b, 1, 1))?.to_dtype(x0.dtype())?;
    let coef0 = ((&t3 * -(1.0 - sigma_min))? + 1.0)?;
    let xt = (x0.broadcast_mul(&coef0)? + x1.broadcast_mul(&t3)?)?;
    let ut = (x1 - (x0 * (1.0 - sigma_min))?)?;
    Ok((xt, ut))
}

/// Unembedding head `d_token → vocab`.
#[derive(Debug, Clone)]
pub struct UnembedHead {
    pub lin: Lin,
}

impl UnembedHead {
    pub fn new(ps: &mut ParamStore, d_token: usize, vocab: usize) -> Result<Self> {
        Ok(Self { lin: ps.linear("head", d_token, vocab, true)? })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lin.forward(x)
    }
    pub fn pack(&self, dtype: DType) -> Result<PackedHead> {
        Ok(PackedHead { lin: self.lin.pack(dtype)? })
    }
}

/// Host-kernel unembedding head.
#[derive(Debug, Clone)]
pub struct PackedHead {
    pub lin: PackedLinear,
}

impl PackedHead {
    /// `tokens[i] = argmax_v (W_head x_i + b)_v` for every row of `x` (one parallel step).
    pub fn decode_into(&self, x: &[f32], logits: &mut [f32], tokens: &mut [u32]) {
        let v = self.lin.d_out;
        self.lin.forward(x, logits);
        for (tok, row) in tokens.iter_mut().zip(logits.chunks_exact(v)) {
            *tok = argmax(row) as u32;
        }
    }
    /// Vocabulary size (number of logits per position).
    pub fn vocab(&self) -> usize {
        self.lin.d_out
    }
}

/// Batched graph-path sampler (evaluation / accelerators): integrates `x0: [B, L, d]`
/// conditioned on `plan: [B, H+1, d_s]`.
pub fn sample_graph(vf: &VectorField, plan: &Tensor, x0: &Tensor, cfg: &ODESolverConfig) -> Result<Tensor> {
    let b = x0.dims()[0];
    let dt = 1.0 / cfg.steps as f64;
    let tvec = |t: f64| Tensor::full(t as f32, b, x0.device());
    let mut x = x0.clone();
    for step in 0..cfg.steps {
        let t = step as f64 * dt;
        x = match cfg.solver {
            SolverKind::Euler => (&x + (vf.forward(&x, &tvec(t)?, plan)? * dt)?)?,
            SolverKind::Midpoint => {
                let v1 = vf.forward(&x, &tvec(t)?, plan)?;
                let xm = (&x + (v1 * (0.5 * dt))?)?;
                (&x + (vf.forward(&xm, &tvec(t + 0.5 * dt)?, plan)? * dt)?)?
            }
            SolverKind::Heun => {
                let v1 = vf.forward(&x, &tvec(t)?, plan)?;
                let xp = (&x + (&v1 * dt)?)?;
                let v2 = vf.forward(&xp, &tvec(t + dt)?, plan)?;
                (&x + ((v1 + v2)? * (0.5 * dt))?)?
            }
        };
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn ot_path_endpoints() -> Result<()> {
        let dev = Device::Cpu;
        let x0 = Tensor::randn(0f32, 1.0, (2, 3, 4), &dev)?;
        let x1 = Tensor::randn(0f32, 1.0, (2, 3, 4), &dev)?;
        let (xt0, _) = ot_path(&x0, &x1, &Tensor::new(&[0f32, 0.], &dev)?, 0.0)?;
        let (xt1, ut) = ot_path(&x0, &x1, &Tensor::new(&[1f32, 1.], &dev)?, 0.0)?;
        assert!((xt0 - &x0)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        assert!((xt1 - &x1)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        assert!((ut - (&x1 - &x0)?)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        Ok(())
    }
}
