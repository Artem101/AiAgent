//! Fast weights `W_fast` and the analytic online-gradient step of TTT-Linear.
//!
//! Inner (self-supervised) objective for token t, with key `k_t` and value `v_t`:
//!
//! ```text
//!   ℓ(W; x_t) = ½ ‖W k_t − v_t‖²          ∇_W ℓ = (W k_t − v_t) ⊗ k_t
//!   W_t = W_{t−1} − η_t (W_{t−1} k_t − v_t) ⊗ k_t
//!   z_t = W_t q_t
//! ```
//!
//! With `‖k_t‖ = 1` and `η_t ∈ (0, 1]` the step is a convex combination along `k_t`
//! (`W_t k_t = (1 − η_t) W_{t−1} k_t + η_t v_t`), so the recurrence is stable for any `N`.
//! Memory is `d²` floats regardless of the context length.

use candle_core::{DType, Device, Result, Tensor};

use crate::arena::Arena;
use crate::kernels::inplace::{self, is_host_f32, MatVecInto, Rank1Update};
use crate::kernels::{matvec_f32, rank1_update};

/// State of the fast weights. Memory is fixed at `O(d_fast²)`.
#[derive(Debug)]
pub struct FastWeightsState {
    /// `[d_fast, d_fast]`, f32 (it is a gradient accumulator).
    pub weights: Tensor,
    /// Default inner learning rate η.
    pub eta: f64,
}

impl FastWeightsState {
    pub fn new(dim: usize, eta: f64, device: &Device) -> Result<Self> {
        Ok(Self { weights: Tensor::zeros((dim, dim), DType::F32, device)?, eta })
    }

    pub fn in_arena(arena: &mut Arena, dim: usize, eta: f64) -> Result<Self> {
        Ok(Self { weights: arena.tensor((dim, dim))?, eta })
    }

    pub fn dim(&self) -> usize {
        self.weights.dims()[0]
    }

    /// `W ← 0` (in place on the host).
    pub fn reset(&mut self) -> Result<()> {
        inplace::fill_(&mut self.weights, 0.0)
    }

    /// Online gradient step with the default η: rank-1 update, no autograd graph.
    #[inline(always)]
    pub fn step_update(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.step_update_eta(k, v, self.eta)
    }

    /// Online gradient step with an explicit (e.g. adaptive, per-token) η.
    ///
    /// Host f32 tensors take a fused in-place kernel (one pass over `W`, zero allocations);
    /// other devices use the reference graph formulation.
    pub fn step_update_eta(&mut self, k: &Tensor, v: &Tensor, eta: f64) -> Result<()> {
        if is_host_f32(&self.weights) && is_host_f32(k) && is_host_f32(v) {
            return self.weights.inplace_op3(k, v, &Rank1Update(eta as f32));
        }
        // pred = W k ; err = pred − v ; W ← W − η err ⊗ k
        let pred = self.weights.matmul(&k.unsqueeze(1)?)?.squeeze(1)?;
        let err = (&pred - v)?;
        let delta = err.unsqueeze(1)?.matmul(&k.unsqueeze(0)?)?;
        self.weights = (&self.weights - (delta * eta)?)?;
        Ok(())
    }

    /// Same step on raw host slices (used by the packed encoder's token loop).
    #[inline]
    pub fn step_update_host(&mut self, k: &[f32], v: &[f32], eta: f32) -> Result<()> {
        inplace::host_write(&self.weights, |w| {
            rank1_update(w, k, v, eta);
            Ok(())
        })
    }

    /// `out ← W q` without allocating (host) / by rebinding `out` (other devices).
    pub fn forward_into(&self, q: &Tensor, out: &mut Tensor) -> Result<()> {
        if is_host_f32(&self.weights) && is_host_f32(q) && is_host_f32(out) {
            return out.inplace_op3(&self.weights, q, &MatVecInto);
        }
        *out = self.forward(q)?;
        Ok(())
    }

    /// `W q` on raw host slices.
    #[inline]
    pub fn forward_host(&self, q: &[f32], out: &mut [f32]) -> Result<()> {
        inplace::host_read(&self.weights, |w| matvec_f32(w, q, out))
    }

    /// `z = W q` (allocating).
    pub fn forward(&self, q: &Tensor) -> Result<Tensor> {
        self.weights.matmul(&q.unsqueeze(1)?)?.squeeze(1)
    }

    /// Inner loss `½ ‖W k − v‖²` (diagnostics).
    pub fn reconstruction_loss(&self, k: &Tensor, v: &Tensor) -> Result<f32> {
        (self.forward(k)? - v)?.sqr()?.sum_all()?.to_scalar::<f32>().map(|s| 0.5 * s)
    }
}

/// Differentiable batched step for training: `W: [B, d, d]`, `k, v: [B, d]`, `eta: [B, 1]`.
pub fn batched_step(w: &Tensor, k: &Tensor, v: &Tensor, eta: &Tensor) -> Result<Tensor> {
    let err = (batched_apply(w, k)? - v)?; // [B, d]
    let delta = err.unsqueeze(2)?.matmul(&k.unsqueeze(1)?)?; // [B, d, d]
    w - delta.broadcast_mul(&eta.unsqueeze(2)?)?
}

/// `z = W q` batched: `W: [B, d, d]`, `q: [B, d]` → `[B, d]`.
pub fn batched_apply(w: &Tensor, q: &Tensor) -> Result<Tensor> {
    w.matmul(&q.unsqueeze(2)?)?.squeeze(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::inplace::copy_from_slice;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn fused_kernel_matches_reference_graph() -> Result<()> {
        let dev = Device::Cpu;
        let d = 6;
        let mut fast = FastWeightsState::new(d, 0.7, &dev)?;
        let mut reference = Tensor::zeros((d, d), DType::F32, &dev)?;
        for t in 0..5 {
            let kv: Vec<f32> = (0..d).map(|i| ((i * 7 + t * 3) as f32 * 0.31).sin()).collect();
            let k = Tensor::new(unit(&kv).as_slice(), &dev)?;
            let v = Tensor::new((0..d).map(|i| ((i + t) as f32 * 0.5).cos()).collect::<Vec<_>>().as_slice(), &dev)?;
            fast.step_update(&k, &v)?;
            let pred = reference.matmul(&k.unsqueeze(1)?)?.squeeze(1)?;
            let delta = (pred - &v)?.unsqueeze(1)?.matmul(&k.unsqueeze(0)?)?;
            reference = (reference - (delta * 0.7)?)?;
        }
        let diff = (&fast.weights - &reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "diff {diff}");
        Ok(())
    }

    #[test]
    fn step_reduces_inner_loss_and_memory_is_constant() -> Result<()> {
        let dev = Device::Cpu;
        let mut fast = FastWeightsState::new(4, 0.5, &dev)?;
        let k = Tensor::new(unit(&[1.0, 2.0, -1.0, 0.5]).as_slice(), &dev)?;
        let v = Tensor::new(&[0.3f32, -0.2, 0.9, 0.1], &dev)?;
        let before = fast.reconstruction_loss(&k, &v)?;
        let bytes = fast.weights.elem_count();
        fast.step_update(&k, &v)?;
        assert!(fast.reconstruction_loss(&k, &v)? < before);
        assert_eq!(fast.weights.elem_count(), bytes);
        let mut out = Tensor::zeros(4, DType::F32, &dev)?;
        fast.forward_into(&k, &mut out)?;
        let alloc = fast.forward(&k)?;
        assert!((out - alloc)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-6);
        fast.reset()?;
        assert_eq!(fast.weights.abs()?.sum_all()?.to_scalar::<f32>()?, 0.0);
        copy_from_slice(&fast.weights, &[1.0; 16])?;
        assert_eq!(fast.weights.sum_all()?.to_scalar::<f32>()?, 16.0);
        Ok(())
    }

    #[test]
    fn batched_matches_single() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1.0, (2, 3, 3), &dev)?;
        let k = crate::nn::l2_normalize(&Tensor::randn(0f32, 1.0, (2, 3), &dev)?)?;
        let v = Tensor::randn(0f32, 1.0, (2, 3), &dev)?;
        let eta = Tensor::new(&[[0.3f32], [0.9]], &dev)?;
        let out = batched_step(&w, &k, &v, &eta)?;
        for b in 0..2 {
            let mut s = FastWeightsState { weights: w.get(b)?.contiguous()?, eta: [0.3, 0.9][b] };
            s.step_update(&k.get(b)?.contiguous()?, &v.get(b)?.contiguous()?)?;
            let diff = (s.weights - out.get(b)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-5);
        }
        Ok(())
    }
}
