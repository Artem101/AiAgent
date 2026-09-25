//! TTT-Linear layer: projections θ_K, θ_V, θ_Q, the adaptive inner learning rate and the
//! recurrent scan over `W_fast`.

use candle_core::{DType, Result, Tensor};

use super::fast_weights::{batched_apply, batched_step};
use crate::config::TTTConfig;
use crate::kernels::{l2_normalize, layer_norm_rows_into, sigmoid, PackedLinear};
use crate::nn::{self, Lin, ParamStore};

/// Per-token projections of a sequence (graph path).
pub struct TttProjections {
    /// `[B, T, d_fast]`, unit norm.
    pub k: Tensor,
    /// `[B, T, d_fast]`.
    pub v: Tensor,
    /// `[B, T, d_fast]`, unit norm.
    pub q: Tensor,
    /// `[B, T, 1]` inner learning rate η_t.
    pub eta: Tensor,
}

/// Snapshot of the recurrent state after a prefix.
pub struct TttSnapshot {
    /// `W_fast` after the prefix, `[B, d, d]`.
    pub w: Tensor,
    /// Running mean of the layer outputs `z_t = W_t q_t`, `[B, d]`.
    pub z_mean: Tensor,
}

#[derive(Debug, Clone)]
pub struct TttLinear {
    pub wk: Lin,
    pub wv: Lin,
    pub wq: Lin,
    pub eta_gate: Lin,
    pub eta: f64,
    pub adaptive: bool,
}

impl TttLinear {
    pub fn new(ps: &mut ParamStore, name: &str, cfg: &TTTConfig) -> Result<Self> {
        Ok(Self {
            wk: ps.linear(&format!("{name}.wk"), cfg.d_model, cfg.d_fast, false)?,
            wv: ps.linear(&format!("{name}.wv"), cfg.d_model, cfg.d_fast, false)?,
            wq: ps.linear(&format!("{name}.wq"), cfg.d_model, cfg.d_fast, false)?,
            eta_gate: ps.linear(&format!("{name}.eta_gate"), cfg.d_model, 1, true)?,
            eta: cfg.learning_rate,
            adaptive: cfg.adaptive_lr,
        })
    }

    pub fn d_fast(&self) -> usize {
        self.wk.d_out()
    }

    /// `x: [B, T, d_model]` → keys/values/queries/η for every token (one batched matmul each).
    pub fn project(&self, x: &Tensor) -> Result<TttProjections> {
        let xn = nn::layer_norm(x)?;
        let k = nn::l2_normalize(&self.wk.forward(&xn)?)?;
        let v = self.wv.forward(&xn)?;
        let q = nn::l2_normalize(&self.wq.forward(&xn)?)?;
        let eta = if self.adaptive {
            (candle_nn::ops::sigmoid(&self.eta_gate.forward(&xn)?)? * self.eta)?
        } else {
            (xn.narrow(2, 0, 1)?.zeros_like()? + self.eta)?
        };
        Ok(TttProjections { k, v, q, eta })
    }

    /// Runs the online-GD recurrence over `x: [B, T, d_model]` starting from `W = 0` and
    /// returns the state after each prefix length in `snapshots` (ascending, ≤ T).
    pub fn scan(&self, x: &Tensor, snapshots: &[usize]) -> Result<Vec<TttSnapshot>> {
        let (b, t_len, _) = x.dims3()?;
        let p = self.project(x)?;
        let d = self.d_fast();
        let mut w = Tensor::zeros((b, d, d), x.dtype(), x.device())?;
        let mut z_sum = Tensor::zeros((b, d), x.dtype(), x.device())?;
        let mut out = Vec::with_capacity(snapshots.len());
        let mut next = snapshots.iter().peekable();
        while next.peek() == Some(&&0) {
            out.push(TttSnapshot { w: w.clone(), z_mean: z_sum.clone() });
            next.next();
        }
        for t in 0..t_len {
            let at = |m: &Tensor| m.narrow(1, t, 1)?.squeeze(1);
            w = batched_step(&w, &at(&p.k)?, &at(&p.v)?, &at(&p.eta)?)?;
            z_sum = (z_sum + batched_apply(&w, &at(&p.q)?)?)?;
            while next.peek() == Some(&&(t + 1)) {
                out.push(TttSnapshot { w: w.clone(), z_mean: (&z_sum / (t + 1) as f64)? });
                next.next();
            }
        }
        if out.len() != snapshots.len() {
            candle_core::bail!("TTT scan: snapshot positions {snapshots:?} must be ascending and ≤ {t_len}")
        }
        Ok(out)
    }

    /// Layer outputs `z_t = W_t q_t` for every token, `[B, T, d_fast]`.
    pub fn outputs(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t_len, _) = x.dims3()?;
        let p = self.project(x)?;
        let d = self.d_fast();
        let mut w = Tensor::zeros((b, d, d), x.dtype(), x.device())?;
        let mut zs = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let at = |m: &Tensor| m.narrow(1, t, 1)?.squeeze(1);
            w = batched_step(&w, &at(&p.k)?, &at(&p.v)?, &at(&p.eta)?)?;
            zs.push(batched_apply(&w, &at(&p.q)?)?);
        }
        Tensor::stack(&zs, 1)
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedTttLinear> {
        Ok(PackedTttLinear {
            wk: self.wk.pack(dtype)?,
            wv: self.wv.pack(dtype)?,
            wq: self.wq.pack(dtype)?,
            eta_gate: self.eta_gate.pack(dtype)?,
            eta: self.eta as f32,
            adaptive: self.adaptive,
        })
    }
}

/// Host-kernel version of [`TttLinear`].
#[derive(Debug, Clone)]
pub struct PackedTttLinear {
    pub wk: PackedLinear,
    pub wv: PackedLinear,
    pub wq: PackedLinear,
    pub eta_gate: PackedLinear,
    pub eta: f32,
    pub adaptive: bool,
}

impl PackedTttLinear {
    pub fn d_model(&self) -> usize {
        self.wk.d_in
    }
    pub fn d_fast(&self) -> usize {
        self.wk.d_out
    }

    /// Projects one token `x` into `(k, v, q)` and returns η_t. `xn` is `d_model` scratch.
    #[inline]
    pub fn project(&self, x: &[f32], xn: &mut [f32], k: &mut [f32], v: &mut [f32], q: &mut [f32]) -> f32 {
        layer_norm_rows_into(x, xn, x.len());
        self.wk.forward(xn, k);
        l2_normalize(k);
        self.wv.forward(xn, v);
        self.wq.forward(xn, q);
        l2_normalize(q);
        if self.adaptive {
            let mut g = [0f32; 1];
            self.eta_gate.forward(xn, &mut g);
            self.eta * sigmoid(g[0])
        } else {
            self.eta
        }
    }

    pub fn bytes(&self) -> usize {
        self.wk.bytes() + self.wv.bytes() + self.wq.bytes() + self.eta_gate.bytes()
    }
}
