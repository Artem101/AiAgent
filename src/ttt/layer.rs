//! TTT-Linear layer: projections θ_K, θ_V, θ_Q (optionally over a short causal window of
//! tokens), the adaptive inner learning rate and the recurrent scan over `W_fast`.

use candle_core::{DType, Result, Tensor};

use super::fast_weights::{batched_apply, batched_step};
use crate::config::TTTConfig;
use crate::kernels::{l2_normalize, sigmoid, PackedLinear};
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
    /// `[B, T, P]` readout-pool gates `σ(w·x̃_t + b)` (when pools are enabled).
    pub gate: Option<Tensor>,
    /// `[B, T, w·d_model]` causal windows `x̃_t` the projections were computed from.
    pub window: Tensor,
}

/// Snapshot of the recurrent state after a prefix.
pub struct TttSnapshot {
    /// `W_fast` after the prefix, `[B, d, d]`.
    pub w: Tensor,
    /// Running mean of the layer outputs `z_t = W_t q_t`, `[B, d]`.
    pub z_mean: Tensor,
    /// The last outputs, newest first (`z_T, z_{T−1}, …`, zeros before the start), `[B, d]` each.
    pub z_last: Vec<Tensor>,
    /// Gated pools `Σ g z / (Σ g + ε)`, `[B, P·d]` (when pools are enabled).
    pub z_pool: Option<Tensor>,
}

/// Causal windows `[B, n, w·d_model]` and layer outputs `[B, n, d_fast]` of the first `n` tokens.
pub type WindowsAndOutputs = (Tensor, Tensor);

/// Denominator guard of the gated pools (shared by both paths).
pub const POOL_EPS: f64 = 1e-6;

/// `[LN(x_t); LN(x_{t−1}); …]` for a causal window of `w` tokens (zeros before the start):
/// `xn: [B, T, d]` → `[B, T, w·d]`.
pub fn causal_window(xn: &Tensor, w: usize) -> Result<Tensor> {
    if w == 1 {
        return Ok(xn.clone());
    }
    let (b, t, d) = xn.dims3()?;
    let mut parts = vec![xn.clone()];
    for j in 1..w {
        let pad = Tensor::zeros((b, j.min(t), d), xn.dtype(), xn.device())?;
        parts.push(if j >= t { pad } else { Tensor::cat(&[&pad, &xn.narrow(1, 0, t - j)?], 1)? });
    }
    Tensor::cat(&parts, 2)
}

#[derive(Debug, Clone)]
pub struct TttLinear {
    pub wk: Lin,
    pub wv: Lin,
    pub wq: Lin,
    pub eta_gate: Lin,
    pub eta: f64,
    pub adaptive: bool,
    /// Causal window width `w` (the projections take `w · d_model` inputs).
    pub conv_width: usize,
    /// Gates of the readout pools (`P` outputs), if any.
    pub pool_gate: Option<Lin>,
}

impl TttLinear {
    pub fn new(ps: &mut ParamStore, name: &str, cfg: &TTTConfig) -> Result<Self> {
        let d_in = cfg.d_model * cfg.conv_width;
        Ok(Self {
            wk: ps.linear(&format!("{name}.wk"), d_in, cfg.d_fast, false)?,
            wv: ps.linear(&format!("{name}.wv"), d_in, cfg.d_fast, false)?,
            wq: ps.linear(&format!("{name}.wq"), d_in, cfg.d_fast, false)?,
            eta_gate: ps.linear(&format!("{name}.eta_gate"), d_in, 1, true)?,
            eta: cfg.learning_rate,
            adaptive: cfg.adaptive_lr,
            conv_width: cfg.conv_width,
            pool_gate: if cfg.readout_pools > 0 {
                Some(ps.linear(&format!("{name}.pool_gate"), d_in, cfg.readout_pools, true)?)
            } else {
                None
            },
        })
    }

    /// Number of readout pools.
    pub fn pools(&self) -> usize {
        self.pool_gate.as_ref().map_or(0, |g| g.d_out())
    }

    pub fn d_fast(&self) -> usize {
        self.wk.d_out()
    }

    /// `x: [B, T, d_model]` → keys/values/queries/η for every token (one batched matmul each).
    pub fn project(&self, x: &Tensor) -> Result<TttProjections> {
        let xn = causal_window(&nn::layer_norm(x)?, self.conv_width)?;
        let k = nn::l2_normalize(&self.wk.forward(&xn)?)?;
        let v = self.wv.forward(&xn)?;
        let q = nn::l2_normalize(&self.wq.forward(&xn)?)?;
        let eta = if self.adaptive {
            (candle_nn::ops::sigmoid(&self.eta_gate.forward(&xn)?)? * self.eta)?
        } else {
            (xn.narrow(2, 0, 1)?.zeros_like()? + self.eta)?
        };
        let gate = self.pool_gate.as_ref().map(|g| candle_nn::ops::sigmoid(&g.forward(&xn)?)).transpose()?;
        Ok(TttProjections { k, v, q, eta, gate, window: xn })
    }

    /// Runs the online-GD recurrence over `x: [B, T, d_model]` starting from `W = 0` and
    /// returns the state after each prefix length in `snapshots` (ascending, ≤ T), keeping the
    /// last `last` outputs in each snapshot.
    pub fn scan(&self, x: &Tensor, snapshots: &[usize], last: usize) -> Result<Vec<TttSnapshot>> {
        Ok(self.scan_with_outputs(x, snapshots, last, 0)?.0)
    }

    /// [`TttLinear::scan`] that also returns, for the first `keep` tokens, their causal windows
    /// `[B, keep, w·d_model]` and layer outputs `[B, keep, d_fast]` (the inputs of the copy
    /// keys); `None` when `keep = 0`.
    pub fn scan_with_outputs(
        &self,
        x: &Tensor,
        snapshots: &[usize],
        last: usize,
        keep: usize,
    ) -> Result<(Vec<TttSnapshot>, Option<WindowsAndOutputs>)> {
        let (b, t_len, _) = x.dims3()?;
        let p = self.project(x)?;
        let d = self.d_fast();
        let mut w = Tensor::zeros((b, d, d), x.dtype(), x.device())?;
        let zeros = Tensor::zeros((b, d), x.dtype(), x.device())?;
        let mut z_sum = zeros.clone();
        let mut recent = vec![zeros; last]; // newest first
        let np = self.pools();
        let (mut pool_num, mut pool_den) =
            (Tensor::zeros((b, np, d), x.dtype(), x.device())?, Tensor::zeros((b, np), x.dtype(), x.device())?);
        let pooled = |num: &Tensor, den: &Tensor| -> Result<Option<Tensor>> {
            if np == 0 {
                return Ok(None);
            }
            Ok(Some(num.broadcast_div(&(den + POOL_EPS)?.unsqueeze(2)?)?.reshape((b, np * d))?))
        };
        let mut out = Vec::with_capacity(snapshots.len());
        let mut kept = Vec::with_capacity(keep);
        let mut next = snapshots.iter().peekable();
        while next.peek() == Some(&&0) {
            let z_pool = pooled(&pool_num, &pool_den)?;
            out.push(TttSnapshot { w: w.clone(), z_mean: z_sum.clone(), z_last: recent.clone(), z_pool });
            next.next();
        }
        for t in 0..t_len {
            let at = |m: &Tensor| m.narrow(1, t, 1)?.squeeze(1);
            w = batched_step(&w, &at(&p.k)?, &at(&p.v)?, &at(&p.eta)?)?;
            let z = batched_apply(&w, &at(&p.q)?)?;
            if t < keep {
                kept.push(z.clone());
            }
            z_sum = (z_sum + &z)?;
            if let Some(gate) = &p.gate {
                let g = at(gate)?; // [B, P]
                pool_num = (pool_num + g.unsqueeze(2)?.broadcast_mul(&z.unsqueeze(1)?)?)?;
                pool_den = (pool_den + g)?;
            }
            if last > 0 {
                recent.pop();
                recent.insert(0, z);
            }
            while next.peek() == Some(&&(t + 1)) {
                let z_pool = pooled(&pool_num, &pool_den)?;
                out.push(TttSnapshot {
                    w: w.clone(),
                    z_mean: (&z_sum / (t + 1) as f64)?,
                    z_last: recent.clone(),
                    z_pool,
                });
                next.next();
            }
        }
        if out.len() != snapshots.len() {
            candle_core::bail!("TTT scan: snapshot positions {snapshots:?} must be ascending and ≤ {t_len}")
        }
        let outputs = if keep > 0 {
            if kept.len() != keep {
                candle_core::bail!("TTT scan: cannot keep {keep} outputs of {t_len} tokens")
            }
            Some((p.window.narrow(1, 0, keep)?, Tensor::stack(&kept, 1)?))
        } else {
            None
        };
        Ok((out, outputs))
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
            conv_width: self.conv_width,
            pool_gate: self.pool_gate.as_ref().map(|g| g.pack(dtype)).transpose()?,
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
    pub conv_width: usize,
    pub pool_gate: Option<PackedLinear>,
}

impl PackedTttLinear {
    /// Number of readout pools.
    pub fn pools(&self) -> usize {
        self.pool_gate.as_ref().map_or(0, |g| g.d_out)
    }

    /// Readout-pool gates `σ(w·x̃ + b)` of one token's causal window.
    #[inline]
    pub fn gates(&self, xn: &[f32], out: &mut [f32]) {
        if let Some(g) = &self.pool_gate {
            g.forward(xn, out);
            out.iter_mut().for_each(|v| *v = sigmoid(*v));
        }
    }

    pub fn d_model(&self) -> usize {
        self.wk.d_in / self.conv_width
    }
    pub fn d_fast(&self) -> usize {
        self.wk.d_out
    }

    /// Projects one token into `(k, v, q)` and returns η_t. `xn` is its causal window
    /// `[LN(x_t); LN(x_{t−1}); …]` (`conv_width · d_model`, see [`causal_window`]).
    #[inline]
    pub fn project(&self, xn: &[f32], k: &mut [f32], v: &mut [f32], q: &mut [f32]) -> f32 {
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
        self.wk.bytes()
            + self.wv.bytes()
            + self.wq.bytes()
            + self.eta_gate.bytes()
            + self.pool_gate.as_ref().map_or(0, |g| g.bytes())
    }
}
