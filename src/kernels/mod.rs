//! Host (CPU) compute kernels used by the inference path.
//!
//! Conventions:
//! * activations, states and accumulators are `f32`;
//! * weights are stored as [`WeightBuf`] (`bf16` by default, `f16`/`f32` optional) in the
//!   candle `[out, in]` layout and widened to f32 inside the dot product;
//! * no kernel allocates — every output and scratch buffer is passed in by the caller;
//! * work above [`PAR_MIN_WORK`] multiply-accumulates is split across rayon.

pub mod inplace;
pub mod rng;
pub mod simd;

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{bail, DType, Device, Result, Tensor};
use half::{bf16, f16};
use rayon::prelude::*;

/// Minimum multiply-accumulates before a kernel fans out to rayon. Below this, waking the
/// pool costs more than the work itself.
pub const PAR_MIN_WORK: usize = 1 << 20;
/// Minimum MACs per rayon task.
const PAR_TASK_WORK: usize = 1 << 16;
pub const LN_EPS: f32 = 1e-5;
pub const L2_EPS: f32 = 1e-6;

static PARALLEL: AtomicBool = AtomicBool::new(true);

/// Globally enables/disables rayon fan-out inside kernels (results are identical either way).
pub fn set_parallel(on: bool) {
    PARALLEL.store(on, Ordering::Relaxed);
}

#[inline]
pub fn parallel_for(work: usize) -> bool {
    work >= PAR_MIN_WORK && PARALLEL.load(Ordering::Relaxed)
}

/// Storage element of packed weights.
pub trait WeightElem: Copy + Send + Sync + 'static {
    const DTYPE: DType;
    fn to_f32(self) -> f32;
    fn from_f32(x: f32) -> Self;
    /// `Σ w_i x_i`, accumulated in f32.
    fn dot(w: &[Self], x: &[f32]) -> f32;
    /// Four rows against the same `x` (register-blocked).
    fn dot4(w: [&[Self]; 4], x: &[f32]) -> [f32; 4];
}

impl WeightElem for f32 {
    const DTYPE: DType = DType::F32;
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(x: f32) -> Self {
        x
    }
    #[inline]
    fn dot(w: &[Self], x: &[f32]) -> f32 {
        simd::dot_f32(w, x)
    }
    #[inline]
    fn dot4(w: [&[Self]; 4], x: &[f32]) -> [f32; 4] {
        simd::dot4_f32(w, x)
    }
}

impl WeightElem for bf16 {
    const DTYPE: DType = DType::BF16;
    #[inline]
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
    #[inline]
    fn from_f32(x: f32) -> Self {
        bf16::from_f32(x)
    }
    #[inline]
    fn dot(w: &[Self], x: &[f32]) -> f32 {
        simd::dot_bf16(w, x)
    }
    #[inline]
    fn dot4(w: [&[Self]; 4], x: &[f32]) -> [f32; 4] {
        simd::dot4_bf16(w, x)
    }
}

impl WeightElem for f16 {
    const DTYPE: DType = DType::F16;
    #[inline]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
    #[inline]
    fn from_f32(x: f32) -> Self {
        f16::from_f32(x)
    }
    #[inline]
    fn dot(w: &[Self], x: &[f32]) -> f32 {
        simd::dot_f16(w, x)
    }
    #[inline]
    fn dot4(w: [&[Self]; 4], x: &[f32]) -> [f32; 4] {
        simd::dot4_f16(w, x)
    }
}

/// Immutable, fixed-size packed weight storage.
#[derive(Debug, Clone)]
pub enum WeightBuf {
    F32(Box<[f32]>),
    F16(Box<[f16]>),
    BF16(Box<[bf16]>),
}

/// Runs `$body` with `$w` bound to the typed weight slice.
macro_rules! with_weights {
    ($buf:expr, |$w:ident| $body:expr) => {
        match $buf {
            $crate::kernels::WeightBuf::F32($w) => $body,
            $crate::kernels::WeightBuf::F16($w) => $body,
            $crate::kernels::WeightBuf::BF16($w) => $body,
        }
    };
}

impl WeightBuf {
    /// Copies a tensor (any device / dtype) into host storage of the requested precision.
    pub fn from_tensor(t: &Tensor, dtype: DType) -> Result<Self> {
        let flat = t.flatten_all()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(match dtype {
            DType::F32 => Self::F32(flat.into_boxed_slice()),
            DType::F16 => Self::F16(flat.iter().map(|&v| f16::from_f32(v)).collect()),
            DType::BF16 => Self::BF16(flat.iter().map(|&v| bf16::from_f32(v)).collect()),
            other => bail!("unsupported weight storage dtype {other:?} (use f32 | f16 | bf16)"),
        })
    }

    pub fn len(&self) -> usize {
        with_weights!(self, |w| w.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
        }
    }

    pub fn bytes(&self) -> usize {
        self.len() * self.dtype().size_in_bytes()
    }

    /// Widens row `r` (of width `cols`) into `out`.
    pub fn row_into(&self, r: usize, cols: usize, out: &mut [f32]) {
        with_weights!(self, |w| widen_into(&w[r * cols..(r + 1) * cols], out))
    }

    /// Adds row `r` (of width `cols`) into `out`.
    pub fn add_row_into(&self, r: usize, cols: usize, out: &mut [f32]) {
        with_weights!(self, |w| {
            for (o, &v) in out.iter_mut().zip(&w[r * cols..(r + 1) * cols]) {
                *o += v.to_f32();
            }
        })
    }

    /// Back to a candle tensor (f32) — used to rebuild graph modules from packed weights.
    pub fn to_tensor(&self, shape: &[usize], device: &Device) -> Result<Tensor> {
        let v: Vec<f32> = with_weights!(self, |w| w.iter().map(|x| x.to_f32()).collect());
        Tensor::from_vec(v, shape, device)
    }
}

#[inline]
fn widen_into<W: WeightElem>(w: &[W], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(w) {
        *o = v.to_f32();
    }
}

/// `y = x Wᵀ + b` with packed weights (`W: [d_out, d_in]`).
#[derive(Debug, Clone)]
pub struct PackedLinear {
    pub w: WeightBuf,
    pub b: Option<Box<[f32]>>,
    pub d_in: usize,
    pub d_out: usize,
}

impl PackedLinear {
    pub fn new(w: &Tensor, b: Option<&Tensor>, dtype: DType) -> Result<Self> {
        let (d_out, d_in) = w.dims2()?;
        let b = match b {
            Some(b) => Some(b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1::<f32>()?.into_boxed_slice()),
            None => None,
        };
        if let Some(b) = &b {
            if b.len() != d_out {
                bail!("bias has {} entries, expected {d_out}", b.len())
            }
        }
        Ok(Self { w: WeightBuf::from_tensor(w, dtype)?, b, d_in, d_out })
    }

    pub fn bytes(&self) -> usize {
        self.w.bytes() + self.b.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// `out[r, :] = W x[r, :] + b` for every row of `x` (`rows = x.len() / d_in`).
    #[inline]
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        self.run(x, out, false)
    }

    /// `out[r, :] += W x[r, :] + b`.
    #[inline]
    pub fn forward_acc(&self, x: &[f32], out: &mut [f32]) {
        self.run(x, out, true)
    }

    fn run(&self, x: &[f32], out: &mut [f32], accumulate: bool) {
        let rows = x.len() / self.d_in;
        debug_assert_eq!(rows * self.d_in, x.len(), "input is not a whole number of rows");
        let out = &mut out[..rows * self.d_out];
        with_weights!(&self.w, |w| linear_impl(w, self.b.as_deref(), x, out, self.d_in, self.d_out, accumulate))
    }
}

fn linear_impl<W: WeightElem>(
    w: &[W],
    b: Option<&[f32]>,
    x: &[f32],
    out: &mut [f32],
    d_in: usize,
    d_out: usize,
    accumulate: bool,
) {
    // Work unit: a block of `bs` consecutive outputs of one row (bs divides d_out), computed
    // four outputs at a time so every load of `x` feeds four FMAs.
    let bs = [64, 32, 16, 8, 4].into_iter().find(|b| d_out.is_multiple_of(*b)).unwrap_or(d_out);
    let blocks_per_row = d_out / bs;
    let block = |(bi, o): (usize, &mut [f32])| {
        let (r, c0) = (bi / blocks_per_row, (bi % blocks_per_row) * bs);
        let xr = &x[r * d_in..(r + 1) * d_in];
        let row = |j: usize| &w[j * d_in..(j + 1) * d_in];
        let n = o.len();
        let mut emit = |k: usize, v: f32| {
            let v = v + b.map_or(0.0, |b| b[c0 + k]);
            if accumulate {
                o[k] += v
            } else {
                o[k] = v
            }
        };
        let mut k = 0;
        while k + 4 <= n {
            let j = c0 + k;
            let d = W::dot4([row(j), row(j + 1), row(j + 2), row(j + 3)], xr);
            for (t, v) in d.into_iter().enumerate() {
                emit(k + t, v);
            }
            k += 4;
        }
        while k < n {
            emit(k, W::dot(row(c0 + k), xr));
            k += 1;
        }
    };
    if parallel_for(out.len() * d_in) {
        let min_len = (PAR_TASK_WORK / (bs * d_in).max(1)).max(1);
        out.par_chunks_mut(bs).enumerate().with_min_len(min_len).for_each(block);
    } else {
        out.chunks_mut(bs).enumerate().for_each(block);
    }
}

/// Two-layer perceptron `l2(GELU(l1(x)))`.
#[derive(Debug, Clone)]
pub struct PackedMlp {
    pub l1: PackedLinear,
    pub l2: PackedLinear,
}

impl PackedMlp {
    /// `hidden` must hold `rows · l1.d_out` floats.
    pub fn forward(&self, x: &[f32], hidden: &mut [f32], out: &mut [f32]) {
        let rows = x.len() / self.l1.d_in;
        let hidden = &mut hidden[..rows * self.l1.d_out];
        self.l1.forward(x, hidden);
        gelu_inplace(hidden);
        self.l2.forward(hidden, out);
    }

    /// `out += l2(GELU(l1(x)))` (residual form).
    pub fn forward_acc(&self, x: &[f32], hidden: &mut [f32], out: &mut [f32]) {
        let rows = x.len() / self.l1.d_in;
        let hidden = &mut hidden[..rows * self.l1.d_out];
        self.l1.forward(x, hidden);
        gelu_inplace(hidden);
        self.l2.forward_acc(hidden, out);
    }

    pub fn bytes(&self) -> usize {
        self.l1.bytes() + self.l2.bytes()
    }
}

// ---------------------------------------------------------------------------------------
// Element-wise and row-wise kernels.
// ---------------------------------------------------------------------------------------

/// Branch-free `tanh` for f32: odd rational minimax approximation `x·P(x²)/Q(x²)` on
/// `[-7.99, 7.99]` (saturated outside), max abs error ≈ 1e-7. No libm call, so loops over
/// it auto-vectorise; libm `tanhf` dominated MPPI rollouts before.
#[inline]
#[allow(clippy::excessive_precision)]
pub fn fast_tanh(x: f32) -> f32 {
    const CLAMP: f32 = 7.998_811_7;
    let x = x.clamp(-CLAMP, CLAMP);
    let x2 = x * x;
    let mut p = x2 * -2.760_768_5e-16 + 2.000_188e-13;
    p = x2 * p + -8.604_672e-11;
    p = x2 * p + 5.122_297e-8;
    p = x2 * p + 1.485_722_4e-5;
    p = x2 * p + 6.372_619_3e-4;
    p = x2 * p + 4.893_524_6e-3;
    let mut q = x2 * 1.198_258_4e-6 + 1.185_347_1e-4;
    q = x2 * q + 2.268_434_6e-3;
    q = x2 * q + 4.893_525e-3;
    x * p / q
}

/// tanh-approximated GELU (same formula as candle's `Tensor::gelu`).
#[inline]
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/π)
    0.5 * x * (1.0 + fast_tanh(C * x * (1.0 + 0.044715 * x * x)))
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn gelu_inplace(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = gelu(*v));
}

pub fn silu_into(x: &[f32], out: &mut [f32]) {
    out.iter_mut().zip(x).for_each(|(o, &v)| *o = silu(v));
}

pub fn tanh_inplace(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = fast_tanh(*v));
}

/// `y += α x`.
#[inline]
pub fn axpy(y: &mut [f32], alpha: f32, x: &[f32]) {
    y.iter_mut().zip(x).for_each(|(y, &x)| *y += alpha * x);
}

/// `out = a x1 + b x2`.
#[inline]
pub fn lincomb_into(out: &mut [f32], a: f32, x1: &[f32], b: f32, x2: &[f32]) {
    for ((o, &u), &v) in out.iter_mut().zip(x1).zip(x2) {
        *o = a * u + b * v;
    }
}

/// `out += a x1 + b x2`.
#[inline]
pub fn acc_lincomb(out: &mut [f32], a: f32, x1: &[f32], b: f32, x2: &[f32]) {
    for ((o, &u), &v) in out.iter_mut().zip(x1).zip(x2) {
        *o += a * u + b * v;
    }
}

pub fn add_inplace(y: &mut [f32], x: &[f32]) {
    y.iter_mut().zip(x).for_each(|(y, &x)| *y += x);
}

/// `y[r, :] += row` for each row of width `row.len()`.
pub fn add_row_bcast(y: &mut [f32], row: &[f32]) {
    for chunk in y.chunks_exact_mut(row.len()) {
        add_inplace(chunk, row);
    }
}

/// `y[r, j] += gate[j] · x[r, j]`.
pub fn add_gated_rows(y: &mut [f32], gate: &[f32], x: &[f32]) {
    let d = gate.len();
    for (yr, xr) in y.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
        for ((y, &x), &g) in yr.iter_mut().zip(xr).zip(gate) {
            *y += g * x;
        }
    }
}

/// Parameter-free LayerNorm over rows of width `dim` (in place).
pub fn layer_norm_rows(x: &mut [f32], dim: usize) {
    for row in x.chunks_exact_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        row.iter_mut().for_each(|v| *v = (*v - mean) * inv);
    }
}

/// `out = LayerNorm(x)` row-wise.
pub fn layer_norm_rows_into(x: &[f32], out: &mut [f32], dim: usize) {
    out[..x.len()].copy_from_slice(x);
    layer_norm_rows(&mut out[..x.len()], dim);
}

/// adaLN modulation: `x[r, :] = x[r, :] ⊙ (1 + scale) + shift`.
pub fn modulate_rows(x: &mut [f32], shift: &[f32], scale: &[f32]) {
    let d = shift.len();
    for row in x.chunks_exact_mut(d) {
        for ((v, &sh), &sc) in row.iter_mut().zip(shift).zip(scale) {
            *v = *v * (1.0 + sc) + sh;
        }
    }
}

/// `x ← x / sqrt(‖x‖² + ε)`.
pub fn l2_normalize(x: &mut [f32]) {
    let inv = 1.0 / (simd::dot_f32(x, x) + L2_EPS).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}

/// Numerically stable softmax in place.
pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    x.iter_mut().for_each(|v| *v *= inv);
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// Mean over rows of width `dim`.
pub fn mean_rows_into(x: &[f32], dim: usize, out: &mut [f32]) {
    out[..dim].fill(0.0);
    let rows = x.len() / dim;
    for row in x.chunks_exact(dim) {
        add_inplace(&mut out[..dim], row);
    }
    let inv = 1.0 / rows as f32;
    out[..dim].iter_mut().for_each(|v| *v *= inv);
}

/// DiT-style sinusoidal embedding of `t ∈ [0, 1]` (scaled by 1000): `[cos(t·f), sin(t·f)]`.
pub fn sinusoidal_embedding(t: f32, out: &mut [f32]) {
    let half = out.len() / 2;
    let tt = t * 1000.0;
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        out[i] = (tt * f).cos();
        out[half + i] = (tt * f).sin();
    }
}

/// Frequencies of [`sinusoidal_embedding`] (for the graph path).
pub fn sinusoidal_freqs(dim: usize) -> Vec<f32> {
    let half = dim / 2;
    (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect()
}

/// Multi-head scaled dot-product attention.
///
/// `q: [lq, dim]`, `k, v: [lk, dim]`, `out: [lq, dim]`, `scratch ≥ lq·heads·lk`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lq: usize,
    lk: usize,
    dim: usize,
    heads: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let dh = dim / heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let task = |(idx, (o, s)): (usize, (&mut [f32], &mut [f32]))| {
        let (i, h) = (idx / heads, idx % heads);
        let qi = &q[i * dim + h * dh..i * dim + (h + 1) * dh];
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = simd::dot_f32(qi, &k[j * dim + h * dh..j * dim + (h + 1) * dh]) * scale;
        }
        softmax_inplace(s);
        o.fill(0.0);
        for (j, &p) in s.iter().enumerate() {
            axpy(o, p, &v[j * dim + h * dh..j * dim + (h + 1) * dh]);
        }
    };
    let out = &mut out[..lq * dim];
    let scratch = &mut scratch[..lq * heads * lk];
    if parallel_for(lq * lk * dim * 2) {
        out.par_chunks_mut(dh).zip(scratch.par_chunks_mut(lk)).enumerate().for_each(task);
    } else {
        out.chunks_mut(dh).zip(scratch.chunks_mut(lk)).enumerate().for_each(task);
    }
}

/// `out = W x` for a row-major f32 matrix `W: [rows, x.len()]`.
pub fn matvec_f32(w: &[f32], x: &[f32], out: &mut [f32]) {
    let d = x.len();
    let row = |(i, o): (usize, &mut f32)| *o = simd::dot_f32(&w[i * d..(i + 1) * d], x);
    if parallel_for(w.len()) {
        out.par_iter_mut().enumerate().with_min_len((PAR_TASK_WORK / d.max(1)).max(1)).for_each(row);
    } else {
        out.iter_mut().enumerate().for_each(row);
    }
}

/// Fused TTT online-gradient step on a row-major `W: [d, d]`:
/// `W ← W − η (W k − v) ⊗ k`.
///
/// Each row only needs its own residual `e_i = W_i·k − v_i`, so rows update independently
/// (no scratch buffer, embarrassingly parallel).
pub fn rank1_update(w: &mut [f32], k: &[f32], v: &[f32], eta: f32) {
    let d = k.len();
    let row = |(i, wi): (usize, &mut [f32])| {
        let e = simd::dot_f32(wi, k) - v[i];
        axpy(wi, -eta * e, k);
    };
    if parallel_for(2 * w.len()) {
        w.par_chunks_mut(d).enumerate().with_min_len((PAR_TASK_WORK / d.max(1)).max(1)).for_each(row);
    } else {
        w.chunks_mut(d).enumerate().for_each(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_candle() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::arange(0f32, 12.0, &dev)?.reshape((3, 4))?.affine(0.1, -0.5)?;
        let b = Tensor::new(&[0.1f32, -0.2, 0.3], &dev)?;
        let x = Tensor::arange(0f32, 8.0, &dev)?.reshape((2, 4))?.affine(0.25, -1.0)?;
        let want = x.matmul(&w.t()?)?.broadcast_add(&b)?.flatten_all()?.to_vec1::<f32>()?;
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let lin = PackedLinear::new(&w, Some(&b), dt)?;
            let mut out = vec![0f32; 6];
            lin.forward(&x.flatten_all()?.to_vec1::<f32>()?, &mut out);
            for (a, b) in out.iter().zip(&want) {
                assert!((a - b).abs() < 2e-2, "{dt:?}: {a} vs {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn fast_tanh_is_accurate() {
        let mut worst = 0f32;
        for i in -200_000..=200_000 {
            let x = i as f32 * 1e-4; // [-20, 20]
            worst = worst.max((fast_tanh(x) - x.tanh()).abs());
        }
        assert!(worst < 5e-7, "max |fast_tanh − tanh| = {worst}");
        assert_eq!(fast_tanh(0.0), 0.0);
        assert!(fast_tanh(1e4) <= 1.0 && fast_tanh(-1e4) >= -1.0);
    }

    #[test]
    fn rank1_update_is_gradient_step() {
        // One step on ½‖Wk − v‖² with ‖k‖ = 1 and η = 1 must map k exactly onto v.
        let d = 4;
        let mut w: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut k = vec![0.3, -0.1, 0.7, 0.2];
        l2_normalize(&mut k);
        let v = vec![1.0, -2.0, 0.5, 0.0];
        rank1_update(&mut w, &k, &v, 1.0);
        let mut out = vec![0f32; d];
        matvec_f32(&w, &k, &mut out);
        for (a, b) in out.iter().zip(&v) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn attention_uniform_keys_average_values() {
        let (lq, lk, dim, heads) = (2, 3, 4, 2);
        let q = vec![0.0; lq * dim];
        let k = vec![1.0; lk * dim];
        let v: Vec<f32> = (0..lk * dim).map(|i| i as f32).collect();
        let mut out = vec![0.0; lq * dim];
        let mut scratch = vec![0.0; lq * heads * lk];
        attention(&q, &k, &v, lq, lk, dim, heads, &mut out, &mut scratch);
        for i in 0..lq {
            for j in 0..dim {
                let mean = (0..lk).map(|r| v[r * dim + j]).sum::<f32>() / lk as f32;
                assert!((out[i * dim + j] - mean).abs() < 1e-5);
            }
        }
    }
}
