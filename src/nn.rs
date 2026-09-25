//! Graph-path building blocks (candle, autograd) and their packing into host kernels.
//!
//! Parameters live in f32 [`Var`]s (master weights; AdamW moments are f32 too). Every
//! module casts its weights to the dtype of its input at use, so running the graph in
//! bf16/f16 gives classic mixed precision: low-precision compute, f32 accumulation of
//! gradients into the master copy.

use std::collections::HashMap;

use candle_core::{bail, DType, Device, Result, Tensor, Var, D};
use candle_nn::VarMap;

use crate::kernels::{rng::Rng, sinusoidal_freqs, PackedLinear, PackedMlp, L2_EPS, LN_EPS};

/// Parameter initialisation schemes (all driven by the deterministic [`Rng`]).
#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// `U(−b, b)`.
    Uniform(f32),
    /// `N(0, σ²)`.
    Normal(f32),
    Zeros,
}

/// Named, deterministically initialised parameters backed by a candle [`VarMap`].
pub struct ParamStore {
    varmap: VarMap,
    device: Device,
    rng: Rng,
}

impl ParamStore {
    pub fn new(device: &Device, seed: u64) -> Self {
        Self { varmap: VarMap::new(), device: device.clone(), rng: Rng::new(seed) }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tensor(&mut self, name: &str, shape: &[usize], init: Init) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let mut data = vec![0f32; n];
        match init {
            Init::Uniform(b) => self.rng.fill_uniform(&mut data, -b, b),
            Init::Normal(s) => self.rng.fill_normal(&mut data, s),
            Init::Zeros => {}
        }
        let var = Var::from_vec(data, shape, &self.device)?;
        let mut map = self.varmap.data().lock().expect("varmap poisoned");
        if map.contains_key(name) {
            bail!("duplicate parameter name '{name}'")
        }
        map.insert(name.to_string(), var.clone());
        Ok(var.as_tensor().clone())
    }

    /// PyTorch-style default init `U(±1/√d_in)`.
    pub fn linear(&mut self, name: &str, d_in: usize, d_out: usize, bias: bool) -> Result<Lin> {
        let bound = 1.0 / (d_in as f32).sqrt();
        let w = self.tensor(&format!("{name}.weight"), &[d_out, d_in], Init::Uniform(bound))?;
        let b = if bias { Some(self.tensor(&format!("{name}.bias"), &[d_out], Init::Uniform(bound))?) } else { None };
        Ok(Lin { w, b })
    }

    /// Zero-initialised linear layer (adaLN-Zero modulations, output projections).
    pub fn linear_zero(&mut self, name: &str, d_in: usize, d_out: usize) -> Result<Lin> {
        let w = self.tensor(&format!("{name}.weight"), &[d_out, d_in], Init::Zeros)?;
        let b = Some(self.tensor(&format!("{name}.bias"), &[d_out], Init::Zeros)?);
        Ok(Lin { w, b })
    }

    pub fn mlp(&mut self, name: &str, d_in: usize, d_hidden: usize, d_out: usize) -> Result<Mlp> {
        Ok(Mlp {
            l1: self.linear(&format!("{name}.fc1"), d_in, d_hidden, true)?,
            l2: self.linear(&format!("{name}.fc2"), d_hidden, d_out, true)?,
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    pub fn var(&self, name: &str) -> Option<Var> {
        self.varmap.data().lock().expect("varmap poisoned").get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.varmap.data().lock().expect("varmap poisoned").keys().cloned().collect();
        v.sort();
        v
    }

    pub fn num_params(&self) -> usize {
        self.vars().iter().map(|v| v.elem_count()).sum()
    }

    /// Snapshot of all tensors under `prefix.name`.
    pub fn export(&self, prefix: &str, out: &mut HashMap<String, Tensor>) {
        for (k, v) in self.varmap.data().lock().expect("varmap poisoned").iter() {
            out.insert(format!("{prefix}.{k}"), v.as_tensor().clone());
        }
    }

    /// Overwrites every parameter from `src[prefix.name]`.
    pub fn import(&self, prefix: &str, src: &HashMap<String, Tensor>) -> Result<()> {
        for (k, v) in self.varmap.data().lock().expect("varmap poisoned").iter() {
            let key = format!("{prefix}.{k}");
            let Some(t) = src.get(&key) else { bail!("checkpoint is missing '{key}'") };
            v.set(&t.to_device(&self.device)?.to_dtype(DType::F32)?)?;
        }
        Ok(())
    }
}

/// Linear layer `y = x Wᵀ + b`, `W: [d_out, d_in]`.
#[derive(Debug, Clone)]
pub struct Lin {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Lin {
    pub fn d_in(&self) -> usize {
        self.w.dims()[1]
    }
    pub fn d_out(&self) -> usize {
        self.w.dims()[0]
    }

    /// Applies to the last dimension of `x` (any rank ≥ 1).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let Some((&d_in, lead)) = dims.split_last() else { bail!("Lin: scalar input") };
        if d_in != self.d_in() {
            bail!("Lin: input width {d_in}, expected {}", self.d_in())
        }
        let w = self.w.to_dtype(x.dtype())?;
        let mut y = x.reshape(((), d_in))?.matmul(&w.t()?)?;
        if let Some(b) = &self.b {
            y = y.broadcast_add(&b.to_dtype(x.dtype())?)?;
        }
        let mut shape = lead.to_vec();
        shape.push(self.d_out());
        y.reshape(shape)
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedLinear> {
        PackedLinear::new(&self.w, self.b.as_ref(), dtype)
    }

    /// Rebuilds a graph layer (f32) from packed weights.
    pub fn from_packed(p: &PackedLinear, device: &Device) -> Result<Self> {
        let w = p.w.to_tensor(&[p.d_out, p.d_in], device)?;
        let b = match &p.b {
            Some(b) => Some(Tensor::from_slice(b, p.d_out, device)?),
            None => None,
        };
        Ok(Self { w, b })
    }
}

/// `l2(GELU(l1(x)))`.
#[derive(Debug, Clone)]
pub struct Mlp {
    pub l1: Lin,
    pub l2: Lin,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.gelu()?)
    }
    pub fn pack(&self, dtype: DType) -> Result<PackedMlp> {
        Ok(PackedMlp { l1: self.l1.pack(dtype)?, l2: self.l2.pack(dtype)? })
    }
    pub fn from_packed(p: &PackedMlp, device: &Device) -> Result<Self> {
        Ok(Self { l1: Lin::from_packed(&p.l1, device)?, l2: Lin::from_packed(&p.l2, device)? })
    }
}

/// Parameter-free LayerNorm over the last dim; statistics in f32.
pub fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + LN_EPS as f64)?.sqrt()?)?.to_dtype(dt)
}

/// `x / sqrt(‖x‖² + ε)` over the last dim.
pub fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(D::Minus1)? + L2_EPS as f64)?.sqrt()?;
    x.broadcast_div(&n)
}

/// adaLN: `x ⊙ (1 + scale) + shift`, `x: [B, L, d]`, `shift, scale: [B, d]`.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?.broadcast_add(&shift.unsqueeze(1)?)
}

/// Multi-head attention, `q: [B, Lq, d]`, `k, v: [B, Lk, d]` → `[B, Lq, d]`.
pub fn mha(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, lq, d) = q.dims3()?;
    let lk = k.dims()[1];
    let dh = d / heads;
    let split = |t: &Tensor, l: usize| t.reshape((b, l, heads, dh))?.transpose(1, 2)?.contiguous();
    let (q, k, v) = (split(q, lq)?, split(k, lk)?, split(v, lk)?);
    let scores = (q.matmul(&k.t()?.contiguous()?)? * (1.0 / (dh as f64).sqrt()))?;
    let dt = scores.dtype();
    // `softmax_last_dim` is a fused kernel without a backward pass: it would silently cut the
    // gradient to the queries and keys.
    let p = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, candle_core::D::Minus1)?.to_dtype(dt)?;
    p.matmul(&v)?.transpose(1, 2)?.reshape((b, lq, d))
}

/// Sinusoidal time embedding `[B] → [B, dim]` (matches [`crate::kernels::sinusoidal_embedding`]).
pub fn sinusoidal(t: &Tensor, dim: usize) -> Result<Tensor> {
    let freqs = Tensor::from_vec(sinusoidal_freqs(dim), (1, dim / 2), t.device())?;
    let args = (t.to_dtype(DType::F32)?.unsqueeze(1)? * 1000.0)?.broadcast_mul(&freqs)?;
    Tensor::cat(&[args.cos()?, args.sin()?], 1)
}

/// Mean squared error (computed in f32).
pub fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.sqr()?.mean_all()
}
