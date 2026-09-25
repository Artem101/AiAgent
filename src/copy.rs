//! Copy mechanism of the thought probe: a latent thought can *point* at tokens of the context
//! instead of spelling them out of the vocabulary.
//!
//! The TTT encoder stores a key `k_t = W_k [x̃_t ; z_t]` for every prompt position `t` (its
//! causal window and layer output, see [`crate::ttt`]). For output slot `l` the probe
//! embedding `e_l = (Θ s)_l` gives a query, a shift gate `σ_l` and a copy gate `λ_l`:
//!
//! ```text
//!   fresh_l(t) = softmax_t(q_l · k_t / √d_k)                         point anywhere
//!   p_0 = fresh_0,   p_l(t) = σ_l · p_{l−1}(t − 1) + (1 − σ_l) · fresh_l(t)   …or at the next token
//!   P_l(v) = λ_l · Σ_{t : x_t = v} p_l(t) + (1 − λ_l) · softmax(head(e_l))_v
//! ```
//!
//! (Pointer mass shifted past the last position is dropped, so `P_l` may sum to slightly less
//! than one.) The shift term copies a span token by token: a thought needs to know *where* a number or a
//! name is («the price in the row of лампа», «the expression in the question»), not every one
//! of its digits. All slots are produced in parallel (the recursion runs over pointer
//! distributions, not over sampled tokens), so decoding stays non-autoregressive, and training
//! maximises the exact marginal likelihood `log P_l(y_l)`.
//!
//! Two details make the pointer learnable. Special tokens (padding, role and verb tokens: ids
//! below `min_token`) are never copied — otherwise the gate learns to "copy" the padding that
//! fills most of a prompt and nothing else. And the pointer gets its own loss,
//! `−log Σ_{t : x_t = y_l} p_l(t)` on every slot whose target occurs in the context: in the
//! mixture alone its gradient is scaled by `λ_l`, and a gate that starts out preferring the
//! vocabulary would never let the pointer learn.
//!
//! [`CopyHead`] is the graph path (training), [`PackedCopyHead`] the zero-allocation kernel
//! path (inference); `tests` checks that they agree.

use candle_core::{DType, Result, Tensor, D};

use crate::kernels::{sigmoid, PackedLinear};
use crate::nn::{Lin, ParamStore};

/// Floor inside `log(pointer mass)`.
const PTR_EPS: f64 = 1e-12;
/// Floor inside the pointer's own loss (bounds its gradient while attention is still flat).
const AUX_EPS: f64 = 1e-6;
/// Weight of the pointer's own loss relative to the mixture likelihood.
pub const POINTER_WEIGHT: f64 = 1.0;

/// Query and gates of the copy mechanism (graph path).
#[derive(Debug, Clone)]
pub struct CopyHead {
    /// `e_l → q_l` (`d_token → d_key`).
    pub query: Lin,
    /// `e_l → (shift, copy)` gate logits.
    pub gate: Lin,
    /// Token ids below this are never copied (special tokens).
    pub min_token: u32,
}

/// `log σ(x)`, numerically stable.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    // −softplus(−x) = −(relu(−x) + log(1 + exp(−|x|)))
    let soft = (x.neg()?.relu()? + ((x.abs()?.neg()?.exp()? + 1.0)?.log()?))?;
    soft.neg()
}

impl CopyHead {
    pub fn new(ps: &mut ParamStore, d_token: usize, d_key: usize, min_token: u32) -> Result<Self> {
        Ok(Self {
            query: ps.linear("jepa.copy_query", d_token, d_key, false)?,
            gate: ps.linear("jepa.copy_gate", d_token, 2, true)?,
            min_token,
        })
    }

    pub fn d_key(&self) -> usize {
        self.query.d_out()
    }

    /// Pointer distributions `p_l`, `[M, L, N]`, and the copy-gate logits `[M, L]`, for probe
    /// embeddings `emb: [M, L, d_token]` over `keys: [M, N, d_key]`.
    pub fn pointer(&self, emb: &Tensor, keys: &Tensor) -> Result<(Tensor, Tensor)> {
        let (m, l, _) = emb.dims3()?;
        let n = keys.dims()[1];
        let q = self.query.forward(emb)?;
        let scores = (q.matmul(&keys.transpose(1, 2)?.contiguous()?)? / (self.d_key() as f64).sqrt())?;
        let fresh = candle_nn::ops::softmax(&scores, D::Minus1)?; // not `softmax_last_dim`: no backward
        let gates = self.gate.forward(emb)?; // [M, L, 2]
        let shift = candle_nn::ops::sigmoid(&gates.narrow(2, 0, 1)?)?; // [M, L, 1]
        let mut p = Vec::with_capacity(l);
        p.push(fresh.narrow(1, 0, 1)?.squeeze(1)?);
        let zero = Tensor::zeros((m, 1), fresh.dtype(), fresh.device())?;
        for i in 1..l {
            let prev: &Tensor = &p[i - 1];
            let shifted = if n > 1 { Tensor::cat(&[&zero, &prev.narrow(1, 0, n - 1)?], 1)? } else { zero.clone() };
            let s = shift.narrow(1, i, 1)?.squeeze(1)?; // [M, 1]
            let f = fresh.narrow(1, i, 1)?.squeeze(1)?;
            p.push((shifted.broadcast_mul(&s)? + f.broadcast_mul(&(s.neg()? + 1.0)?)?)?);
        }
        Ok((Tensor::stack(&p, 1)?, gates.narrow(2, 1, 1)?.squeeze(2)?))
    }

    /// `log P_l(y_l)` of the mixture, `[M, L]`, and the pointer's own loss (a scalar).
    ///
    /// `emb: [M, L, d_token]`, `logits: [M, L, V]` (vocabulary head, f32),
    /// `keys: [M, N, d_key]`, `targets: [M, L]` (u32), `context: [M, N]` (u32 token ids).
    pub fn losses(
        &self,
        emb: &Tensor,
        logits: &Tensor,
        keys: &Tensor,
        targets: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (m, l, _) = emb.dims3()?;
        let n = context.dims()[1];
        let (p, copy_logit) = self.pointer(emb, keys)?;
        let (p, copy_logit) = (p.to_dtype(DType::F32)?, copy_logit.to_dtype(DType::F32)?);
        // pointer mass on (copyable) positions holding the target token
        let copyable = context.ge(self.min_token)?.to_dtype(DType::F32)?.unsqueeze(1)?; // [M, 1, N]
        let hits = context
            .unsqueeze(1)?
            .broadcast_as((m, l, n))?
            .eq(&targets.unsqueeze(2)?.broadcast_as((m, l, n))?)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&copyable)?;
        let ptr = (p * &hits)?.sum(D::Minus1)?; // [M, L]
        let vocab =
            candle_nn::ops::log_softmax(logits, D::Minus1)?.gather(&targets.unsqueeze(2)?, D::Minus1)?.squeeze(2)?;
        let a = (log_sigmoid(&copy_logit)? + (&ptr + PTR_EPS)?.log()?)?;
        let b = (log_sigmoid(&copy_logit.neg()?)? + vocab)?;
        let mx = a.maximum(&b)?.detach();
        let loglik = (((a - &mx)?.exp()? + (b - &mx)?.exp()?)?.log()? + mx)?;
        // the pointer's own loss on the slots it could have copied
        let copyable_slot = hits.max(D::Minus1)?; // [M, L], 1 where the target occurs
        let count = copyable_slot.sum_all()?.to_scalar::<f32>()?.max(1.0) as f64;
        let aux = ((((ptr + AUX_EPS)?.log()?.neg()? * copyable_slot)?.sum_all()?) / count)?;
        Ok((loglik, aux))
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedCopyHead> {
        Ok(PackedCopyHead { query: self.query.pack(dtype)?, gate: self.gate.pack(dtype)?, min_token: self.min_token })
    }
}

/// Query and gates of the copy mechanism (kernel path).
#[derive(Debug, Clone)]
pub struct PackedCopyHead {
    pub query: PackedLinear,
    pub gate: PackedLinear,
    /// Token ids below this are never copied.
    pub min_token: u32,
}

/// Scratch of [`PackedCopyHead::mix`] (pre-allocated; `capacity` = longest context).
#[derive(Debug)]
pub struct CopyScratch {
    q: Box<[f32]>,
    fresh: Box<[f32]>,
    prev: Box<[f32]>,
}

impl CopyScratch {
    pub fn new(arena: &mut crate::arena::Arena, d_key: usize, capacity: usize) -> Self {
        Self { q: arena.host(d_key), fresh: arena.host(capacity), prev: arena.host(capacity) }
    }
}

impl PackedCopyHead {
    pub fn d_key(&self) -> usize {
        self.query.d_out
    }

    pub fn bytes(&self) -> usize {
        self.query.bytes() + self.gate.bytes()
    }

    /// Turns vocabulary logits into the copy mixture, in place: `rows` holds `L` rows of `V`
    /// logits on entry and `P_l(v)` on exit. `emb`: `L × d_token` probe embeddings; `keys`:
    /// `n × d_key` keys of the context tokens `tokens` (`n ≤` scratch capacity). No allocation.
    pub fn mix(&self, emb: &[f32], rows: &mut [f32], keys: &[f32], tokens: &[u32], s: &mut CopyScratch) {
        let (dt, dk, n) = (self.query.d_in, self.d_key(), tokens.len());
        let v = rows.len() / (emb.len() / dt);
        let scale = 1.0 / (dk as f32).sqrt();
        for (l, (e, row)) in emb.chunks_exact(dt).zip(rows.chunks_exact_mut(v)).enumerate() {
            let mut g = [0f32; 2];
            self.gate.forward(e, &mut g);
            let (shift, lambda) = (sigmoid(g[0]), sigmoid(g[1]));
            self.query.forward(e, &mut s.q);
            let fresh = &mut s.fresh[..n];
            for (f, k) in fresh.iter_mut().zip(keys.chunks_exact(dk)) {
                *f = s.q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax(fresh);
            if l > 0 {
                for t in (0..n).rev() {
                    let before = if t > 0 { s.prev[t - 1] } else { 0.0 };
                    fresh[t] = shift * before + (1.0 - shift) * fresh[t];
                }
            }
            s.prev[..n].copy_from_slice(fresh);
            softmax(row);
            row.iter_mut().for_each(|x| *x *= 1.0 - lambda);
            for (&tok, &p) in tokens.iter().zip(fresh.iter()) {
                if tok >= self.min_token {
                    if let Some(x) = row.get_mut(tok as usize) {
                        *x += lambda * p;
                    }
                }
            }
        }
    }
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    x.iter_mut().for_each(|v| *v *= inv);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::kernels::rng::Rng;
    use candle_core::Device;

    #[test]
    fn packed_mixture_matches_graph_likelihood() -> Result<()> {
        let dev = Device::Cpu;
        let (l, dt, dk, n, v) = (5, 6, 4, 8, 11);
        let mut ps = ParamStore::new(&dev, 5);
        let head = CopyHead::new(&mut ps, dt, dk, 2)?; // tokens 0 and 1 are "special"
        let mut rng = Rng::new(3);
        let mut rand = |len: usize| {
            let mut x = vec![0f32; len];
            rng.fill_normal(&mut x, 1.0);
            x
        };
        let (emb, logits, keys) = (rand(l * dt), rand(l * v), rand(n * dk));
        let context: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 0];
        // packed: full distributions per slot
        let packed = head.pack(DType::F32)?;
        let mut scratch = CopyScratch::new(&mut Arena::new(&dev), dk, 16);
        let mut rows = logits.clone();
        packed.mix(&emb, &mut rows, &keys, &context, &mut scratch);
        for row in rows.chunks_exact(v) {
            // pointer mass shifted past the last position is dropped
            let total = row.iter().sum::<f32>();
            assert!(total > 0.0 && total <= 1.0 + 1e-5, "{total}");
        }
        // graph: log P of every token at every slot
        let t = |x: &[f32], s: &[usize]| Tensor::from_slice(x, s, &dev);
        let (e, lg, k) = (t(&emb, &[1, l, dt])?, t(&logits, &[1, l, v])?, t(&keys, &[1, n, dk])?);
        let ctx = Tensor::from_slice(&context, (1, n), &dev)?;
        for tok in 0..v as u32 {
            let targets = Tensor::from_slice(&vec![tok; l], (1, l), &dev)?;
            let ll = head.losses(&e, &lg, &k, &targets, &ctx)?.0.squeeze(0)?.to_vec1::<f32>()?;
            for (slot, want) in ll.iter().enumerate() {
                let got = rows[slot * v + tok as usize].ln();
                assert!((got - want).abs() < 1e-4, "slot {slot} token {tok}: packed {got} vs graph {want}");
            }
        }
        Ok(())
    }
}
