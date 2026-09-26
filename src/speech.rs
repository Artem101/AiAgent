//! Speech decoder of the unified model ([`crate::unified`]): the latent plan and the context in,
//! text out, one token at a time.
//!
//! ```text
//!   memory  = LN([ W_p s_0 + e_0 ; … ; W_p s_H + e_H ;  W_c [x̃_1 ; z_1] ; … ; W_c [x̃_N ; z_N] ])
//!             └──────── the thoughts of the plan ─────┘  └── every context token (TTT features) ──┘
//!   h_0     = E[<bot>] + p_0,   h_l = E[y_{l−1}] + p_l
//!   block   = h += SelfAttn_causal(LN h);  h += CrossAttn(LN h, memory);  h += W_2 ReLU(W_1 LN h)
//!   P(y_l)  = λ_l · pointer_l + (1 − λ_l) · softmax(LN(h_l) Eᵀ + b)
//! ```
//!
//! The decoder is a small pre-LN transformer. Its self-attention is causal over what it has
//! said so far; its cross-attention reads the latent plan the planner produced (the model's
//! "thoughts", [`crate::jepa`]) and the TTT features of every context token (the page, the
//! conversation and the question), with padding masked out; the keys and values of this memory
//! are shared by all blocks. The output embedding is tied to
//! the input one. On top sits the copy mechanism of [`crate::copy`]: a pointer into the context
//! (keys from the TTT encoder) mixed with the vocabulary by a learned gate, so a price, a
//! product name or a word of the question is copied rather than spelled.
//!
//! Unlike the non-autoregressive decoders of [`crate::pipeline`] (CFM, the thought probe),
//! every token here is conditioned on the previous ones — what fluent text needs. Actions keep
//! their format (`CLICK [link] ␣лампа <end>`, `ANSWER <none> ␣Привет! <end>`): talking is just
//! the `ANSWER` action.

use candle_core::{DType, Device, Result, Tensor, D};

use crate::copy::{CopyHead, POINTER_WEIGHT};
use crate::kernels::rng::Rng;
use crate::nn::{self, Init, Lin, ParamStore};
use crate::text::{BOT, END, PAD};

/// Architecture of the speech decoder.
#[derive(Debug, Clone)]
pub struct SpeechConfig {
    /// Transformer width.
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub mlp_ratio: usize,
    /// Longest output `L` (teacher forcing window; generation stops at `<end>` or after `L`).
    pub max_len: usize,
    /// Key width of the pointer into the context (`0` = no copy mechanism).
    pub copy_dim: usize,
}

/// One transformer block (pre-LN, parameter-free LayerNorm, bias-free projections, ReLU MLP —
/// the cheapest choices for training on a CPU, where candle runs element-wise ops on one core).
#[derive(Debug, Clone)]
struct Block {
    self_qkv: Lin,
    self_o: Lin,
    cross_q: Lin,
    cross_o: Lin,
    mlp_in: Lin,
    mlp_out: Lin,
}

/// Keys and values of the memory, split into heads and shared by all blocks (computed once
/// per utterance).
pub struct Memory {
    k: Tensor,
    v: Tensor,
    /// `[B, 1, 1, P + N]` additive mask (padding of the context).
    mask: Tensor,
    /// Pointer keys `[B, N, d_key]` and context tokens `[B, N]` (with a copy mechanism).
    keys: Option<Tensor>,
    context: Tensor,
}

/// How [`SpeechDecoder::generate`] picks tokens.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    /// `0` = greedy.
    pub temperature: f32,
    /// Sample among the `top_k` most likely tokens (`0` = all).
    pub top_k: usize,
    /// The first `greedy_prefix` tokens are always greedy (the verb and role of an action).
    pub greedy_prefix: usize,
}

impl Sampling {
    pub const GREEDY: Self = Self { temperature: 0.0, top_k: 0, greedy_prefix: 0 };
}

/// Per-example negative log-likelihood sums, token counts and the mean loss (see
/// [`SpeechDecoder::loss`]).
pub struct SpeechLoss {
    /// Mean NLL per output token plus the weighted pointer loss (differentiable).
    pub total: Tensor,
    pub nll: f32,
    pub pointer: f32,
    /// `Σ_l −log P(y_l)` per example, and the number of scored tokens.
    pub per_example: Vec<(f32, usize)>,
    /// Tokens whose most likely prediction was right, per example.
    pub correct: Vec<usize>,
}

/// The speech decoder (graph path; see the module docs).
#[derive(Debug, Clone)]
pub struct SpeechDecoder {
    /// `[V, d]`, shared by the input and the output.
    pub tok_emb: Tensor,
    /// `[L, d]`.
    pub pos_emb: Tensor,
    /// Plan states `d_s → d`, plus a learned embedding per plan step `[H + 1, d]`.
    pub plan_in: Lin,
    pub plan_pos: Tensor,
    /// Context features `[x̃ ; z] → d`.
    pub ctx_in: Lin,
    /// Keys and values of the memory (`d → 2d`), shared by every block's cross-attention.
    pub mem_kv: Lin,
    blocks: Vec<Block>,
    pub out_bias: Tensor,
    pub copy: Option<CopyHead>,
    heads: usize,
    max_len: usize,
}

impl SpeechDecoder {
    /// `d_state`: width of a plan state; `plan_len`: `H + 1`; `d_features`: width of the TTT
    /// features; ids below `min_token` are never copied.
    pub fn new(
        ps: &mut ParamStore,
        cfg: &SpeechConfig,
        vocab: usize,
        d_state: usize,
        plan_len: usize,
        d_features: usize,
        min_token: u32,
    ) -> Result<Self> {
        let d = cfg.d_model;
        if !d.is_multiple_of(cfg.n_heads) {
            candle_core::bail!("speech.d_model ({d}) must be divisible by n_heads ({})", cfg.n_heads)
        }
        let blocks = (0..cfg.n_layers)
            .map(|i| {
                let n = format!("speech.block{i}");
                Ok(Block {
                    self_qkv: ps.linear(&format!("{n}.self_qkv"), d, 3 * d, false)?,
                    self_o: ps.linear(&format!("{n}.self_o"), d, d, false)?,
                    cross_q: ps.linear(&format!("{n}.cross_q"), d, d, false)?,
                    cross_o: ps.linear(&format!("{n}.cross_o"), d, d, false)?,
                    mlp_in: ps.linear(&format!("{n}.mlp_in"), d, cfg.mlp_ratio * d, false)?,
                    mlp_out: ps.linear(&format!("{n}.mlp_out"), cfg.mlp_ratio * d, d, false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tok_emb: ps.tensor("speech.tok_emb", &[vocab, d], Init::Normal(0.02))?,
            pos_emb: ps.tensor("speech.pos_emb", &[cfg.max_len, d], Init::Normal(0.02))?,
            plan_in: ps.linear("speech.plan_in", d_state, d, true)?,
            plan_pos: ps.tensor("speech.plan_pos", &[plan_len, d], Init::Normal(0.02))?,
            ctx_in: ps.linear("speech.ctx_in", d_features, d, true)?,
            mem_kv: ps.linear("speech.mem_kv", d, 2 * d, false)?,
            blocks,
            out_bias: ps.tensor("speech.out_bias", &[vocab], Init::Zeros)?,
            copy: if cfg.copy_dim > 0 {
                let mut c = CopyHead::named(ps, "speech.copy", d, cfg.copy_dim, min_token)?;
                c.mask_specials = true;
                Some(c)
            } else {
                None
            },
            heads: cfg.n_heads,
            max_len: cfg.max_len,
        })
    }

    pub fn d_model(&self) -> usize {
        self.tok_emb.dims()[1]
    }

    pub fn vocab(&self) -> usize {
        self.tok_emb.dims()[0]
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Builds the memory from `plan: [B, H + 1, d_s]`, the TTT features `[B, N, d_feat]` of the
    /// context tokens `context: [B, N]` (u32) and their pointer keys `[B, N, d_key]`.
    pub fn memory(&self, plan: &Tensor, features: &Tensor, context: &Tensor, keys: Option<&Tensor>) -> Result<Memory> {
        let (b, p, _) = plan.dims3()?;
        let n = features.dims()[1];
        let thoughts = self.plan_in.forward(plan)?.broadcast_add(&self.plan_pos.unsqueeze(0)?)?;
        let ctx = self.ctx_in.forward(features)?;
        let mem = nn::layer_norm(&Tensor::cat(&[thoughts, ctx], 1)?)?;
        let d = self.d_model();
        let kv = self.mem_kv.forward(&mem)?;
        let (k, v) =
            (nn::split_heads(&kv.narrow(2, 0, d)?, self.heads)?, nn::split_heads(&kv.narrow(2, d, d)?, self.heads)?);
        // padding of the context is invisible; thoughts are always visible
        let pad = context.eq(PAD)?.to_dtype(DType::F32)?.affine(nn::MASKED as f64, 0.0)?;
        let mask =
            Tensor::cat(&[Tensor::zeros((b, p), DType::F32, plan.device())?, pad], 1)?.reshape((b, 1, 1, p + n))?;
        Ok(Memory { k, v, mask, keys: keys.cloned(), context: context.clone() })
    }

    /// Hidden states `[B, l, d]` (after the final LayerNorm) for the inputs `prev: [B, l]` (u32;
    /// `prev[:, 0] = <bot>`, then the tokens said so far).
    pub fn hidden(&self, prev: &Tensor, mem: &Memory) -> Result<Tensor> {
        let (b, l) = prev.dims2()?;
        let d = self.d_model();
        let mut x = self
            .tok_emb
            .index_select(&prev.flatten_all()?, 0)?
            .reshape((b, l, d))?
            .broadcast_add(&self.pos_emb.narrow(0, 0, l)?.unsqueeze(0)?)?;
        let causal = nn::causal_mask(l, prev.device())?;
        for blk in &self.blocks {
            let qkv = blk.self_qkv.forward(&nn::layer_norm(&x)?)?;
            let (q, ks, vs) = (qkv.narrow(2, 0, d)?, qkv.narrow(2, d, d)?, qkv.narrow(2, 2 * d, d)?);
            x = (&x + blk.self_o.forward(&nn::mha_masked(&q, &ks, &vs, self.heads, Some(&causal))?)?)?;
            let q = blk.cross_q.forward(&nn::layer_norm(&x)?)?;
            x = (&x + blk.cross_o.forward(&nn::attend(&q, &mem.k, &mem.v, Some(&mem.mask))?)?)?;
            x = (&x + blk.mlp_out.forward(&blk.mlp_in.forward(&nn::layer_norm(&x)?)?.relu()?)?)?;
        }
        nn::layer_norm(&x)
    }

    /// Vocabulary logits `[B, l, V]` (f32) of hidden states `[B, l, d]`.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, l, d) = hidden.dims3()?;
        let w = self.tok_emb.to_dtype(hidden.dtype())?;
        hidden
            .reshape((b * l, d))?
            .matmul(&w.t()?)?
            .broadcast_add(&self.out_bias.to_dtype(hidden.dtype())?)?
            .to_dtype(DType::F32)?
            .reshape((b, l, self.vocab()))
    }

    /// Decoder inputs for teacher forcing: `[<bot>, y_0, …, y_{L−2}]`.
    pub fn shift_right(targets: &Tensor) -> Result<Tensor> {
        let (b, l) = targets.dims2()?;
        let bos = Tensor::full(BOT, (b, 1), targets.device())?;
        Tensor::cat(&[&bos, &targets.narrow(1, 0, l - 1)?], 1)
    }

    /// Teacher-forced loss of `targets: [B, L]` (u32, `<pad>` after `<end>` is not scored): the
    /// mean negative log-likelihood per token of the copy mixture (or of the vocabulary alone),
    /// plus the pointer's own loss.
    pub fn loss(&self, targets: &Tensor, mem: &Memory) -> Result<SpeechLoss> {
        let (b, l) = targets.dims2()?;
        let hidden = self.hidden(&Self::shift_right(targets)?, mem)?;
        let logits = self.logits(&hidden)?;
        let (loglik, aux) = match (&self.copy, &mem.keys) {
            (Some(copy), Some(keys)) => {
                let (ll, aux) = copy.losses(&hidden, &logits, keys, targets, &mem.context)?;
                (ll, Some(aux))
            }
            _ => (
                candle_nn::ops::log_softmax(&logits, D::Minus1)?
                    .gather(&targets.unsqueeze(2)?, D::Minus1)?
                    .squeeze(2)?,
                None,
            ),
        };
        let mask = targets.ne(PAD)?.to_dtype(DType::F32)?;
        let count = mask.sum_all()?.to_scalar::<f32>()?.max(1.0);
        let nll_sum = (loglik.neg()? * &mask)?;
        let nll = (nll_sum.sum_all()? / count as f64)?;
        let total = match &aux {
            Some(a) => (&nll + (a * POINTER_WEIGHT)?)?,
            None => nll.clone(),
        };
        // per-example statistics (argmax of the vocabulary head; the pointer only adds mass to
        // context tokens, so this is a lower bound on the mixture's accuracy)
        let sums = nll_sum.sum(1)?.to_vec1::<f32>()?;
        let counts = mask.sum(1)?.to_vec1::<f32>()?;
        let hits = (logits.argmax(D::Minus1)?.eq(targets)?.to_dtype(DType::F32)? * &mask)?.sum(1)?.to_vec1::<f32>()?;
        let _ = (b, l);
        Ok(SpeechLoss {
            nll: nll.to_scalar::<f32>()?,
            pointer: aux.as_ref().map(|a| a.to_scalar::<f32>()).transpose()?.unwrap_or(0.0),
            per_example: sums.iter().zip(&counts).map(|(&s, &c)| (s, c as usize)).collect(),
            correct: hits.iter().map(|&h| h as usize).collect(),
            total,
        })
    }

    /// Output distributions of the last input position of every row: `[B][V]` probabilities
    /// (the copy mixture when the decoder has one).
    pub fn next_distribution(&self, prev: &Tensor, mem: &Memory) -> Result<Vec<Vec<f32>>> {
        let (b, l) = prev.dims2()?;
        let hidden = self.hidden(prev, mem)?;
        let last = hidden.narrow(1, l - 1, 1)?;
        let probs = candle_nn::ops::softmax(&self.logits(&last)?.squeeze(1)?, D::Minus1)?.to_vec2::<f32>()?;
        let (Some(copy), Some(keys)) = (&self.copy, &mem.keys) else { return Ok(probs) };
        let (p, gate) = copy.pointer_over(&hidden, keys, Some(&mem.context))?;
        let p = p.narrow(1, l - 1, 1)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let gate = gate.narrow(1, l - 1, 1)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let context = mem.context.to_vec2::<u32>()?;
        let mut out = probs;
        for r in 0..b {
            let lambda = 1.0 / (1.0 + (-gate[r]).exp());
            out[r].iter_mut().for_each(|x| *x *= 1.0 - lambda);
            for (&tok, &w) in context[r].iter().zip(&p[r]) {
                if tok >= copy.min_token {
                    out[r][tok as usize] += lambda * w;
                }
            }
        }
        Ok(out)
    }

    /// Hidden state `[B, 1, d]` of the input token `tokens: [B, 1]` at position `pos`, given the
    /// self-attention keys and values of the earlier positions in `cache` (which it extends).
    pub fn step_hidden(
        &self,
        tokens: &Tensor,
        pos: usize,
        mem: &Memory,
        cache: &mut Vec<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let b = tokens.dims()[0];
        let d = self.d_model();
        let mut x = self
            .tok_emb
            .index_select(&tokens.flatten_all()?, 0)?
            .reshape((b, 1, d))?
            .broadcast_add(&self.pos_emb.narrow(0, pos, 1)?.unsqueeze(0)?)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            let qkv = blk.self_qkv.forward(&nn::layer_norm(&x)?)?;
            let (q, k, v) = (qkv.narrow(2, 0, d)?, qkv.narrow(2, d, d)?, qkv.narrow(2, 2 * d, d)?);
            let (k, v) = (nn::split_heads(&k, self.heads)?, nn::split_heads(&v, self.heads)?);
            let (k, v) = match cache.get(i) {
                Some((ck, cv)) => (Tensor::cat(&[ck, &k], 2)?, Tensor::cat(&[cv, &v], 2)?),
                None => (k, v),
            };
            x = (&x + blk.self_o.forward(&nn::attend(&q, &k, &v, None)?)?)?;
            if i < cache.len() {
                cache[i] = (k, v);
            } else {
                cache.push((k, v));
            }
            let q = blk.cross_q.forward(&nn::layer_norm(&x)?)?;
            x = (&x + blk.cross_o.forward(&nn::attend(&q, &mem.k, &mem.v, Some(&mem.mask))?)?)?;
            x = (&x + blk.mlp_out.forward(&blk.mlp_in.forward(&nn::layer_norm(&x)?)?.relu()?)?)?;
        }
        nn::layer_norm(&x)
    }

    /// Says up to `L` tokens for every row of the memory (a batch of utterances), stopping each
    /// at `<end>`, with incremental decoding (cached self-attention keys and values, the running
    /// pointer distribution). Returns the tokens (padded with `<pad>` after `<end>`) and the
    /// log-probability of each utterance under the model.
    pub fn generate(&self, mem: &Memory, sampling: Sampling, rng: &mut Rng) -> Result<Vec<(Vec<u32>, f32)>> {
        let b = mem.mask.dims()[0];
        let device = mem.mask.device();
        let mut said: Vec<Vec<u32>> = vec![Vec::with_capacity(self.max_len); b];
        let mut logp = vec![0f32; b];
        let mut done = vec![false; b];
        let mut cache = Vec::with_capacity(self.blocks.len());
        let mut pointer: Option<Tensor> = None;
        let context = mem.context.to_vec2::<u32>()?;
        let mut input = Tensor::full(BOT, (b, 1), device)?;
        for step in 0..self.max_len {
            let h = self.step_hidden(&input, step, mem, &mut cache)?;
            let mut dists = candle_nn::ops::softmax(&self.logits(&h)?.squeeze(1)?, D::Minus1)?.to_vec2::<f32>()?;
            if let (Some(copy), Some(keys)) = (&self.copy, &mem.keys) {
                let (p, gate) = copy.pointer_step(&h, keys, Some(&mem.context), pointer.as_ref())?;
                let (pv, gv) =
                    (p.to_dtype(DType::F32)?.to_vec2::<f32>()?, gate.to_dtype(DType::F32)?.to_vec1::<f32>()?);
                pointer = Some(p);
                for (r, dist) in dists.iter_mut().enumerate() {
                    let lambda = 1.0 / (1.0 + (-gv[r]).exp());
                    dist.iter_mut().for_each(|x| *x *= 1.0 - lambda);
                    for (&tok, &w) in context[r].iter().zip(&pv[r]) {
                        if tok >= copy.min_token {
                            dist[tok as usize] += lambda * w;
                        }
                    }
                }
            }
            for (r, mut dist) in dists.into_iter().enumerate() {
                if done[r] {
                    said[r].push(PAD);
                    continue;
                }
                dist[PAD as usize] = 0.0; // padding only ever follows <end>
                let greedy = sampling.temperature <= 0.0 || step < sampling.greedy_prefix;
                let tok = if greedy { argmax(&dist) } else { sample(&dist, sampling, rng) };
                logp[r] += dist[tok as usize].max(1e-30).ln();
                said[r].push(tok);
                done[r] = tok == END;
            }
            if done.iter().all(|&d| d) {
                break;
            }
            let last: Vec<u32> = said.iter().map(|s| *s.last().expect("one token per step")).collect();
            input = Tensor::from_vec(last, (b, 1), device)?;
        }
        for s in &mut said {
            s.resize(self.max_len, PAD);
        }
        Ok(said.into_iter().zip(logp).collect())
    }
}

fn argmax(p: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[best] {
            best = i;
        }
    }
    best as u32
}

/// A token from `p` sharpened by the temperature, among the `top_k` most likely.
fn sample(p: &[f32], s: Sampling, rng: &mut Rng) -> u32 {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i] > 0.0).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    if s.top_k > 0 {
        idx.truncate(s.top_k);
    }
    let w: Vec<f64> = idx.iter().map(|&i| (p[i] as f64).powf(1.0 / s.temperature as f64)).collect();
    let total: f64 = w.iter().sum();
    let mut u = rng.uniform() * total;
    for (&i, &x) in idx.iter().zip(&w) {
        if u < x {
            return i as u32;
        }
        u -= x;
    }
    idx.first().copied().unwrap_or(END as usize) as u32
}

/// `[B, L]` u32 tensor of equally long rows.
pub fn rows_tensor(rows: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let l = rows.first().map_or(0, Vec::len);
    Tensor::from_vec(rows.concat(), (rows.len(), l), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(copy_dim: usize) -> (SpeechDecoder, ParamStore) {
        let mut ps = ParamStore::new(&Device::Cpu, 3);
        let cfg = SpeechConfig { d_model: 16, n_layers: 2, n_heads: 4, mlp_ratio: 2, max_len: 6, copy_dim };
        let dec = SpeechDecoder::new(&mut ps, &cfg, 30, 8, 3, 12, 20).unwrap();
        (dec, ps)
    }

    fn memory(dec: &SpeechDecoder, b: usize) -> Result<Memory> {
        let dev = Device::Cpu;
        let plan = Tensor::randn(0f32, 1.0, (b, 3, 8), &dev)?;
        let feats = Tensor::randn(0f32, 1.0, (b, 5, 12), &dev)?;
        let ctx = Tensor::from_vec((0..b * 5).map(|i| [0u32, 21, 22, 7, 23][i % 5]).collect(), (b, 5), &dev)?;
        let keys = match dec.copy {
            Some(ref c) => Some(Tensor::randn(0f32, 1.0, (b, 5, c.d_key()), &dev)?),
            None => None,
        };
        dec.memory(&plan, &feats, &ctx, keys.as_ref())
    }

    #[test]
    fn causal_outputs_do_not_see_the_future() -> Result<()> {
        let (dec, _) = tiny(0);
        let mem = memory(&dec, 1)?;
        let a = Tensor::from_vec(vec![BOT, 21, 22, 23], (1, 4), &Device::Cpu)?;
        let b = Tensor::from_vec(vec![BOT, 21, 5, 9], (1, 4), &Device::Cpu)?;
        let (ha, hb) = (dec.hidden(&a, &mem)?, dec.hidden(&b, &mem)?);
        let diff = |i: usize| -> Result<f32> {
            (ha.narrow(1, i, 1)? - hb.narrow(1, i, 1)?)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        assert!(diff(0)? < 1e-6 && diff(1)? < 1e-6, "positions before the change are identical");
        assert!(diff(2)? > 1e-4, "the changed position differs");
        Ok(())
    }

    #[test]
    fn distributions_sum_to_one_and_generation_stops_at_end() -> Result<()> {
        for copy in [0, 4] {
            let (dec, _) = tiny(copy);
            let mem = memory(&dec, 2)?;
            let prev = Tensor::from_vec(vec![BOT, 21, BOT, 22], (2, 2), &Device::Cpu)?;
            for d in dec.next_distribution(&prev, &mem)? {
                // pointer mass shifted onto a special token or past the end is dropped
                let s: f32 = d.iter().sum();
                assert!(s <= 1.0 + 1e-4 && s > if copy == 0 { 1.0 - 1e-4 } else { 0.5 }, "copy {copy}: {s}");
            }
            let out = dec.generate(&mem, Sampling::GREEDY, &mut Rng::new(1))?;
            for (toks, lp) in out {
                assert_eq!(toks.len(), 6);
                assert!(lp.is_finite() && lp <= 0.0);
                if let Some(e) = toks.iter().position(|&t| t == END) {
                    assert!(toks[e + 1..].iter().all(|&t| t == PAD));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn incremental_decoding_matches_the_full_pass() -> Result<()> {
        let (dec, _) = tiny(4);
        let mem = memory(&dec, 2)?;
        let out = dec.generate(&mem, Sampling::GREEDY, &mut Rng::new(1))?;
        // the log-probability of each greedy utterance, recomputed with full passes
        for (r, (toks, lp)) in out.iter().enumerate() {
            let row = Memory {
                k: mem.k.narrow(0, r, 1)?,
                v: mem.v.narrow(0, r, 1)?,
                mask: mem.mask.narrow(0, r, 1)?,
                keys: mem.keys.as_ref().map(|k| k.narrow(0, r, 1)).transpose()?,
                context: mem.context.narrow(0, r, 1)?,
            };
            let mut prev = vec![BOT];
            let mut full = 0f32;
            for &t in toks.iter().take_while(|&&t| t != PAD) {
                let p = Tensor::from_vec(prev.clone(), (1, prev.len()), &Device::Cpu)?;
                let mut d = dec.next_distribution(&p, &row)?.remove(0);
                d[PAD as usize] = 0.0;
                assert_eq!(argmax(&d), t, "greedy token differs");
                full += d[t as usize].max(1e-30).ln();
                prev.push(t);
            }
            assert!((full - lp).abs() < 1e-3, "row {r}: full {full} vs incremental {lp}");
        }
        Ok(())
    }

    #[test]
    fn learns_to_copy_from_the_context() -> Result<()> {
        // the answer is the second context token followed by <end>: only the pointer can know it
        let (dec, ps) = tiny(4);
        let vars = ps.vars();
        let mut opt = candle_nn::AdamW::new(vars, candle_nn::ParamsAdamW { lr: 1e-2, ..Default::default() })?;
        use candle_nn::Optimizer;
        let dev = Device::Cpu;
        let mut rng = Rng::new(9);
        let batch = |rng: &mut Rng| -> Result<(Tensor, Tensor)> {
            let (mut ctx, mut tgt) = (Vec::new(), Vec::new());
            for _ in 0..16 {
                let w = 20 + rng.below(10) as u32;
                ctx.extend([0, w, 7, 7, 0]);
                tgt.extend([w, END, PAD, PAD, PAD, PAD]);
            }
            Ok((Tensor::from_vec(ctx, (16, 5), &dev)?, Tensor::from_vec(tgt, (16, 6), &dev)?))
        };
        let mut last = f32::MAX;
        for step in 0..150 {
            let (ctx, tgt) = batch(&mut rng)?;
            // the features carry nothing: the answer can only come through the pointer, whose
            // keys know the positions (one-hot)
            let feats = Tensor::zeros((16, 5, 12), DType::F32, &dev)?;
            let keys = Tensor::eye(5, DType::F32, &dev)?.narrow(1, 0, 4)?.unsqueeze(0)?.repeat((16, 1, 1))?;
            let plan = Tensor::zeros((16, 3, 8), DType::F32, &dev)?;
            let mem = dec.memory(&plan, &feats, &ctx, Some(&keys))?;
            let loss = dec.loss(&tgt, &mem)?;
            opt.backward_step(&loss.total)?;
            if step >= 140 {
                last = last.min(loss.nll);
            }
        }
        assert!(last < 0.3, "copy loss stays at {last}");
        Ok(())
    }
}
