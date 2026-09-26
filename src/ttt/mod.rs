//! Module 1 — TTT encoder: context of any length → fixed-size state `S_prompt`.
//!
//! `tokens → (embedding + position + segment) → TTT-Linear scan over W_fast → readout`.
//!
//! Readout of the final fast weights uses learned probes `P ∈ ℝ^{d_fast×r}`, the running
//! mean of the layer outputs and, optionally, the last `m = readout_last` outputs and `p =
//! readout_pools` gated pools:
//! `S_prompt = MLP(LN([vec(W_fast^{(N)} P) ; mean_t z_t ; z_N ; … ; z_{N−m+1} ; pool_1 ; … ; pool_p]))`,
//! `pool_j = Σ_t g_{t,j} z_t / (Σ_t g_{t,j} + ε)` — O(d²) memory, independent of N.
//!
//! With the copy mechanism ([`crate::copy`], `jepa.copy_dim > 0`) the encoder also keeps a key
//! `k_t = W_k [x̃_t ; z_t]` and the token id of every prompt position — an O(N · d_key) copy
//! memory, bounded by `max_prompt_len`, that the decoder can point into.

pub mod fast_weights;
pub mod layer;

use candle_core::{bail, DType, Result, Tensor};

pub use fast_weights::FastWeightsState;
pub use layer::{PackedTttLinear, TttLinear, TttSnapshot};

use crate::arena::Arena;
use crate::config::EngineConfig;
use crate::kernels::{
    add_inplace, gelu_inplace, inplace, layer_norm_rows, layer_norm_rows_into, PackedLinear, PackedMlp, WeightBuf,
};
use crate::nn::{self, Init, Lin, Mlp, ParamStore};
use crate::types::PromptState;

/// Segment ids added to the token embeddings.
pub const SEGMENT_PROMPT: usize = 0;
pub const SEGMENT_ANSWER: usize = 1;

#[derive(Debug, Clone)]
pub struct TttEncoder {
    /// `[vocab, d_model]`.
    pub tok_emb: Tensor,
    /// `[max_positions, d_model]`.
    pub pos_emb: Tensor,
    /// `[2, d_model]` (prompt / answer continuation).
    pub seg_emb: Tensor,
    pub layer: TttLinear,
    /// Probe matrix stored as a bias-free linear layer with weight `Pᵀ ∈ ℝ^{r×d_fast}`.
    pub probes: Lin,
    pub readout: Mlp,
    /// Number of final layer outputs fed to the readout.
    pub readout_last: usize,
    /// Keys of the copy mechanism, `[x̃_t ; z_t] → k_t` (when `jepa.copy_dim > 0`).
    pub copy_key: Option<Lin>,
}

impl TttEncoder {
    pub fn new(ps: &mut ParamStore, cfg: &EngineConfig) -> Result<Self> {
        let t = &cfg.ttt;
        Ok(Self {
            tok_emb: ps.tensor("ttt.tok_emb", &[cfg.vocab_size, t.d_model], Init::Normal(1.0))?,
            pos_emb: ps.tensor("ttt.pos_emb", &[cfg.max_positions(), t.d_model], Init::Normal(0.5))?,
            seg_emb: ps.tensor("ttt.seg_emb", &[2, t.d_model], Init::Normal(0.5))?,
            layer: TttLinear::new(ps, "ttt.layer", t)?,
            probes: ps.linear("ttt.probes", t.d_fast, t.readout_probes, false)?,
            readout: ps.mlp(
                "ttt.readout",
                t.d_fast * t.readout_probes + t.d_fast * (1 + t.readout_last + t.readout_pools),
                t.d_ctx,
                t.d_ctx,
            )?,
            readout_last: t.readout_last,
            copy_key: if cfg.jepa.copy_dim > 0 {
                Some(ps.linear("ttt.copy_key", t.d_model * t.conv_width + t.d_fast, cfg.jepa.copy_dim, false)?)
            } else {
                None
            },
        })
    }

    pub fn d_ctx(&self) -> usize {
        self.readout.l2.d_out()
    }

    /// `tokens: [B, T]` (u32) → `[B, T, d_model]` with positions starting at `pos_offset`.
    pub fn embed(&self, tokens: &Tensor, pos_offset: usize, segment: usize) -> Result<Tensor> {
        let (b, t) = tokens.dims2()?;
        let d = self.tok_emb.dims()[1];
        let tok = self.tok_emb.index_select(&tokens.flatten_all()?, 0)?.reshape((b, t, d))?;
        let pos = self.pos_emb.narrow(0, pos_offset, t)?;
        let seg = self.seg_emb.narrow(0, segment, 1)?;
        tok.broadcast_add(&pos)?.broadcast_add(&seg)
    }

    /// Readout MLP applied to a scan snapshot → `[B, d_ctx]`.
    pub fn readout(&self, snap: &TttSnapshot) -> Result<Tensor> {
        let (b, d, _) = snap.w.dims3()?;
        let r = self.probes.d_out();
        let probed = self.probes.forward(&snap.w)?.reshape((b, d * r))?; // vec(W P)
        let mut parts = vec![probed, snap.z_mean.clone()];
        parts.extend(snap.z_last.iter().cloned());
        parts.extend(snap.z_pool.iter().cloned());
        let feat = Tensor::cat(&parts, 1)?;
        self.readout.forward(&nn::layer_norm(&feat)?)
    }

    /// Readouts `S` after each prefix length in `snapshots`.
    pub fn encode_snapshots(&self, x: &Tensor, snapshots: &[usize]) -> Result<Vec<Tensor>> {
        self.layer.scan(x, snapshots, self.readout_last)?.iter().map(|s| self.readout(s)).collect()
    }

    /// Readouts after each prefix length in `snapshots`, plus the copy keys `[B, n, d_key]` of
    /// the first `n` tokens (`None` without a copy mechanism).
    pub fn encode_with_keys(&self, x: &Tensor, snapshots: &[usize], n: usize) -> Result<(Vec<Tensor>, Option<Tensor>)> {
        let keep = if self.copy_key.is_some() { n } else { 0 };
        let (snaps, outputs) = self.layer.scan_with_outputs(x, snapshots, self.readout_last, keep)?;
        let readouts = snaps.iter().map(|s| self.readout(s)).collect::<Result<Vec<_>>>()?;
        let keys = match (&self.copy_key, outputs) {
            (Some(k), Some((window, z))) => Some(k.forward(&Tensor::cat(&[window, z], 2)?)?),
            _ => None,
        };
        Ok((readouts, keys))
    }

    /// Readouts after each prefix length in `snapshots`, the per-token features `[x̃_t ; z_t]`
    /// `[B, n, w·d_model + d_fast]` of the first `n` tokens (causal window and layer output — the
    /// context memory the speech decoder attends to, see [`crate::speech`]) and their copy keys
    /// (`None` without a copy mechanism).
    pub fn encode_with_features(
        &self,
        x: &Tensor,
        snapshots: &[usize],
        n: usize,
    ) -> Result<(Vec<Tensor>, Tensor, Option<Tensor>)> {
        let (snaps, outputs) = self.layer.scan_with_outputs(x, snapshots, self.readout_last, n)?;
        let readouts = snaps.iter().map(|s| self.readout(s)).collect::<Result<Vec<_>>>()?;
        let Some((window, z)) = outputs else { bail!("encode_with_features: no context tokens (n = 0)") };
        let feats = Tensor::cat(&[window, z], 2)?;
        let keys = self.copy_key.as_ref().map(|k| k.forward(&feats)).transpose()?;
        Ok((readouts, feats, keys))
    }

    /// Width of the per-token features of [`TttEncoder::encode_with_features`].
    pub fn d_features(&self) -> usize {
        self.layer.wk.d_in() + self.layer.d_fast()
    }

    /// `prompt: [B, N]` → `S_prompt: [B, d_ctx]` (graph path).
    pub fn encode(&self, prompt: &Tensor, dtype: DType) -> Result<Tensor> {
        let n = prompt.dims2()?.1;
        let x = self.embed(prompt, 0, SEGMENT_PROMPT)?.to_dtype(dtype)?;
        Ok(self.encode_snapshots(&x, &[n])?.remove(0))
    }

    /// Packs the weights; the copy memory holds the keys of the first `copy_capacity` tokens.
    pub fn pack(&self, dtype: DType, copy_capacity: usize) -> Result<PackedTttEncoder> {
        let (vocab, d_model) = self.tok_emb.dims2()?;
        Ok(PackedTttEncoder {
            tok_emb: WeightBuf::from_tensor(&self.tok_emb, dtype)?,
            pos_emb: WeightBuf::from_tensor(&self.pos_emb, dtype)?,
            seg_emb: WeightBuf::from_tensor(&self.seg_emb, dtype)?,
            layer: self.layer.pack(dtype)?,
            probes: self.probes.pack(dtype)?,
            readout: self.readout.pack(dtype)?,
            vocab,
            max_pos: self.pos_emb.dims()[0],
            d_model,
            readout_last: self.readout_last,
            copy_key: self.copy_key.as_ref().map(|k| k.pack(dtype)).transpose()?,
            copy_capacity,
        })
    }
}

/// Host-kernel TTT encoder: streams tokens one at a time through the fused rank-1 update.
#[derive(Debug, Clone)]
pub struct PackedTttEncoder {
    pub tok_emb: WeightBuf,
    pub pos_emb: WeightBuf,
    pub seg_emb: WeightBuf,
    pub layer: PackedTttLinear,
    pub probes: PackedLinear,
    pub readout: PackedMlp,
    pub vocab: usize,
    pub max_pos: usize,
    pub d_model: usize,
    pub readout_last: usize,
    pub copy_key: Option<PackedLinear>,
    /// Positions the copy memory holds.
    pub copy_capacity: usize,
}

/// Pre-allocated buffers of [`PackedTttEncoder`] (O(d²), independent of the context length).
#[derive(Debug)]
pub struct TttWorkspace {
    x: Box<[f32]>,
    /// Causal window `[LN(x_t); LN(x_{t−1}); …]`.
    xn: Box<[f32]>,
    k: Box<[f32]>,
    v: Box<[f32]>,
    q: Box<[f32]>,
    z: Box<[f32]>,
    z_sum: Box<[f32]>,
    /// The last `readout_last` outputs, newest first.
    z_last: Box<[f32]>,
    /// Gates of the current token, and running `Σ g z` / `Σ g` of the readout pools.
    gates: Box<[f32]>,
    pool_num: Box<[f32]>,
    pool_den: Box<[f32]>,
    feat: Box<[f32]>,
    hidden: Box<[f32]>,
    /// Copy memory: `[x̃_t ; z_t]` scratch, keys `[capacity, d_key]` and token ids.
    copy_in: Box<[f32]>,
    copy_keys: Box<[f32]>,
    copy_tokens: Box<[u32]>,
    /// Fast weights `W_fast`.
    pub state: FastWeightsState,
    prompt: Tensor,
    n_tokens: usize,
}

impl TttWorkspace {
    pub fn tokens_seen(&self) -> usize {
        self.n_tokens
    }

    /// Copy keys (`n × d_key`) and token ids (`n`) of the absorbed prompt (empty without a
    /// copy mechanism).
    pub fn copy_memory(&self) -> (&[f32], &[u32]) {
        let n = self.n_tokens.min(self.copy_tokens.len());
        let dk = self.copy_keys.len() / self.copy_tokens.len().max(1);
        (&self.copy_keys[..n * dk], &self.copy_tokens[..n])
    }
}

impl PackedTttEncoder {
    pub fn d_fast(&self) -> usize {
        self.layer.d_fast()
    }
    pub fn d_ctx(&self) -> usize {
        self.readout.l2.d_out
    }

    pub fn workspace(&self, arena: &mut Arena) -> Result<TttWorkspace> {
        let (dm, df) = (self.d_model, self.d_fast());
        Ok(TttWorkspace {
            x: arena.host(dm),
            xn: arena.host(dm * self.layer.conv_width),
            k: arena.host(df),
            v: arena.host(df),
            q: arena.host(df),
            z: arena.host(df),
            z_sum: arena.host(df),
            z_last: arena.host(df * self.readout_last),
            gates: arena.host(self.layer.pools()),
            pool_num: arena.host(df * self.layer.pools()),
            pool_den: arena.host(self.layer.pools()),
            feat: arena.host(self.readout.l1.d_in),
            hidden: arena.host(self.readout.l1.d_out),
            copy_in: arena.host(self.copy_key.as_ref().map_or(0, |k| k.d_in)),
            copy_keys: arena.host(self.copy_key.as_ref().map_or(0, |k| k.d_out * self.copy_capacity)),
            copy_tokens: arena.host_u32(if self.copy_key.is_some() { self.copy_capacity } else { 0 }),
            state: FastWeightsState::in_arena(arena, df, self.layer.eta as f64)?,
            prompt: arena.tensor(self.d_ctx())?,
            n_tokens: 0,
        })
    }

    /// Clears the fast weights for a new context.
    pub fn reset(&self, ws: &mut TttWorkspace) -> Result<()> {
        ws.state.reset()?;
        ws.xn.fill(0.0);
        ws.z_sum.fill(0.0);
        ws.z_last.fill(0.0);
        ws.pool_num.fill(0.0);
        ws.pool_den.fill(0.0);
        ws.n_tokens = 0;
        Ok(())
    }

    /// Absorbs one token: embed → causal window → project → `W ← W − η_t (W k − v) ⊗ k` → `z = W q`.
    /// Positions beyond the learned table reuse its last row, so the stream length is unbounded.
    pub fn absorb(&self, token: u32, segment: usize, ws: &mut TttWorkspace) -> Result<()> {
        if token as usize >= self.vocab {
            bail!("token id {token} out of range (vocab {})", self.vocab)
        }
        let dm = self.d_model;
        let pos = ws.n_tokens.min(self.max_pos - 1);
        self.tok_emb.row_into(token as usize, dm, &mut ws.x);
        self.pos_emb.add_row_into(pos, dm, &mut ws.x);
        self.seg_emb.add_row_into(segment, dm, &mut ws.x);
        // Shift the window by one token and normalise the new one into slot 0.
        let len = ws.xn.len();
        ws.xn.copy_within(..len - dm, dm);
        layer_norm_rows_into(&ws.x, &mut ws.xn[..dm], dm);
        let eta = self.layer.project(&ws.xn, &mut ws.k, &mut ws.v, &mut ws.q);
        ws.state.step_update_host(&ws.k, &ws.v, eta)?;
        ws.state.forward_host(&ws.q, &mut ws.z)?;
        add_inplace(&mut ws.z_sum, &ws.z);
        if !ws.z_last.is_empty() {
            let (len, df) = (ws.z_last.len(), ws.z.len());
            ws.z_last.copy_within(..len - df, df);
            ws.z_last[..df].copy_from_slice(&ws.z);
        }
        if let Some(key) = &self.copy_key {
            let t = ws.n_tokens;
            if t < ws.copy_tokens.len() {
                let wd = ws.xn.len();
                ws.copy_in[..wd].copy_from_slice(&ws.xn);
                ws.copy_in[wd..].copy_from_slice(&ws.z);
                key.forward(&ws.copy_in, &mut ws.copy_keys[t * key.d_out..(t + 1) * key.d_out]);
                ws.copy_tokens[t] = token;
            }
        }
        if !ws.gates.is_empty() {
            self.layer.gates(&ws.xn, &mut ws.gates);
            for ((num, den), &g) in
                ws.pool_num.chunks_exact_mut(ws.z.len()).zip(ws.pool_den.iter_mut()).zip(ws.gates.iter())
            {
                *den += g;
                crate::kernels::axpy(num, g, &ws.z);
            }
        }
        ws.n_tokens += 1;
        Ok(())
    }

    /// Reads `S_prompt` out of the current fast weights.
    pub fn finish(&self, ws: &mut TttWorkspace) -> Result<PromptState> {
        let TttWorkspace { feat, z_sum, z_last, pool_num, pool_den, hidden, state, prompt, n_tokens, .. } = ws;
        let (rd, df) = (self.d_fast() * self.probes.d_out, self.d_fast());
        inplace::host_read(&state.weights, |w| self.probes.forward(w, &mut feat[..rd]))?;
        let inv = if *n_tokens > 0 { 1.0 / *n_tokens as f32 } else { 0.0 };
        for (f, &z) in feat[rd..rd + df].iter_mut().zip(z_sum.iter()) {
            *f = z * inv;
        }
        let zl = rd + df + z_last.len();
        feat[rd + df..zl].copy_from_slice(z_last);
        for ((f, num), &den) in feat[zl..].chunks_exact_mut(df).zip(pool_num.chunks_exact(df)).zip(pool_den.iter()) {
            let inv = 1.0 / (den + layer::POOL_EPS as f32);
            f.iter_mut().zip(num).for_each(|(f, &n)| *f = n * inv);
        }
        let n = feat.len();
        layer_norm_rows(feat, n);
        self.readout.l1.forward(feat, hidden);
        gelu_inplace(hidden);
        inplace::host_write(prompt, |out| {
            self.readout.l2.forward(hidden, out);
            Ok(())
        })?;
        PromptState::new(prompt.clone(), self.d_ctx())
    }

    /// Encodes a full prompt (zero allocations after the workspace exists).
    pub fn encode(&self, tokens: &[u32], ws: &mut TttWorkspace) -> Result<PromptState> {
        self.reset(ws)?;
        for &tok in tokens {
            self.absorb(tok, SEGMENT_PROMPT, ws)?;
        }
        self.finish(ws)
    }

    pub fn bytes(&self) -> usize {
        self.tok_emb.bytes()
            + self.pos_emb.bytes()
            + self.seg_emb.bytes()
            + self.layer.bytes()
            + self.probes.bytes()
            + self.readout.bytes()
            + self.copy_key.as_ref().map_or(0, |k| k.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn packed_encoder_matches_graph() -> Result<()> {
        encoder_parity(EngineConfig::tiny(11, 6, 4))?;
        // causal window over 3 tokens + the last 2 outputs in the readout
        let mut cfg = EngineConfig::tiny(11, 6, 4);
        cfg.ttt.conv_width = 3;
        cfg.ttt.readout_last = 2;
        cfg.ttt.readout_pools = 3;
        encoder_parity(cfg.clone())?;
        // copy memory
        cfg.jepa.copy_dim = 5;
        encoder_parity(cfg)
    }

    fn encoder_parity(cfg: EngineConfig) -> Result<()> {
        let dev = Device::Cpu;
        let mut ps = ParamStore::new(&dev, 3);
        let enc = TttEncoder::new(&mut ps, &cfg)?;
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9];
        let prompt = Tensor::from_slice(&tokens, (1, 6), &dev)?;
        let want = enc.encode(&prompt, DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;

        let packed = enc.pack(DType::F32, 6)?;
        let mut arena = Arena::new(&dev);
        let mut ws = packed.workspace(&mut arena)?;
        let got = packed.encode(&tokens, &mut ws)?.tensor().to_vec1::<f32>()?;
        let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "f32 packed vs graph: {err}");
        if enc.copy_key.is_some() {
            let x = enc.embed(&prompt, 0, SEGMENT_PROMPT)?;
            let (_, keys) = enc.encode_with_keys(&x, &[6], 6)?;
            let want = keys.unwrap().flatten_all()?.to_vec1::<f32>()?;
            let (got, toks) = ws.copy_memory();
            assert_eq!(toks, &tokens[..]);
            let err = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err < 1e-4, "copy keys packed vs graph: {err}");
        } else {
            assert!(ws.copy_memory().1.is_empty());
        }

        let packed16 = enc.pack(DType::BF16, 6)?;
        let mut ws16 = packed16.workspace(&mut Arena::new(&dev))?;
        let got16 = packed16.encode(&tokens, &mut ws16)?.tensor().to_vec1::<f32>()?;
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let err16 = want.iter().zip(&got16).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err16 < 0.05 * scale.max(1.0), "bf16 packed vs graph: {err16}");

        // Fast-weight memory is O(d²) regardless of the number of absorbed tokens.
        let before = ws.state.weights.elem_count();
        for i in 0..1000u32 {
            packed.absorb(i % 11, SEGMENT_PROMPT, &mut ws)?;
        }
        assert_eq!(ws.state.weights.elem_count(), before);
        assert_eq!(ws.tokens_seen(), 1006);
        assert!(ws.copy_memory().1.len() <= 6, "the copy memory is bounded");
        assert!(packed.absorb(99, SEGMENT_PROMPT, &mut ws).is_err());
        Ok(())
    }
}
