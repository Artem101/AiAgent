//! Module 1 — TTT encoder: context of any length → fixed-size state `S_prompt`.
//!
//! `tokens → (embedding + position + segment) → TTT-Linear scan over W_fast → readout`.
//!
//! Readout of the final fast weights uses learned probes `P ∈ ℝ^{d_fast×r}` and the running
//! mean of the layer outputs:
//! `S_prompt = MLP(LN([vec(W_fast^{(N)} P) ; mean_t z_t]))` — O(d²) memory, independent of N.

pub mod fast_weights;
pub mod layer;

use candle_core::{bail, DType, Result, Tensor};

pub use fast_weights::FastWeightsState;
pub use layer::{PackedTttLinear, TttLinear, TttSnapshot};

use crate::arena::Arena;
use crate::config::EngineConfig;
use crate::kernels::{add_inplace, gelu_inplace, inplace, layer_norm_rows, PackedLinear, PackedMlp, WeightBuf};
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
            readout: ps.mlp("ttt.readout", t.d_fast * t.readout_probes + t.d_fast, t.d_ctx, t.d_ctx)?,
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
        let feat = Tensor::cat(&[probed, snap.z_mean.clone()], 1)?;
        self.readout.forward(&nn::layer_norm(&feat)?)
    }

    /// Readouts `S` after each prefix length in `snapshots`.
    pub fn encode_snapshots(&self, x: &Tensor, snapshots: &[usize]) -> Result<Vec<Tensor>> {
        self.layer.scan(x, snapshots)?.iter().map(|s| self.readout(s)).collect()
    }

    /// `prompt: [B, N]` → `S_prompt: [B, d_ctx]` (graph path).
    pub fn encode(&self, prompt: &Tensor, dtype: DType) -> Result<Tensor> {
        let n = prompt.dims2()?.1;
        let x = self.embed(prompt, 0, SEGMENT_PROMPT)?.to_dtype(dtype)?;
        Ok(self.encode_snapshots(&x, &[n])?.remove(0))
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedTttEncoder> {
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
}

/// Pre-allocated buffers of [`PackedTttEncoder`] (O(d²), independent of the context length).
#[derive(Debug)]
pub struct TttWorkspace {
    x: Box<[f32]>,
    xn: Box<[f32]>,
    k: Box<[f32]>,
    v: Box<[f32]>,
    q: Box<[f32]>,
    z: Box<[f32]>,
    z_sum: Box<[f32]>,
    feat: Box<[f32]>,
    hidden: Box<[f32]>,
    /// Fast weights `W_fast`.
    pub state: FastWeightsState,
    prompt: Tensor,
    n_tokens: usize,
}

impl TttWorkspace {
    pub fn tokens_seen(&self) -> usize {
        self.n_tokens
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
            xn: arena.host(dm),
            k: arena.host(df),
            v: arena.host(df),
            q: arena.host(df),
            z: arena.host(df),
            z_sum: arena.host(df),
            feat: arena.host(self.readout.l1.d_in),
            hidden: arena.host(self.readout.l1.d_out),
            state: FastWeightsState::in_arena(arena, df, self.layer.eta as f64)?,
            prompt: arena.tensor(self.d_ctx())?,
            n_tokens: 0,
        })
    }

    /// Clears the fast weights for a new context.
    pub fn reset(&self, ws: &mut TttWorkspace) -> Result<()> {
        ws.state.reset()?;
        ws.z_sum.fill(0.0);
        ws.n_tokens = 0;
        Ok(())
    }

    /// Absorbs one token: embed → project → `W ← W − η_t (W k − v) ⊗ k` → `z = W q`.
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
        let eta = self.layer.project(&ws.x, &mut ws.xn, &mut ws.k, &mut ws.v, &mut ws.q);
        ws.state.step_update_host(&ws.k, &ws.v, eta)?;
        ws.state.forward_host(&ws.q, &mut ws.z)?;
        add_inplace(&mut ws.z_sum, &ws.z);
        ws.n_tokens += 1;
        Ok(())
    }

    /// Reads `S_prompt` out of the current fast weights.
    pub fn finish(&self, ws: &mut TttWorkspace) -> Result<PromptState> {
        let TttWorkspace { feat, z_sum, hidden, state, prompt, n_tokens, .. } = ws;
        let rd = self.d_fast() * self.probes.d_out;
        inplace::host_read(&state.weights, |w| self.probes.forward(w, &mut feat[..rd]))?;
        let inv = if *n_tokens > 0 { 1.0 / *n_tokens as f32 } else { 0.0 };
        for (f, &z) in feat[rd..].iter_mut().zip(z_sum.iter()) {
            *f = z * inv;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn packed_encoder_matches_graph() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EngineConfig::tiny(11, 6, 4);
        let mut ps = ParamStore::new(&dev, 3);
        let enc = TttEncoder::new(&mut ps, &cfg)?;
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9];
        let prompt = Tensor::from_slice(&tokens, (1, 6), &dev)?;
        let want = enc.encode(&prompt, DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;

        let packed = enc.pack(DType::F32)?;
        let mut arena = Arena::new(&dev);
        let mut ws = packed.workspace(&mut arena)?;
        let got = packed.encode(&tokens, &mut ws)?.tensor().to_vec1::<f32>()?;
        let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "f32 packed vs graph: {err}");

        let packed16 = enc.pack(DType::BF16)?;
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
        assert!(packed.absorb(99, SEGMENT_PROMPT, &mut ws).is_err());
        Ok(())
    }
}
