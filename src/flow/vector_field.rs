//! DiT-style vector field `v_θ(X_t, t, S_plan)`.
//!
//! ```text
//!  c   = MLP_t(sinusoidal(t)) + W_pool · mean(S_plan)              (global condition)
//!  h   = W_in X_t + pos                                             [L, d_h]
//!  mem = W_plan S_plan + plan_pos                                   [H+1, d_h]
//!  per block (adaLN-Zero, modulation = W_mod SiLU(c)):
//!     h += g1 ⊙ SelfAttn(LN(h)(1+s1)+b1)
//!     h += CrossAttn(Q = LN(h), K/V = mem)                          (plan conditioning)
//!     h += g2 ⊙ MLP(LN(h)(1+s2)+b2)
//!  v   = W_out (LN(h)(1+s_f)+b_f)
//! ```
//!
//! The host version caches the per-layer cross-attention K/V of the plan in
//! [`VectorFieldEstimator::prepare`] — they do not depend on `t`, so each ODE step only
//! runs the `L`-row part of the network.

use candle_core::{DType, Result, Tensor, TensorId};

use super::ode_solver::VectorFieldEstimator;
use crate::arena::Arena;
use crate::config::EngineConfig;
use crate::kernels::inplace::{host_read, host_write};
use crate::kernels::{
    add_gated_rows, add_inplace, attention, gelu_inplace, layer_norm_rows_into, mean_rows_into, modulate_rows,
    silu_into, sinusoidal_embedding, PackedLinear, WeightBuf,
};
use crate::nn::{self, Init, Lin, ParamStore};
use crate::types::LatentPlan;

#[derive(Debug, Clone)]
pub struct DitBlock {
    /// `SiLU(c) → [shift1, scale1, gate1, shift2, scale2, gate2]` (zero-initialised).
    pub modulation: Lin,
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub xq: Lin,
    pub xk: Lin,
    pub xv: Lin,
    pub xo: Lin,
    pub fc1: Lin,
    pub fc2: Lin,
}

#[derive(Debug, Clone)]
pub struct VectorField {
    pub heads: usize,
    pub d_time: usize,
    pub in_proj: Lin,
    /// `[L, d_h]`.
    pub pos: Tensor,
    pub plan_proj: Lin,
    /// `[H + 1, d_h]`.
    pub plan_pos: Tensor,
    pub plan_pool: Lin,
    pub t1: Lin,
    pub t2: Lin,
    pub blocks: Vec<DitBlock>,
    pub final_mod: Lin,
    pub out_proj: Lin,
}

impl VectorField {
    pub fn new(ps: &mut ParamStore, cfg: &EngineConfig) -> Result<Self> {
        let f = &cfg.flow;
        let (d, ds) = (f.d_hidden, cfg.jepa.d_state);
        let mut blocks = Vec::with_capacity(f.n_layers);
        for i in 0..f.n_layers {
            let n = |s: &str| format!("flow.blocks.{i}.{s}");
            blocks.push(DitBlock {
                modulation: ps.linear_zero(&n("modulation"), d, 6 * d)?,
                q: ps.linear(&n("attn.q"), d, d, true)?,
                k: ps.linear(&n("attn.k"), d, d, true)?,
                v: ps.linear(&n("attn.v"), d, d, true)?,
                o: ps.linear(&n("attn.o"), d, d, true)?,
                xq: ps.linear(&n("xattn.q"), d, d, true)?,
                xk: ps.linear(&n("xattn.k"), d, d, true)?,
                xv: ps.linear(&n("xattn.v"), d, d, true)?,
                xo: ps.linear(&n("xattn.o"), d, d, true)?,
                fc1: ps.linear(&n("mlp.fc1"), d, f.mlp_ratio * d, true)?,
                fc2: ps.linear(&n("mlp.fc2"), f.mlp_ratio * d, d, true)?,
            });
        }
        Ok(Self {
            heads: f.n_heads,
            d_time: f.d_time,
            in_proj: ps.linear("flow.in_proj", f.d_token, d, true)?,
            pos: ps.tensor("flow.pos", &[f.seq_len, d], Init::Normal(0.1))?,
            plan_proj: ps.linear("flow.plan_proj", ds, d, true)?,
            plan_pos: ps.tensor("flow.plan_pos", &[cfg.plan_len(), d], Init::Normal(0.1))?,
            plan_pool: ps.linear("flow.plan_pool", ds, d, true)?,
            t1: ps.linear("flow.t1", f.d_time, d, true)?,
            t2: ps.linear("flow.t2", d, d, true)?,
            blocks,
            final_mod: ps.linear_zero("flow.final_mod", d, 2 * d)?,
            out_proj: ps.linear_zero("flow.out_proj", d, f.d_token)?,
        })
    }

    pub fn d_hidden(&self) -> usize {
        self.in_proj.d_out()
    }

    /// `x: [B, L, d_token]`, `t: [B]`, `plan: [B, H+1, d_s]` → velocity `[B, L, d_token]`.
    /// Runs in `x.dtype()` (mixed precision); norms/softmax are computed in f32.
    pub fn forward(&self, x: &Tensor, t: &Tensor, plan: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let heads = self.heads;
        let temb = nn::sinusoidal(t, self.d_time)?.to_dtype(dt)?;
        let plan = plan.to_dtype(dt)?;
        let c = (self.t2.forward(&self.t1.forward(&temb)?.silu()?)? + self.plan_pool.forward(&plan.mean(1)?)?)?;
        let sc = c.silu()?;
        let mut h = self.in_proj.forward(x)?.broadcast_add(&self.pos.to_dtype(dt)?)?;
        let mem = self.plan_proj.forward(&plan)?.broadcast_add(&self.plan_pos.to_dtype(dt)?)?;
        for blk in &self.blocks {
            let m = blk.modulation.forward(&sc)?.chunk(6, 1)?;
            let a = nn::modulate(&nn::layer_norm(&h)?, &m[0], &m[1])?;
            let att = blk.o.forward(&nn::mha(&blk.q.forward(&a)?, &blk.k.forward(&a)?, &blk.v.forward(&a)?, heads)?)?;
            h = (h + att.broadcast_mul(&m[2].unsqueeze(1)?)?)?;
            let a = nn::layer_norm(&h)?;
            let xatt = nn::mha(&blk.xq.forward(&a)?, &blk.xk.forward(&mem)?, &blk.xv.forward(&mem)?, heads)?;
            h = (h + blk.xo.forward(&xatt)?)?;
            let a = nn::modulate(&nn::layer_norm(&h)?, &m[3], &m[4])?;
            let y = blk.fc2.forward(&blk.fc1.forward(&a)?.gelu()?)?;
            h = (h + y.broadcast_mul(&m[5].unsqueeze(1)?)?)?;
        }
        let fm = self.final_mod.forward(&sc)?.chunk(2, 1)?;
        self.out_proj.forward(&nn::modulate(&nn::layer_norm(&h)?, &fm[0], &fm[1])?)
    }

    /// Packs the weights and allocates the per-sequence workspace from `arena`.
    pub fn pack(&self, dtype: DType, arena: &mut Arena) -> Result<PackedVectorField> {
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                Ok(PackedBlock {
                    modulation: b.modulation.pack(dtype)?,
                    q: b.q.pack(dtype)?,
                    k: b.k.pack(dtype)?,
                    v: b.v.pack(dtype)?,
                    o: b.o.pack(dtype)?,
                    xq: b.xq.pack(dtype)?,
                    xk: b.xk.pack(dtype)?,
                    xv: b.xv.pack(dtype)?,
                    xo: b.xo.pack(dtype)?,
                    fc1: b.fc1.pack(dtype)?,
                    fc2: b.fc2.pack(dtype)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let w = VfWeights {
            heads: self.heads,
            d_time: self.d_time,
            seq_len: self.pos.dims()[0],
            plan_len: self.plan_pos.dims()[0],
            in_proj: self.in_proj.pack(dtype)?,
            pos: WeightBuf::from_tensor(&self.pos, dtype)?,
            plan_proj: self.plan_proj.pack(dtype)?,
            plan_pos: WeightBuf::from_tensor(&self.plan_pos, dtype)?,
            plan_pool: self.plan_pool.pack(dtype)?,
            t1: self.t1.pack(dtype)?,
            t2: self.t2.pack(dtype)?,
            blocks,
            final_mod: self.final_mod.pack(dtype)?,
            out_proj: self.out_proj.pack(dtype)?,
        };
        let ws = VfWorkspace::new(&w, arena);
        Ok(PackedVectorField { w, ws })
    }
}

#[derive(Debug, Clone)]
struct PackedBlock {
    modulation: PackedLinear,
    q: PackedLinear,
    k: PackedLinear,
    v: PackedLinear,
    o: PackedLinear,
    xq: PackedLinear,
    xk: PackedLinear,
    xv: PackedLinear,
    xo: PackedLinear,
    fc1: PackedLinear,
    fc2: PackedLinear,
}

impl PackedBlock {
    fn bytes(&self) -> usize {
        [
            &self.modulation,
            &self.q,
            &self.k,
            &self.v,
            &self.o,
            &self.xq,
            &self.xk,
            &self.xv,
            &self.xo,
            &self.fc1,
            &self.fc2,
        ]
        .iter()
        .map(|l| l.bytes())
        .sum()
    }
}

#[derive(Debug, Clone)]
struct VfWeights {
    heads: usize,
    d_time: usize,
    seq_len: usize,
    plan_len: usize,
    in_proj: PackedLinear,
    pos: WeightBuf,
    plan_proj: PackedLinear,
    plan_pos: WeightBuf,
    plan_pool: PackedLinear,
    t1: PackedLinear,
    t2: PackedLinear,
    blocks: Vec<PackedBlock>,
    final_mod: PackedLinear,
    out_proj: PackedLinear,
}

impl VfWeights {
    fn d(&self) -> usize {
        self.in_proj.d_out
    }
}

#[derive(Debug)]
struct VfWorkspace {
    temb: Box<[f32]>,
    c: Box<[f32]>,
    c_tmp: Box<[f32]>,
    sc: Box<[f32]>,
    plan_c: Box<[f32]>,
    plan_mean: Box<[f32]>,
    modv: Box<[f32]>,
    fmod: Box<[f32]>,
    h: Box<[f32]>,
    a: Box<[f32]>,
    q: Box<[f32]>,
    k: Box<[f32]>,
    v: Box<[f32]>,
    att: Box<[f32]>,
    proj: Box<[f32]>,
    mlp: Box<[f32]>,
    mem: Box<[f32]>,
    /// Per-layer cross-attention keys/values of the plan (`[H+1, d_h]` each).
    xk: Vec<Box<[f32]>>,
    xv: Vec<Box<[f32]>>,
    scores: Box<[f32]>,
    prepared_for: Option<TensorId>,
}

impl VfWorkspace {
    fn new(w: &VfWeights, arena: &mut Arena) -> Self {
        let (d, l, p) = (w.d(), w.seq_len, w.plan_len);
        let hidden = w.blocks.first().map_or(d, |b| b.fc1.d_out);
        Self {
            temb: arena.host(w.d_time),
            c: arena.host(d),
            c_tmp: arena.host(d),
            sc: arena.host(d),
            plan_c: arena.host(d),
            plan_mean: arena.host(w.plan_proj.d_in),
            modv: arena.host(6 * d),
            fmod: arena.host(2 * d),
            h: arena.host(l * d),
            a: arena.host(l * d),
            q: arena.host(l * d),
            k: arena.host(l * d),
            v: arena.host(l * d),
            att: arena.host(l * d),
            proj: arena.host(l * d),
            mlp: arena.host(l * hidden),
            mem: arena.host(p * d),
            xk: w.blocks.iter().map(|_| arena.host(p * d)).collect(),
            xv: w.blocks.iter().map(|_| arena.host(p * d)).collect(),
            scores: arena.host(w.heads * l * l.max(p)),
            prepared_for: None,
        }
    }
}

/// Host-kernel vector field with its own pre-allocated workspace.
#[derive(Debug)]
pub struct PackedVectorField {
    w: VfWeights,
    ws: VfWorkspace,
}

impl PackedVectorField {
    pub fn seq_len(&self) -> usize {
        self.w.seq_len
    }
    pub fn d_token(&self) -> usize {
        self.w.in_proj.d_in
    }
    pub fn bytes(&self) -> usize {
        let w = &self.w;
        w.in_proj.bytes()
            + w.pos.bytes()
            + w.plan_proj.bytes()
            + w.plan_pos.bytes()
            + w.plan_pool.bytes()
            + w.t1.bytes()
            + w.t2.bytes()
            + w.final_mod.bytes()
            + w.out_proj.bytes()
            + w.blocks.iter().map(|b| b.bytes()).sum::<usize>()
    }
}

impl VectorFieldEstimator for PackedVectorField {
    fn prepare(&mut self, plan: &LatentPlan) -> Result<()> {
        let (w, ws) = (&self.w, &mut self.ws);
        let d = w.d();
        if plan.len() != w.plan_len || plan.d_state() != w.plan_proj.d_in {
            candle_core::bail!("plan shape [{}, {}] does not match the vector field", plan.len(), plan.d_state())
        }
        host_read(plan.trajectory(), |p| {
            w.plan_proj.forward(p, &mut ws.mem);
            for (r, row) in ws.mem.chunks_exact_mut(d).enumerate() {
                w.plan_pos.add_row_into(r, d, row);
            }
            for (l, blk) in w.blocks.iter().enumerate() {
                blk.xk.forward(&ws.mem, &mut ws.xk[l]);
                blk.xv.forward(&ws.mem, &mut ws.xv[l]);
            }
            mean_rows_into(p, w.plan_proj.d_in, &mut ws.plan_mean);
            w.plan_pool.forward(&ws.plan_mean, &mut ws.plan_c);
        })?;
        ws.prepared_for = Some(plan.trajectory().id());
        Ok(())
    }

    fn estimate_velocity_into(&mut self, x: &Tensor, t: f32, plan: &LatentPlan, out: &mut Tensor) -> Result<()> {
        if self.ws.prepared_for != Some(plan.trajectory().id()) {
            self.prepare(plan)?;
        }
        let (w, ws) = (&self.w, &mut self.ws);
        let (d, l, p, heads) = (w.d(), w.seq_len, w.plan_len, w.heads);

        // global condition c = MLP_t(emb(t)) + pooled plan
        sinusoidal_embedding(t, &mut ws.temb);
        w.t1.forward(&ws.temb, &mut ws.c_tmp);
        silu_into(&ws.c_tmp, &mut ws.c);
        w.t2.forward(&ws.c, &mut ws.c_tmp);
        add_inplace(&mut ws.c_tmp, &ws.plan_c);
        silu_into(&ws.c_tmp, &mut ws.sc);

        host_read(x, |xs| w.in_proj.forward(xs, &mut ws.h))?;
        for (r, row) in ws.h.chunks_exact_mut(d).enumerate() {
            w.pos.add_row_into(r, d, row);
        }

        for (li, blk) in w.blocks.iter().enumerate() {
            blk.modulation.forward(&ws.sc, &mut ws.modv);
            let m = &ws.modv;
            let (sh1, sc1, g1) = (&m[0..d], &m[d..2 * d], &m[2 * d..3 * d]);
            let (sh2, sc2, g2) = (&m[3 * d..4 * d], &m[4 * d..5 * d], &m[5 * d..6 * d]);

            // self-attention over the L output positions
            layer_norm_rows_into(&ws.h, &mut ws.a, d);
            modulate_rows(&mut ws.a, sh1, sc1);
            blk.q.forward(&ws.a, &mut ws.q);
            blk.k.forward(&ws.a, &mut ws.k);
            blk.v.forward(&ws.a, &mut ws.v);
            attention(&ws.q, &ws.k, &ws.v, l, l, d, heads, &mut ws.att, &mut ws.scores);
            blk.o.forward(&ws.att, &mut ws.proj);
            add_gated_rows(&mut ws.h, g1, &ws.proj);

            // cross-attention to the latent plan (cached K/V)
            layer_norm_rows_into(&ws.h, &mut ws.a, d);
            blk.xq.forward(&ws.a, &mut ws.q);
            attention(&ws.q, &ws.xk[li], &ws.xv[li], l, p, d, heads, &mut ws.att, &mut ws.scores);
            blk.xo.forward_acc(&ws.att, &mut ws.h);

            // gated MLP
            layer_norm_rows_into(&ws.h, &mut ws.a, d);
            modulate_rows(&mut ws.a, sh2, sc2);
            blk.fc1.forward(&ws.a, &mut ws.mlp);
            gelu_inplace(&mut ws.mlp);
            blk.fc2.forward(&ws.mlp, &mut ws.proj);
            add_gated_rows(&mut ws.h, g2, &ws.proj);
        }

        w.final_mod.forward(&ws.sc, &mut ws.fmod);
        layer_norm_rows_into(&ws.h, &mut ws.a, d);
        modulate_rows(&mut ws.a, &ws.fmod[..d], &ws.fmod[d..]);
        let a = &ws.a;
        host_write(out, |o| {
            w.out_proj.forward(a, o);
            Ok(())
        })
    }
}

/// Graph-path estimator (device generic, allocating) — reference for parity tests and the
/// sampler on accelerators.
pub struct GraphVectorField<'a> {
    pub net: &'a VectorField,
    plan: Option<Tensor>,
}

impl<'a> GraphVectorField<'a> {
    pub fn new(net: &'a VectorField) -> Self {
        Self { net, plan: None }
    }
}

impl VectorFieldEstimator for GraphVectorField<'_> {
    fn prepare(&mut self, plan: &LatentPlan) -> Result<()> {
        self.plan = Some(plan.trajectory().unsqueeze(0)?);
        Ok(())
    }

    fn estimate_velocity_into(&mut self, x: &Tensor, t: f32, plan: &LatentPlan, out: &mut Tensor) -> Result<()> {
        let p = match &self.plan {
            Some(p) => p.clone(),
            None => plan.trajectory().unsqueeze(0)?,
        };
        let tt = Tensor::new(&[t], x.device())?;
        *out = self.net.forward(&x.unsqueeze(0)?, &tt, &p)?.squeeze(0)?.to_dtype(x.dtype())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Init;
    use candle_core::Device;

    /// Randomises every parameter (the zero-init adaLN would otherwise hide bugs).
    fn randomized(cfg: &EngineConfig) -> Result<VectorField> {
        let dev = Device::Cpu;
        let mut ps = ParamStore::new(&dev, 11);
        let vf = VectorField::new(&mut ps, cfg)?;
        let mut rs = ParamStore::new(&dev, 12);
        for (i, v) in ps.vars().iter().enumerate() {
            let r = rs.tensor(&format!("r{i}"), v.dims(), Init::Normal(0.2))?;
            v.set(&r)?;
        }
        Ok(vf)
    }

    #[test]
    fn packed_matches_graph() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EngineConfig::tiny(10, 8, 8);
        let vf = randomized(&cfg)?;
        let (l, dt, p, ds) = (cfg.flow.seq_len, cfg.flow.d_token, cfg.plan_len(), cfg.jepa.d_state);
        let x = Tensor::randn(0f32, 1.0, (l, dt), &dev)?;
        let plan = LatentPlan::new(Tensor::randn(0f32, 1.0, (p, ds), &dev)?, p, ds)?;

        let mut graph = GraphVectorField::new(&vf);
        let want = graph.estimate_velocity(&x, 0.37, &plan)?;

        let mut arena = Arena::new(&dev);
        let mut packed = vf.pack(DType::F32, &mut arena)?;
        let got = packed.estimate_velocity(&x, 0.37, &plan)?;
        let err = (&want - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "f32 packed vs graph: {err}");

        let mut packed16 = vf.pack(DType::BF16, &mut Arena::new(&dev))?;
        let got16 = packed16.estimate_velocity(&x, 0.37, &plan)?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
        let err16 = (&want - &got16)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err16 < 0.05 * scale.max(1.0), "bf16 packed vs graph: {err16} (scale {scale})");
        Ok(())
    }
}
