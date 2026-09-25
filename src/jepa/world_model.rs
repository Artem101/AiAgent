//! Transition predictor `P_φ(s_t, a_t) → ŝ_{t+1} = s_t + MLP([s_t ; a_t])`.

use candle_core::{DType, Result, Tensor, D};

use crate::kernels::PackedMlp;
use crate::nn::{Mlp, ParamStore};

#[derive(Debug, Clone)]
pub struct WorldModel {
    pub mlp: Mlp,
    pub d_state: usize,
    pub d_action: usize,
}

impl WorldModel {
    pub fn new(ps: &mut ParamStore, name: &str, d_state: usize, d_action: usize, d_hidden: usize) -> Result<Self> {
        Ok(Self { mlp: ps.mlp(name, d_state + d_action, d_hidden, d_state)?, d_state, d_action })
    }

    /// One transition; `s: [.., d_s]`, `a: [.., d_a]`.
    pub fn forward(&self, s: &Tensor, a: &Tensor) -> Result<Tensor> {
        let a = a.to_dtype(s.dtype())?;
        s + self.mlp.forward(&Tensor::cat(&[s, &a], D::Minus1)?)?
    }

    /// `s0: [B, d_s]`, `actions: [B, H, d_a]` → trajectory `[B, H + 1, d_s]`.
    pub fn rollout(&self, s0: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let h = actions.dims3()?.1;
        let mut states = Vec::with_capacity(h + 1);
        states.push(s0.clone());
        for t in 0..h {
            let next = self.forward(&states[t], &actions.narrow(1, t, 1)?.squeeze(1)?)?;
            states.push(next);
        }
        Tensor::stack(&states, 1)
    }

    pub fn pack(&self, dtype: DType) -> Result<PackedWorldModel> {
        Ok(PackedWorldModel { mlp: self.mlp.pack(dtype)?, d_state: self.d_state, d_action: self.d_action })
    }

    pub fn from_packed(p: &PackedWorldModel, device: &candle_core::Device) -> Result<Self> {
        Ok(Self { mlp: Mlp::from_packed(&p.mlp, device)?, d_state: p.d_state, d_action: p.d_action })
    }
}

/// Host-kernel world model used inside the MPPI rollouts.
#[derive(Debug, Clone)]
pub struct PackedWorldModel {
    pub mlp: PackedMlp,
    pub d_state: usize,
    pub d_action: usize,
}

impl PackedWorldModel {
    pub fn d_hidden(&self) -> usize {
        self.mlp.l1.d_out
    }

    /// `s_next = s + MLP([s ; a])`. `cat` (`d_s + d_a`) and `hidden` (`d_hidden`) are scratch.
    #[inline]
    pub fn step(&self, s: &[f32], a: &[f32], cat: &mut [f32], hidden: &mut [f32], s_next: &mut [f32]) {
        let ds = self.d_state;
        cat[..ds].copy_from_slice(s);
        cat[ds..ds + self.d_action].copy_from_slice(a);
        s_next[..ds].copy_from_slice(s);
        self.mlp.forward_acc(&cat[..ds + self.d_action], hidden, &mut s_next[..ds]);
    }

    /// Rolls `actions: [H, d_a]` out from `s0` into `states: [H + 1, d_s]`.
    pub fn rollout(&self, s0: &[f32], actions: &[f32], states: &mut [f32], cat: &mut [f32], hidden: &mut [f32]) {
        let (ds, da) = (self.d_state, self.d_action);
        states[..ds].copy_from_slice(s0);
        for (t, a) in actions.chunks_exact(da).enumerate() {
            let (done, rest) = states.split_at_mut((t + 1) * ds);
            self.step(&done[t * ds..], a, cat, hidden, &mut rest[..ds]);
        }
    }

    pub fn bytes(&self) -> usize {
        self.mlp.bytes()
    }
}
