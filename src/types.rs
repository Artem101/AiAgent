//! Semantic newtypes for the three representation spaces of the pipeline.
//!
//! Each wrapper validates rank and width at construction, so a prompt state can never be
//! fed where a latent plan or a flow state is expected (and vice versa). Fields are private;
//! the inner tensor is only reachable through read-only accessors.

use candle_core::{bail, Result, Tensor};

macro_rules! vector_newtype {
    ($(#[$m:meta])* $name:ident, $what:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone)]
        pub struct $name(Tensor);

        impl $name {
            /// Wraps a rank-1 tensor of width `dim`.
            pub fn new(t: Tensor, dim: usize) -> Result<Self> {
                match t.dims() {
                    [d] if *d == dim => Ok(Self(t)),
                    dims => bail!(concat!($what, ": expected shape [{}], got {:?}"), dim, dims),
                }
            }
            pub fn tensor(&self) -> &Tensor {
                &self.0
            }
            pub fn dim(&self) -> usize {
                self.0.dims()[0]
            }
            pub fn into_inner(self) -> Tensor {
                self.0
            }
        }
    };
}

vector_newtype!(
    /// `S_prompt ∈ ℝ^{d_ctx}` — readout of the final fast weights `W_fast^{(N)}`.
    PromptState,
    "PromptState"
);

vector_newtype!(
    /// A single latent state `s ∈ ℝ^{d_s}` (initial thought `s_0`, or the energy target).
    LatentState,
    "LatentState"
);

/// `S_plan = [s_0, s_1, …, s_H] ∈ ℝ^{(H+1)×d_s}` — the latent reasoning trajectory.
#[derive(Debug, Clone)]
pub struct LatentPlan {
    trajectory: Tensor,
}

impl LatentPlan {
    pub fn new(trajectory: Tensor, plan_len: usize, d_state: usize) -> Result<Self> {
        match trajectory.dims() {
            [p, d] if *p == plan_len && *d == d_state => Ok(Self { trajectory }),
            dims => bail!("LatentPlan: expected shape [{plan_len}, {d_state}], got {dims:?}"),
        }
    }
    /// Shape `[H + 1, d_s]`.
    pub fn trajectory(&self) -> &Tensor {
        &self.trajectory
    }
    pub fn len(&self) -> usize {
        self.trajectory.dims()[0]
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn d_state(&self) -> usize {
        self.trajectory.dims()[1]
    }
    pub fn initial_state(&self) -> Result<LatentState> {
        LatentState::new(self.trajectory.get(0)?, self.d_state())
    }
    pub fn terminal_state(&self) -> Result<LatentState> {
        LatentState::new(self.trajectory.get(self.len() - 1)?, self.d_state())
    }
}

/// `X_flow ∈ ℝ^{L×d_token}` — a point on the flow-matching path (`X_1` once integrated).
#[derive(Debug, Clone)]
pub struct FlowState(Tensor);

impl FlowState {
    pub fn new(t: Tensor, seq_len: usize, d_token: usize) -> Result<Self> {
        match t.dims() {
            [l, d] if *l == seq_len && *d == d_token => Ok(Self(t)),
            dims => bail!("FlowState: expected shape [{seq_len}, {d_token}], got {dims:?}"),
        }
    }
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }
    pub fn seq_len(&self) -> usize {
        self.0.dims()[0]
    }
    pub fn into_inner(self) -> Tensor {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn shapes_are_validated() -> Result<()> {
        let dev = Device::Cpu;
        assert!(PromptState::new(Tensor::zeros(8, candle_core::DType::F32, &dev)?, 8).is_ok());
        assert!(PromptState::new(Tensor::zeros(8, candle_core::DType::F32, &dev)?, 4).is_err());
        assert!(LatentPlan::new(Tensor::zeros((5, 4), candle_core::DType::F32, &dev)?, 5, 4).is_ok());
        assert!(LatentPlan::new(Tensor::zeros((4, 5), candle_core::DType::F32, &dev)?, 5, 4).is_err());
        assert!(FlowState::new(Tensor::zeros((3, 2, 1), candle_core::DType::F32, &dev)?, 3, 2).is_err());
        Ok(())
    }
}
