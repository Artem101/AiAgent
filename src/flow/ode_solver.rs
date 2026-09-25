//! Non-autoregressive ODE sampling: `dX = v_θ(X, t | S_plan) dt`, `t: 0 → 1`.
//!
//! All `L` positions move together; the cost is `K` (Euler) or `2K` (Midpoint/Heun)
//! vector-field evaluations regardless of `L` — compute-bound, no KV cache.
//!
//! The integration loop allocates nothing on the host: `X`, the intermediate state and both
//! velocity slots live in [`SamplerBuffers`] (created once from the arena), the estimator
//! writes into `&mut Tensor`, and the updates are in-place axpy kernels.

use candle_core::{Device, Result, Shape, Tensor};

use crate::arena::Arena;
use crate::kernels::inplace::{acc_lincomb_, axpy_, fill_normal_, lincomb_};
use crate::kernels::rng::Rng;
use crate::types::{FlowState, LatentPlan};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// 1st order, 1 evaluation per step.
    Euler,
    /// 2nd order (RK2 midpoint), 2 evaluations per step.
    Midpoint,
    /// 2nd order (explicit trapezoid / Heun), 2 evaluations per step.
    Heun,
}

impl SolverKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "euler" => Ok(Self::Euler),
            "midpoint" => Ok(Self::Midpoint),
            "heun" => Ok(Self::Heun),
            other => candle_core::bail!("unknown solver '{other}' (euler | midpoint | heun)"),
        }
    }
    pub fn evals_per_step(&self) -> usize {
        match self {
            Self::Euler => 1,
            Self::Midpoint | Self::Heun => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ODESolverConfig {
    pub steps: usize,
    /// σ_min of the conditional OT path `X_t = (1 − (1 − σ_min) t) X_0 + t X_1`.
    pub sigma_min: f32,
    pub solver: SolverKind,
}

/// A learned (or analytic) velocity field `v(X, t | plan)`.
pub trait VectorFieldEstimator {
    /// Called once per trajectory before integrating, e.g. to project the plan into
    /// cross-attention keys/values that are constant over `t`.
    fn prepare(&mut self, _plan: &LatentPlan) -> Result<()> {
        Ok(())
    }

    /// Writes `v(x, t | plan)` into `out` (same shape as `x`). Host implementations must not
    /// allocate.
    fn estimate_velocity_into(&mut self, x: &Tensor, t: f32, plan: &LatentPlan, out: &mut Tensor) -> Result<()>;

    /// Allocating convenience wrapper.
    fn estimate_velocity(&mut self, x: &Tensor, t: f32, plan: &LatentPlan) -> Result<Tensor> {
        let mut out = x.zeros_like()?;
        self.estimate_velocity_into(x, t, plan, &mut out)?;
        Ok(out)
    }
}

/// Integrator state, allocated once.
#[derive(Debug)]
pub struct SamplerBuffers {
    /// Current `X_t` (holds `X_1` after integration).
    pub x: Tensor,
    x_tmp: Tensor,
    v1: Tensor,
    v2: Tensor,
}

impl SamplerBuffers {
    pub fn new(arena: &mut Arena, seq_len: usize, d_token: usize) -> Result<Self> {
        Ok(Self {
            x: arena.tensor((seq_len, d_token))?,
            x_tmp: arena.tensor((seq_len, d_token))?,
            v1: arena.tensor((seq_len, d_token))?,
            v2: arena.tensor((seq_len, d_token))?,
        })
    }

    pub fn from_shape(shape: &Shape, device: &Device) -> Result<Self> {
        let z = || Tensor::zeros(shape, candle_core::DType::F32, device);
        Ok(Self { x: z()?, x_tmp: z()?, v1: z()?, v2: z()? })
    }

    /// `X_1` as a validated flow state (shares the buffer's storage).
    pub fn flow_state(&self) -> Result<FlowState> {
        let (l, d) = self.x.dims2()?;
        FlowState::new(self.x.clone(), l, d)
    }
}

pub struct FlowMatchingSampler<'a> {
    pub estimator: &'a mut dyn VectorFieldEstimator,
    pub config: ODESolverConfig,
}

impl<'a> FlowMatchingSampler<'a> {
    pub fn new(estimator: &'a mut dyn VectorFieldEstimator, config: ODESolverConfig) -> Self {
        Self { estimator, config }
    }

    /// Integrates `bufs.x` from `t = 0` to `t = 1` in place.
    pub fn integrate(&mut self, plan: &LatentPlan, bufs: &mut SamplerBuffers) -> Result<()> {
        self.estimator.prepare(plan)?;
        let steps = self.config.steps;
        let dt = 1.0 / steps as f32;
        let SamplerBuffers { x, x_tmp, v1, v2 } = bufs;
        for step in 0..steps {
            let t = step as f32 * dt;
            match self.config.solver {
                SolverKind::Euler => {
                    self.estimator.estimate_velocity_into(x, t, plan, v1)?;
                    axpy_(x, dt, v1)?;
                }
                SolverKind::Midpoint => {
                    self.estimator.estimate_velocity_into(x, t, plan, v1)?;
                    lincomb_(x_tmp, 1.0, x, 0.5 * dt, v1)?;
                    self.estimator.estimate_velocity_into(x_tmp, t + 0.5 * dt, plan, v2)?;
                    axpy_(x, dt, v2)?;
                }
                SolverKind::Heun => {
                    // predictor: velocity at the start of the step
                    self.estimator.estimate_velocity_into(x, t, plan, v1)?;
                    lincomb_(x_tmp, 1.0, x, dt, v1)?;
                    // corrector: velocity at the end of the step; x += dt · (v1 + v2) / 2
                    self.estimator.estimate_velocity_into(x_tmp, t + dt, plan, v2)?;
                    acc_lincomb_(x, 0.5 * dt, v1, 0.5 * dt, v2)?;
                }
            }
        }
        Ok(())
    }

    /// Draws `X_0 ~ N(0, I)` into `bufs.x` and integrates to `X_1`.
    pub fn sample_into(&mut self, plan: &LatentPlan, bufs: &mut SamplerBuffers, rng: &mut Rng) -> Result<()> {
        fill_normal_(&mut bufs.x, rng, 1.0)?;
        self.integrate(plan, bufs)
    }

    /// Allocating convenience: fresh buffers of `shape` on `device`.
    pub fn sample(&mut self, shape: Shape, plan: &LatentPlan, device: &Device, seed: u64) -> Result<FlowState> {
        let mut bufs = SamplerBuffers::from_shape(&shape, device)?;
        self.sample_into(plan, &mut bufs, &mut Rng::new(seed))?;
        bufs.flow_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::inplace::{host_read, host_write};

    /// Analytic field `v(x, t) = −x`: the exact solution at `t = 1` is `x0 · e^{−1}`, which
    /// lets us check each solver's accuracy (Euler O(dt), Midpoint/Heun O(dt²)).
    struct Decay;
    impl VectorFieldEstimator for Decay {
        fn estimate_velocity_into(&mut self, x: &Tensor, _t: f32, _p: &LatentPlan, out: &mut Tensor) -> Result<()> {
            host_read(x, |xs| {
                host_write(out, |o| {
                    o.iter_mut().zip(xs).for_each(|(o, &x)| *o = -x);
                    Ok(())
                })
            })?
        }
    }

    #[test]
    fn solvers_converge_with_expected_order() -> Result<()> {
        let dev = Device::Cpu;
        let plan = LatentPlan::new(Tensor::zeros((2, 3), candle_core::DType::F32, &dev)?, 2, 3)?;
        let exact = (-1f32).exp();
        for (solver, tol) in [(SolverKind::Euler, 2e-2), (SolverKind::Midpoint, 1e-3), (SolverKind::Heun, 1e-3)] {
            let mut est = Decay;
            let mut s = FlowMatchingSampler::new(&mut est, ODESolverConfig { steps: 16, sigma_min: 0.0, solver });
            let mut bufs = SamplerBuffers::from_shape(&Shape::from((1, 1)), &dev)?;
            crate::kernels::inplace::copy_from_slice(&bufs.x, &[1.0])?;
            s.integrate(&plan, &mut bufs)?;
            let got = bufs.x.flatten_all()?.to_vec1::<f32>()?[0];
            assert!((got - exact).abs() < tol, "{solver:?}: {got} vs {exact}");
        }
        Ok(())
    }
}
