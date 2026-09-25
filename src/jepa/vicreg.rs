//! VICReg criterion (Bardes et al., 2022) — keeps the latent space from collapsing.
//!
//! ```text
//!   invariance  s(Z, Z') = MSE(Z, Z')
//!   variance    v(Z) = 1/d Σ_j max(0, γ − sqrt(Var(Z_{:,j}) + ε))
//!   covariance  c(Z) = 1/d Σ_{i≠j} C(Z)_{ij}²,   C(Z) = (Z − Z̄)ᵀ(Z − Z̄) / (N − 1)
//! ```

use candle_core::{bail, DType, Result, Tensor};

use crate::config::VicRegConfig;

pub fn invariance(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    crate::nn::mse(pred, target)
}

/// Per-dimension unbiased variance of `z: [N, d]` → `[d]`.
fn column_variance(z: &Tensor) -> Result<(Tensor, Tensor)> {
    let n = z.dims2()?.0;
    if n < 2 {
        bail!("VICReg needs a batch of at least 2 embeddings")
    }
    let zc = z.broadcast_sub(&z.mean_keepdim(0)?)?;
    let var = (zc.sqr()?.sum(0)? / (n - 1) as f64)?;
    Ok((zc, var))
}

pub fn variance(z: &Tensor, gamma: f64, eps: f64) -> Result<Tensor> {
    let z = z.to_dtype(DType::F32)?;
    let (_, var) = column_variance(&z)?;
    let std = (var + eps)?.sqrt()?;
    (std.neg()? + gamma)?.relu()?.mean_all()
}

pub fn covariance(z: &Tensor) -> Result<Tensor> {
    let z = z.to_dtype(DType::F32)?;
    let (n, d) = z.dims2()?;
    let (zc, var) = column_variance(&z)?;
    let c = (zc.t()?.matmul(&zc)? / (n - 1) as f64)?;
    // Σ_{i≠j} C_ij² = Σ C² − Σ diag(C)² and diag(C) is the column variance.
    (c.sqr()?.sum_all()? - var.sqr()?.sum_all()?)? / d as f64
}

/// Weighted VICReg terms.
pub struct VicRegLoss {
    pub invariance: Tensor,
    pub variance: Tensor,
    pub covariance: Tensor,
}

impl VicRegLoss {
    pub fn compute(pred: &Tensor, target: &Tensor, cfg: &VicRegConfig) -> Result<Self> {
        Ok(Self {
            invariance: invariance(pred, target)?,
            variance: variance(pred, cfg.gamma, cfg.eps)?,
            covariance: covariance(pred)?,
        })
    }

    pub fn weighted(&self, cfg: &VicRegConfig) -> Result<Tensor> {
        ((&self.invariance * cfg.inv_weight)? + (&self.variance * cfg.var_weight)?)?
            + (&self.covariance * cfg.cov_weight)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn collapsed_embeddings_are_penalised() -> Result<()> {
        let dev = Device::Cpu;
        let collapsed = Tensor::ones((64, 8), DType::F32, &dev)?;
        let spread = Tensor::randn(0f32, 1.0, (4096, 8), &dev)?;
        assert!(variance(&collapsed, 1.0, 1e-4)?.to_scalar::<f32>()? > 0.9);
        assert!(variance(&spread, 1.0, 1e-4)?.to_scalar::<f32>()? < 0.05);
        // perfectly correlated dimensions → large covariance penalty
        let col = Tensor::randn(0f32, 1.0, (512, 1), &dev)?;
        let correlated = col.repeat((1, 8))?;
        assert!(covariance(&correlated)?.to_scalar::<f32>()? > 5.0);
        assert!(covariance(&spread)?.to_scalar::<f32>()? < 0.05);
        Ok(())
    }
}
