//! In-place mutation of candle tensors without allocation.
//!
//! candle tensors are logically immutable, but candle exposes `InplaceOp{1,2,3}` custom ops
//! that receive the raw storage under a write lock. For host `f32` tensors these helpers
//! run our kernels directly on that storage — no new tensor, no allocation. For any other
//! device/dtype they fall back to graph ops and rebind the `&mut Tensor` (allocating but
//! device-generic), so callers are correct everywhere and allocation-free on the host.
//!
//! Aliasing rule: the mutated tensor must not share storage with an input (the storage
//! write lock would dead-lock). Arena buffers are always separate allocations.

use std::cell::RefCell;

use candle_core::{bail, CpuStorage, DType, InplaceOp1, InplaceOp2, InplaceOp3, Layout, Result, Storage, Tensor};

use super::rng::Rng;

/// `true` when `t` lives on the host as contiguous f32 — the in-place fast path.
#[inline]
pub fn is_host_f32(t: &Tensor) -> bool {
    t.device().is_cpu() && t.dtype() == DType::F32 && t.is_contiguous()
}

#[inline]
fn f32_mut<'a>(s: &'a mut CpuStorage, l: &Layout, op: &'static str) -> Result<&'a mut [f32]> {
    let Some((a, b)) = l.contiguous_offsets() else { bail!("{op}: tensor must be contiguous") };
    match s {
        CpuStorage::F32(v) => Ok(&mut v[a..b]),
        _ => bail!("{op}: expected f32 storage"),
    }
}

#[inline]
fn f32_ref<'a>(s: &'a CpuStorage, l: &Layout, op: &'static str) -> Result<&'a [f32]> {
    let Some((a, b)) = l.contiguous_offsets() else { bail!("{op}: tensor must be contiguous") };
    match s {
        CpuStorage::F32(v) => Ok(&v[a..b]),
        _ => bail!("{op}: expected f32 storage"),
    }
}

#[inline]
fn same_len(op: &'static str, a: usize, b: usize) -> Result<()> {
    if a != b {
        bail!("{op}: length mismatch ({a} vs {b})")
    }
    Ok(())
}

/// Reads a host f32 tensor as a slice.
pub fn host_read<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<R> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cpu(cpu) => Ok(f(f32_ref(cpu, layout, "host_read")?)),
        _ => bail!("host_read: tensor is not on the host"),
    }
}

struct HostWrite<F>(RefCell<Option<F>>);

impl<F: FnOnce(&mut [f32]) -> Result<()>> InplaceOp1 for HostWrite<F> {
    fn name(&self) -> &'static str {
        "host-write"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> {
        let data = f32_mut(s, l, "host-write")?;
        match self.0.borrow_mut().take() {
            Some(f) => f(data),
            None => bail!("host-write: closure already consumed"),
        }
    }
}

/// Runs `f` on the mutable storage of a host f32 tensor.
pub fn host_write(t: &Tensor, f: impl FnOnce(&mut [f32]) -> Result<()>) -> Result<()> {
    if !is_host_f32(t) {
        bail!("host_write: tensor must be a contiguous f32 host tensor")
    }
    t.inplace_op1(&HostWrite(RefCell::new(Some(f))))
}

/// Copies a host slice into a host f32 tensor.
pub fn copy_from_slice(t: &Tensor, src: &[f32]) -> Result<()> {
    host_write(t, |dst| {
        same_len("copy_from_slice", dst.len(), src.len())?;
        dst.copy_from_slice(src);
        Ok(())
    })
}

struct Axpy(f32);
impl InplaceOp2 for Axpy {
    fn name(&self) -> &'static str {
        "axpy"
    }
    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<()> {
        let y = f32_mut(s1, l1, "axpy")?;
        let x = f32_ref(s2, l2, "axpy")?;
        same_len("axpy", y.len(), x.len())?;
        super::axpy(y, self.0, x);
        Ok(())
    }
}

/// `y ← y + α x`.
pub fn axpy_(y: &mut Tensor, alpha: f32, x: &Tensor) -> Result<()> {
    if is_host_f32(y) && is_host_f32(x) {
        y.inplace_op2(x, &Axpy(alpha))
    } else {
        *y = (&*y + (x * alpha as f64)?)?;
        Ok(())
    }
}

struct LinComb {
    a: f32,
    b: f32,
    accumulate: bool,
}
impl InplaceOp3 for LinComb {
    fn name(&self) -> &'static str {
        "lincomb"
    }
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        let y = f32_mut(s1, l1, "lincomb")?;
        let (x1, x2) = (f32_ref(s2, l2, "lincomb")?, f32_ref(s3, l3, "lincomb")?);
        same_len("lincomb", y.len(), x1.len())?;
        same_len("lincomb", y.len(), x2.len())?;
        if self.accumulate {
            super::acc_lincomb(y, self.a, x1, self.b, x2)
        } else {
            super::lincomb_into(y, self.a, x1, self.b, x2)
        }
        Ok(())
    }
}

/// `y ← a x1 + b x2`.
pub fn lincomb_(y: &mut Tensor, a: f32, x1: &Tensor, b: f32, x2: &Tensor) -> Result<()> {
    if is_host_f32(y) && is_host_f32(x1) && is_host_f32(x2) {
        y.inplace_op3(x1, x2, &LinComb { a, b, accumulate: false })
    } else {
        *y = ((x1 * a as f64)? + (x2 * b as f64)?)?;
        Ok(())
    }
}

/// `y ← y + a x1 + b x2`.
pub fn acc_lincomb_(y: &mut Tensor, a: f32, x1: &Tensor, b: f32, x2: &Tensor) -> Result<()> {
    if is_host_f32(y) && is_host_f32(x1) && is_host_f32(x2) {
        y.inplace_op3(x1, x2, &LinComb { a, b, accumulate: true })
    } else {
        *y = ((&*y + (x1 * a as f64)?)? + (x2 * b as f64)?)?;
        Ok(())
    }
}

/// Fused TTT rank-1 step `W ← W − η (W k − v) ⊗ k` (see [`super::rank1_update`]).
pub(crate) struct Rank1Update(pub f32);
impl InplaceOp3 for Rank1Update {
    fn name(&self) -> &'static str {
        "ttt-rank1-update"
    }
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        let w = f32_mut(s1, l1, "ttt-rank1-update")?;
        let (k, v) = (f32_ref(s2, l2, "ttt-rank1-update")?, f32_ref(s3, l3, "ttt-rank1-update")?);
        same_len("ttt-rank1-update", w.len(), k.len() * v.len())?;
        super::rank1_update(w, k, v, self.0);
        Ok(())
    }
}

/// `out ← W q` for a square f32 state matrix.
pub(crate) struct MatVecInto;
impl InplaceOp3 for MatVecInto {
    fn name(&self) -> &'static str {
        "matvec-into"
    }
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        let out = f32_mut(s1, l1, "matvec-into")?;
        let (w, q) = (f32_ref(s2, l2, "matvec-into")?, f32_ref(s3, l3, "matvec-into")?);
        same_len("matvec-into", w.len(), out.len() * q.len())?;
        super::matvec_f32(w, q, out);
        Ok(())
    }
}

struct FillNormal<'r> {
    rng: RefCell<&'r mut Rng>,
    std: f32,
}
impl InplaceOp1 for FillNormal<'_> {
    fn name(&self) -> &'static str {
        "fill-normal"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> {
        let data = f32_mut(s, l, "fill-normal")?;
        self.rng.borrow_mut().fill_normal(data, self.std);
        Ok(())
    }
}

/// Fills `t` with `N(0, std²)` from a deterministic stream.
pub fn fill_normal_(t: &mut Tensor, rng: &mut Rng, std: f32) -> Result<()> {
    if is_host_f32(t) {
        t.inplace_op1(&FillNormal { rng: RefCell::new(rng), std })
    } else {
        let mut v = vec![0f32; t.elem_count()];
        rng.fill_normal(&mut v, std);
        *t = Tensor::from_vec(v, t.shape(), t.device())?.to_dtype(t.dtype())?;
        Ok(())
    }
}

struct Fill(f32);
impl InplaceOp1 for Fill {
    fn name(&self) -> &'static str {
        "fill"
    }
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()> {
        f32_mut(s, l, "fill")?.fill(self.0);
        Ok(())
    }
}

/// Sets every element of `t` to `value`.
pub fn fill_(t: &mut Tensor, value: f32) -> Result<()> {
    if is_host_f32(t) {
        t.inplace_op1(&Fill(value))
    } else {
        *t = (t.zeros_like()? + value as f64)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn inplace_ops_mutate_storage() -> Result<()> {
        let dev = Device::Cpu;
        let mut y = Tensor::new(&[1f32, 2., 3.], &dev)?;
        let x = Tensor::new(&[1f32, 1., 1.], &dev)?;
        let before = y.id();
        axpy_(&mut y, 2.0, &x)?;
        assert_eq!(y.id(), before, "fast path must not rebind the tensor");
        assert_eq!(y.to_vec1::<f32>()?, vec![3., 4., 5.]);
        lincomb_(&mut y, 0.5, &x, 2.0, &x)?;
        assert_eq!(y.to_vec1::<f32>()?, vec![2.5; 3]);
        acc_lincomb_(&mut y, 1.0, &x, -1.0, &x)?;
        assert_eq!(y.to_vec1::<f32>()?, vec![2.5; 3]);
        copy_from_slice(&y, &[9., 8., 7.])?;
        assert_eq!(host_read(&y, |s| s.to_vec())?, vec![9., 8., 7.]);
        fill_(&mut y, 0.0)?;
        assert_eq!(y.to_vec1::<f32>()?, vec![0.; 3]);
        Ok(())
    }

    #[test]
    fn fallback_for_non_f32() -> Result<()> {
        let dev = Device::Cpu;
        let mut y = Tensor::new(&[1f64, 2.], &dev)?;
        let x = Tensor::new(&[1f64, 1.], &dev)?;
        axpy_(&mut y, 1.0, &x)?;
        assert_eq!(y.to_vec1::<f64>()?, vec![2., 3.]);
        Ok(())
    }
}
