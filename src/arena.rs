//! Pre-allocated buffer pool.
//!
//! Every buffer the inference graph touches is created here once, when the graph is built.
//! Host scratch space is handed out as `Box<[f32]>` — a fixed-size allocation that cannot
//! grow, so a hot loop that only receives boxed slices provably cannot reallocate. Tensor
//! buffers (states that cross module boundaries: `W_fast`, plans, flow states, velocities)
//! are fresh candle tensors, each with its own storage, and are later mutated in place via
//! [`crate::kernels::inplace`].

use candle_core::{DType, Device, Result, Shape, Tensor};

/// Hands out pre-allocated buffers and keeps count of them (see the module docs).
#[derive(Debug)]
pub struct Arena {
    device: Device,
    bytes: usize,
    buffers: usize,
}

impl Arena {
    pub fn new(device: &Device) -> Self {
        Self { device: device.clone(), bytes: 0, buffers: 0 }
    }

    /// Device on which [`Arena::tensor`] allocates.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A zero-initialised f32 tensor with its own storage.
    pub fn tensor<S: Into<Shape>>(&mut self, shape: S) -> Result<Tensor> {
        let t = Tensor::zeros(shape, DType::F32, &self.device)?;
        self.bytes += t.elem_count() * 4;
        self.buffers += 1;
        Ok(t)
    }

    /// A zero-initialised fixed-size host scratch buffer.
    pub fn host(&mut self, len: usize) -> Box<[f32]> {
        self.bytes += len * 4;
        self.buffers += 1;
        vec![0f32; len].into_boxed_slice()
    }

    /// A fixed-size host buffer of token ids.
    pub fn host_u32(&mut self, len: usize) -> Box<[u32]> {
        self.bytes += len * 4;
        self.buffers += 1;
        vec![0u32; len].into_boxed_slice()
    }

    /// Total bytes handed out so far.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Number of buffers handed out so far.
    pub fn buffers(&self) -> usize {
        self.buffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_and_distinct_storage() -> Result<()> {
        let mut a = Arena::new(&Device::Cpu);
        let t1 = a.tensor((2, 3))?;
        let t2 = a.tensor((2, 3))?;
        let h = a.host(10);
        assert_eq!(a.bytes(), (6 + 6 + 10) * 4);
        assert_eq!(a.buffers(), 3);
        assert_eq!(h.len(), 10);
        // distinct storages: mutating one must not affect the other
        crate::kernels::inplace::copy_from_slice(&t1, &[1.0; 6])?;
        assert_eq!(t2.sum_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}
