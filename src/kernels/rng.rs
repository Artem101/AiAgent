//! Deterministic, allocation-free random numbers (xoshiro256++ seeded through SplitMix64).
//!
//! Parallel consumers derive independent streams with [`Rng::stream`] from
//! `(seed, a, b)` — e.g. `(seed, mppi_iteration, sample_index)` — so results do not
//! depend on how rayon schedules the work.

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self { s: [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)] }
    }

    /// Independent stream keyed by `(seed, a, b)`.
    pub fn stream(seed: u64, a: u64, b: u64) -> Self {
        let mut sm = seed ^ 0xA076_1D64_78BD_642F;
        let x = splitmix64(&mut sm) ^ a.wrapping_mul(0xE703_7ED1_A0B4_28DB);
        let mut sm2 = x;
        let y = splitmix64(&mut sm2) ^ b.wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
        Self::new(y)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }

    /// Two independent standard normals (Box–Muller).
    #[inline]
    pub fn normal_pair(&mut self) -> (f32, f32) {
        let u1 = 1.0 - self.uniform(); // (0, 1]
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        ((r * th.cos()) as f32, (r * th.sin()) as f32)
    }

    #[inline]
    pub fn normal(&mut self) -> f32 {
        self.normal_pair().0
    }

    /// Fills `out` with `N(0, std²)` samples.
    pub fn fill_normal(&mut self, out: &mut [f32], std: f32) {
        let mut chunks = out.chunks_exact_mut(2);
        for c in &mut chunks {
            let (a, b) = self.normal_pair();
            c[0] = a * std;
            c[1] = b * std;
        }
        if let [last] = chunks.into_remainder() {
            *last = self.normal() * std;
        }
    }

    pub fn fill_uniform(&mut self, out: &mut [f32], lo: f32, hi: f32) {
        for v in out {
            *v = lo + (hi - lo) * self.uniform() as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_deterministic_and_distinct() {
        let a: Vec<u64> = (0..4).map(|_| Rng::stream(1, 2, 3).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        assert_ne!(Rng::stream(1, 2, 3).next_u64(), Rng::stream(1, 2, 4).next_u64());
    }

    #[test]
    fn normal_moments() {
        let mut r = Rng::new(42);
        let mut v = vec![0f32; 100_000];
        r.fill_normal(&mut v, 1.0);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
