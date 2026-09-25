//! Dot products with f32 accumulation — the inner loop of every host kernel.
//!
//! `unsafe` is confined to this file: AVX2/FMA(/F16C) intrinsics behind runtime feature
//! detection, exposed only through safe functions that fall back to portable code.

use half::{bf16, f16};

#[inline]
fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let (ca, cb) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn dot_portable_conv<W: Copy>(w: &[W], x: &[f32], conv: impl Fn(W) -> f32) -> f32 {
    let n = w.len().min(x.len());
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        for i in 0..8 {
            acc[i] += conv(w[c * 8 + i]) * x[c * 8 + i];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += conv(w[i]) * x[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use half::{bf16, f16};
    use std::arch::x86_64::*;

    #[inline]
    pub fn has_avx2_fma() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[inline]
    pub fn has_f16c() -> bool {
        has_avx2_fma() && is_x86_feature_detected!("f16c")
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA. Reads stay within `min(a.len(), b.len())`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            s += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        s
    }

    /// bf16 → f32 is a 16-bit left shift of the raw bits.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA. `bf16` is `repr(transparent)` over `u16`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16(w: &[bf16], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let (pw, px) = (w.as_ptr() as *const u16, x.as_ptr());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let raw = _mm_loadu_si128(pw.add(i) as *const __m128i);
            let wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(wide), _mm256_loadu_ps(px.add(i)), acc);
            i += 8;
        }
        let mut s = hsum(acc);
        while i < n {
            s += f32::from_bits((*pw.add(i) as u32) << 16) * *px.add(i);
            i += 1;
        }
        s
    }

    /// # Safety
    /// The CPU must support AVX2, FMA and F16C. `f16` is `repr(transparent)` over `u16`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(w: &[f16], x: &[f32]) -> f32 {
        let n = w.len().min(x.len());
        let (pw, px) = (w.as_ptr() as *const u16, x.as_ptr());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let raw = _mm_loadu_si128(pw.add(i) as *const __m128i);
            acc = _mm256_fmadd_ps(_mm256_cvtph_ps(raw), _mm256_loadu_ps(px.add(i)), acc);
            i += 8;
        }
        let mut s = hsum(acc);
        while i < n {
            s += f16::from_bits(*pw.add(i)).to_f32() * *px.add(i);
            i += 1;
        }
        s
    }

    /// Four weight rows against one activation row: `x` is loaded once per 8 lanes and
    /// feeds four FMAs (register blocking).
    macro_rules! dot4_impl {
        ($name:ident, $t:ty, $feat:literal, |$p:ident, $i:ident| $load:expr, |$v:ident| $scalar:expr) => {
            /// # Safety
            /// The CPU must support the enabled features and every `w[j]` must hold at least
            /// `x.len()` elements.
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(w: [&[$t]; 4], x: &[f32]) -> [f32; 4] {
                let n = x.len();
                let px = x.as_ptr();
                let pw = [w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr(), w[3].as_ptr()];
                let (mut a0, mut a1, mut a2, mut a3) =
                    (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
                let mut $i = 0;
                while $i + 8 <= n {
                    let xv = _mm256_loadu_ps(px.add($i));
                    let $p = pw[0];
                    a0 = _mm256_fmadd_ps($load, xv, a0);
                    let $p = pw[1];
                    a1 = _mm256_fmadd_ps($load, xv, a1);
                    let $p = pw[2];
                    a2 = _mm256_fmadd_ps($load, xv, a2);
                    let $p = pw[3];
                    a3 = _mm256_fmadd_ps($load, xv, a3);
                    $i += 8;
                }
                let mut out = [hsum(a0), hsum(a1), hsum(a2), hsum(a3)];
                while $i < n {
                    let xi = *px.add($i);
                    for (o, p) in out.iter_mut().zip(pw) {
                        let $v = *p.add($i);
                        *o += $scalar * xi;
                    }
                    $i += 1;
                }
                out
            }
        };
    }

    dot4_impl!(dot4_f32, f32, "avx2,fma", |p, i| _mm256_loadu_ps(p.add(i)), |v| v);
    dot4_impl!(
        dot4_bf16,
        bf16,
        "avx2,fma",
        |p, i| _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(i) as *const __m128i)),
            16
        )),
        |v| v.to_f32()
    );
    dot4_impl!(
        dot4_f16,
        f16,
        "avx2,fma,f16c",
        |p, i| _mm256_cvtph_ps(_mm_loadu_si128(p.add(i) as *const __m128i)),
        |v| v.to_f32()
    );
}

/// `Σ a_i b_i` in f32.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        // SAFETY: feature presence checked at runtime just above.
        return unsafe { x86::dot_f32(a, b) };
    }
    dot_portable(a, b)
}

/// `Σ w_i x_i` with bf16 weights, f32 activations and f32 accumulation.
#[inline]
pub fn dot_bf16(w: &[bf16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        // SAFETY: feature presence checked at runtime just above.
        return unsafe { x86::dot_bf16(w, x) };
    }
    dot_portable_conv(w, x, |v| v.to_f32())
}

/// `Σ w_i x_i` with f16 weights, f32 activations and f32 accumulation.
#[inline]
pub fn dot_f16(w: &[f16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x86::has_f16c() {
        // SAFETY: feature presence checked at runtime just above.
        return unsafe { x86::dot_f16(w, x) };
    }
    dot_portable_conv(w, x, |v| v.to_f32())
}

#[inline]
fn check4<T>(w: &[&[T]; 4], x: &[f32]) {
    assert!(w.iter().all(|r| r.len() >= x.len()), "dot4: weight rows shorter than the activation");
}

/// Four dot products `[w_0·x, w_1·x, w_2·x, w_3·x]` sharing the loads of `x`.
#[inline]
pub fn dot4_f32(w: [&[f32]; 4], x: &[f32]) -> [f32; 4] {
    check4(&w, x);
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        // SAFETY: features checked at runtime; row lengths checked by `check4`.
        return unsafe { x86::dot4_f32(w, x) };
    }
    w.map(|r| dot_portable(r, x))
}

/// bf16 variant of [`dot4_f32`].
#[inline]
pub fn dot4_bf16(w: [&[bf16]; 4], x: &[f32]) -> [f32; 4] {
    check4(&w, x);
    #[cfg(target_arch = "x86_64")]
    if x86::has_avx2_fma() {
        // SAFETY: features checked at runtime; row lengths checked by `check4`.
        return unsafe { x86::dot4_bf16(w, x) };
    }
    w.map(|r| dot_portable_conv(r, x, |v| v.to_f32()))
}

/// f16 variant of [`dot4_f32`].
#[inline]
pub fn dot4_f16(w: [&[f16]; 4], x: &[f32]) -> [f32; 4] {
    check4(&w, x);
    #[cfg(target_arch = "x86_64")]
    if x86::has_f16c() {
        // SAFETY: features checked at runtime; row lengths checked by `check4`.
        return unsafe { x86::dot4_f16(w, x) };
    }
    w.map(|r| dot_portable_conv(r, x, |v| v.to_f32()))
}

/// Human-readable name of the SIMD path in use.
pub fn simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if x86::has_f16c() {
        return "avx2+fma+f16c";
    } else if x86::has_avx2_fma() {
        return "avx2+fma";
    }
    "portable"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        (0..n).map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 500.0 - 1.0).collect()
    }

    #[test]
    fn simd_matches_portable() {
        for n in [0, 1, 7, 8, 15, 16, 17, 33, 100, 257] {
            let (a, b) = (data(n, 1), data(n, 2));
            let r = dot_portable(&a, &b);
            assert!((dot_f32(&a, &b) - r).abs() < 1e-3, "f32 n={n}");
            let wb: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
            let rb = dot_portable_conv(&wb, &b, |v| v.to_f32());
            assert!((dot_bf16(&wb, &b) - rb).abs() < 1e-3, "bf16 n={n}");
            let wh: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
            let rh = dot_portable_conv(&wh, &b, |v| v.to_f32());
            assert!((dot_f16(&wh, &b) - rh).abs() < 1e-3, "f16 n={n}");

            let rows: Vec<Vec<f32>> = (0..4).map(|j| data(n, 10 + j)).collect();
            let want: Vec<f32> = rows.iter().map(|r| dot_portable(r, &b)).collect();
            let got = dot4_f32([&rows[0], &rows[1], &rows[2], &rows[3]], &b);
            let rows_b: Vec<Vec<bf16>> = rows.iter().map(|r| r.iter().map(|&v| bf16::from_f32(v)).collect()).collect();
            let got_b = dot4_bf16([&rows_b[0], &rows_b[1], &rows_b[2], &rows_b[3]], &b);
            let rows_h: Vec<Vec<f16>> = rows.iter().map(|r| r.iter().map(|&v| f16::from_f32(v)).collect()).collect();
            let got_h = dot4_f16([&rows_h[0], &rows_h[1], &rows_h[2], &rows_h[3]], &b);
            for j in 0..4 {
                assert!((got[j] - want[j]).abs() < 1e-3, "dot4 f32 n={n}");
                assert!((got_b[j] - want[j]).abs() < 2e-2 * (1.0 + want[j].abs()), "dot4 bf16 n={n}");
                assert!((got_h[j] - want[j]).abs() < 1e-2 * (1.0 + want[j].abs()), "dot4 f16 n={n}");
            }
        }
    }
}
