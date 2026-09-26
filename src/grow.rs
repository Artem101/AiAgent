//! Growing a trained unified model into a wider one without losing what it learned
//! (docs/scaling.md, section 5).
//!
//! * **Speech decoder, width × 2** by duplication. Its LayerNorm has no parameters, so a
//!   residual stream `x` duplicated into `[x ; x]` keeps its mean and variance:
//!   `LN([x ; x]) = [LN(x) ; LN(x)]`. A layer that reads the stream becomes `[W/2, W/2]`
//!   (same pre-activation), a layer that writes it duplicates its outputs:
//!   `W' = [[W/2, W/2]; [W/2, W/2]]`. Attention goes from `h` heads to `2h` heads of the same
//!   width — the originals and their copies — so every head attends exactly as before; the MLP's
//!   hidden layer is duplicated (ReLU keeps duplicates). The tied embeddings `[E ; E]` double
//!   the logits, which the logit scale (½) undoes. The pointer's queries read the stream through
//!   `[W/2, W/2]`.
//! * **Latent planner, widths × 2** by zero-padding: new rows of every MLP layer and new columns
//!   for the new inputs are zero, so the new state and action dimensions stay 0 and the old ones
//!   are computed exactly as before (`GELU(0) = 0`, the world model is residual).
//! * **TTT encoder**: unchanged, except the vocabulary (the appended `THINK`, `LOOKUP` start
//!   as the mean of `CALC` and `ANSWER`) and the positions when the observation format changes
//!   (the page keeps its positions at the start, the tail — conversation, question, tool
//!   results — keeps its positions at the end; answer positions continue).
//!
//! With the same format and vocabulary the grown model computes the same outputs as the
//! original (checked by a test); a small noise then breaks the symmetry between copies.

use candle_core::{bail, Result, Tensor};

use crate::browser::obs::Layout;
use crate::kernels::rng::Rng;
use crate::text::{ANSWER, CALC};
use crate::unified::UnifiedModel;

/// Host copy of a parameter: data (row-major) and shape.
struct Host {
    data: Vec<f32>,
    dims: Vec<usize>,
}

impl Host {
    fn of(t: &Tensor) -> Result<Self> {
        Ok(Self { data: t.flatten_all()?.to_vec1::<f32>()?, dims: t.dims().to_vec() })
    }
    fn rows(&self) -> usize {
        self.dims[0]
    }
    fn cols(&self) -> usize {
        self.dims.get(1).copied().unwrap_or(1)
    }
    fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols() + j]
    }
}

/// `[r, c] → [2r, 2c]`, `W' = [[W/2, W/2]; [W/2, W/2]]` (reads and writes a duplicated stream).
fn dup2(w: &Host) -> Vec<f32> {
    let (r, c) = (w.rows(), w.cols());
    (0..2 * r).flat_map(|i| (0..2 * c).map(move |j| w.at(i % r, j % c) / 2.0)).collect()
}

/// `[r, c] → [r, 2c]`, `W' = [W/2, W/2]` (reads a duplicated stream).
fn read2(w: &Host) -> Vec<f32> {
    let (r, c) = (w.rows(), w.cols());
    (0..r).flat_map(|i| (0..2 * c).map(move |j| w.at(i, j % c) / 2.0)).collect()
}

/// `[r, c] → [2r, c]`, rows duplicated (writes a duplicated stream from an unchanged input).
fn write2(w: &Host) -> Vec<f32> {
    let (r, c) = (w.rows(), w.cols());
    (0..2 * r).flat_map(|i| (0..c).map(move |j| w.at(i % r, j))).collect()
}

/// `[r, c] → [r', c']` zero-padded: old row `i` stays row `i`, old column `j` goes to `col(j)`.
fn pad(w: &Host, rows: usize, cols: usize, col: impl Fn(usize) -> usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for i in 0..w.rows() {
        for j in 0..w.cols() {
            out[i * cols + col(j)] = w.at(i, j);
        }
    }
    out
}

/// Blocks of `k` equal row groups (Q, K, V) each transformed by [`dup2`], stacked again.
fn dup2_blocks(w: &Host, k: usize) -> Vec<f32> {
    let (r, c) = (w.rows() / k, w.cols());
    (0..k)
        .flat_map(|b| {
            let block = Host { data: w.data[b * r * c..(b + 1) * r * c].to_vec(), dims: vec![r, c] };
            dup2(&block)
        })
        .collect()
}

/// Position table for a new observation / action length: the start of the observation (the
/// page; up to 64 positions) keeps its positions, the end (conversation, question, tool results) keeps its distance
/// to the end, positions in between copy the first tail position; answer positions continue.
fn remap_positions(old: &Host, (n0, l0): (usize, usize), (n1, l1): (usize, usize)) -> Vec<f32> {
    let d = old.cols();
    let head = (n0 / 2).min(64).min(n1);
    let tail = (n0 - head).min(n1 - head);
    let mut rows = Vec::with_capacity(n1 + l1);
    for i in 0..n1 {
        let src = if i < head {
            i
        } else if i >= n1 - tail {
            n0 - (n1 - i)
        } else {
            n0 - tail
        };
        rows.push(src);
    }
    for j in 0..l1 {
        rows.push(n0 + j.min(l0 - 1));
    }
    rows.iter().flat_map(|&r| (0..d).map(move |c| old.at(r, c))).collect()
}

/// Rows `0..V` kept, rows beyond: the mean of the `CALC` and `ANSWER` rows (then `write` applied).
fn extend_vocab(w: &Host, vocab: usize) -> Host {
    let c = w.cols();
    let mut data = w.data.clone();
    for _ in w.rows()..vocab {
        data.extend((0..c).map(|j| 0.5 * (w.at(CALC as usize, j) + w.at(ANSWER as usize, j))));
    }
    Host { data, dims: vec![vocab, c] }
}

/// A copy of `base` twice as wide in the speech decoder and the latent planner, in the
/// observation format `layout` with `vocab` tokens (see the module docs). `noise` (relative to
/// each tensor's std) breaks the symmetry between duplicated units; `0` gives an exact copy.
pub fn grow(base: &UnifiedModel, layout: Layout, vocab: usize, noise: f32, seed: u64) -> Result<UnifiedModel> {
    let mut cfg = base.cfg.clone();
    cfg.preset = match base.cfg.preset.as_str() {
        "base" => "l".to_string(),
        "tiny" | "tiny2" => "tiny2".to_string(),
        other => bail!("no grown preset for '{other}'"),
    };
    cfg.speech.d_model *= 2;
    cfg.speech.n_heads *= 2;
    let ds = cfg.engine.jepa.d_state;
    cfg.engine.jepa.d_state *= 2;
    cfg.engine.jepa.d_action *= 2;
    cfg.engine.jepa.d_hidden *= 2;
    cfg.logit_scale *= 0.5;
    let old_lens = (base.cfg.layout.len, base.cfg.layout.action_len);
    cfg.set_layout(layout);
    cfg.engine.vocab_size = vocab;
    if vocab < base.cfg.engine.vocab_size {
        bail!("the grown vocabulary ({vocab}) must contain the original one")
    }
    let new = UnifiedModel::new(cfg, base.device())?;
    let ds2 = 2 * ds;
    let mut rng = Rng::new(seed);
    for (store_new, store_old) in [(&new.online, &base.online), (&new.target, &base.target)] {
        for name in store_new.names() {
            let var = store_new.var(&name).expect("listed parameter");
            let Some(old) = store_old.var(&name) else { bail!("grow: '{name}' is missing from the original model") };
            let w_host = Host::of(old.as_tensor())?;
            let w = &w_host;
            let (r, c) = (w.rows(), w.cols());
            let module = name.split('.').nth(1).unwrap_or_default();
            let mut duplicated = false;
            let data: Vec<f32> = if name == "ttt.tok_emb" {
                extend_vocab(w, vocab).data
            } else if name == "ttt.pos_emb" {
                remap_positions(w, old_lens, (layout.len, layout.action_len))
            } else if name.starts_with("ttt.") {
                w.data.clone()
            } else if name.starts_with("jepa.") {
                // fc1: new hidden rows 0; inputs mapped (state and action parts move apart)
                let rows = var.as_tensor().dims()[0];
                let cols = var.as_tensor().dims().get(1).copied().unwrap_or(1);
                if name.ends_with("bias") {
                    let mut b = w.data.clone();
                    b.resize(rows, 0.0);
                    b
                } else {
                    let col = |j: usize| -> usize {
                        match (module, name.contains("fc1")) {
                            ("world", true) => {
                                if j < ds {
                                    j
                                } else {
                                    ds2 + (j - ds)
                                }
                            }
                            ("inverse" | "policy", true) => {
                                if j < ds {
                                    j
                                } else {
                                    ds2 + (j - ds)
                                }
                            }
                            _ => j,
                        }
                    };
                    pad(w, rows, cols, col)
                }
            } else {
                // speech decoder
                duplicated = true;
                let leaf = name.rsplit('.').nth(1).unwrap_or_default();
                match (leaf, name.ends_with("bias")) {
                    ("speech", _) if name == "speech.tok_emb" => {
                        let e_host = extend_vocab(w, vocab);
                        let e = &e_host;
                        (0..vocab).flat_map(|i| (0..2 * c).map(move |j| e.at(i, j % c))).collect::<Vec<_>>()
                    }
                    ("speech", _) if name == "speech.pos_emb" => {
                        let l1 = layout.action_len;
                        (0..l1).flat_map(|i| (0..2 * c).map(move |j| w.at(i.min(r - 1), j % c))).collect()
                    }
                    ("speech", _) if name == "speech.plan_pos" => {
                        (0..r).flat_map(|i| (0..2 * c).map(move |j| w.at(i, j % c))).collect()
                    }
                    ("speech", _) if name == "speech.out_bias" => {
                        let min = w.data.iter().copied().fold(f32::INFINITY, f32::min);
                        let mut b = w.data.clone();
                        b.resize(vocab, min);
                        duplicated = false;
                        b
                    }
                    (_, true) if name.starts_with("speech.copy_gate") => {
                        duplicated = false;
                        w.data.clone()
                    }
                    (_, true) => [w.data.clone(), w.data.clone()].concat(), // plan_in, ctx_in
                    ("plan_in", false) => {
                        // inputs: the plan state, whose new dimensions are 0
                        (0..2 * r)
                            .flat_map(|i| (0..ds2).map(move |j| if j < ds { w.at(i % r, j) } else { 0.0 }))
                            .collect()
                    }
                    ("ctx_in", false) => write2(w),
                    ("mem_kv", false) => dup2_blocks(w, 2),
                    ("self_qkv", false) => dup2_blocks(w, 3),
                    ("copy_query" | "copy_gate", false) => read2(w),
                    _ => dup2(w),
                }
            };
            let shape = var.as_tensor().dims().to_vec();
            if data.len() != shape.iter().product::<usize>() {
                bail!("grow: '{name}' {:?} → {:?} produced {} values", w.dims, shape, data.len())
            }
            let mut data = data;
            if noise > 0.0 && duplicated {
                let std = (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt();
                let mut n = vec![0f32; data.len()];
                rng.fill_normal(&mut n, noise * std);
                data.iter_mut().zip(n).for_each(|(x, e)| *x += e);
            }
            var.set(&Tensor::from_vec(data, shape, base.device())?)?;
        }
    }
    Ok(new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::goal::Split;
    use crate::dialog::{self, LanguageData};
    use crate::unified::{UnifiedBatch, UnifiedConfig, UnifiedTrainConfig, UnifiedTrainer};
    use candle_core::Device;

    #[test]
    fn grown_model_computes_the_same_outputs() -> Result<()> {
        // a briefly trained tiny model (weights away from their initialisation)
        let model = UnifiedModel::new(UnifiedConfig::preset("tiny")?, &Device::Cpu)?;
        let tc =
            UnifiedTrainConfig { batch_size: 8, steps: 5, warmup: 1, log_every: 5, workers: 1, ..Default::default() };
        let mut tr = UnifiedTrainer::new(model, tc, LanguageData::builtin())?;
        tr.run_until(5, |_| {})?;
        let base = tr.model;
        let grown = grow(&base, Layout::V1, base.cfg.engine.vocab_size, 0.0, 1)?;
        assert_eq!(grown.speech.d_model(), 2 * base.speech.d_model());
        assert!(grown.num_params() > base.num_params());
        let data = LanguageData::builtin();
        let mut rng = Rng::new(3);
        let ex = dialog::mixed(&mut rng, &data, 12, Split::Train, &Layout::V1);
        let batch = UnifiedBatch::new(&ex, &Device::Cpu)?;
        let (_, a) = base.loss_with(&batch, &mut Rng::new(5), 0.0, 0.0)?;
        let (_, b) = grown.loss_with(&batch, &mut Rng::new(5), 0.0, 0.0)?;
        assert!((a.nll - b.nll).abs() < 1e-3 * a.nll.abs().max(1.0), "speech NLL {} vs {}", a.nll, b.nll);
        assert!((a.pointer - b.pointer).abs() < 1e-3, "pointer {} vs {}", a.pointer, b.pointer);
        // and the grown model can move to the scratchpad format with the appended tokens
        let v2 = grow(&base, Layout::V2, crate::text::ru().vocab_size(), 1e-3, 1)?;
        let ex = dialog::mixed(&mut rng, &data, 8, Split::Train, &Layout::V2);
        let (loss, _) = v2.loss(&UnifiedBatch::new(&ex, &Device::Cpu)?, &mut Rng::new(1))?;
        assert!(loss.to_scalar::<f32>()?.is_finite());
        Ok(())
    }

    #[test]
    fn positions_keep_the_page_start_and_the_tail_end() {
        let d = 2;
        let old = Host { data: (0..(10 + 4) * d).map(|x| (x / d) as f32).collect(), dims: vec![14, d] };
        let new = remap_positions(&old, (10, 4), (20, 6));
        let row = |i: usize| new[i * d];
        assert_eq!((row(0), row(4), row(9), row(19), row(18)), (0.0, 4.0, 5.0, 9.0, 8.0));
        assert_eq!((row(20), row(23), row(25)), (10.0, 13.0, 13.0), "answer positions continue");
        let same = remap_positions(&old, (10, 4), (10, 4));
        assert_eq!(same, old.data);
    }
}
