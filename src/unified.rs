//! The unified model: one network that browses, calculates and talks.
//!
//! ```text
//!  observation ──► TTT encoder ──► s_0 = E(S_prompt) ──► latent reasoning ──► plan s_0 … s_H ─┐
//!  page, turns,     W_fast +            goal ĝ = s_0 + G(s_0)   tree of hypotheses + MPPI       │
//!  question, tool   features of every token ─────────────────────────────────────────────┐    │
//!                                                                                         ▼    ▼
//!                                              speech decoder (causal transformer, cross-attention
//!                                              to the plan and the tokens, pointer into the context)
//!                                                                                         │
//!                                      CLICK [link] ␣лампа <end> · CALC <none> ␣5+5 <end> · ANSWER <none> ␣Привет! Чем помочь? <end>
//! ```
//!
//! The encoder and the latent planner are those of the browsing agent ([`crate::ttt`],
//! [`crate::jepa`]): the model first "thinks" — searches a trajectory of latent states towards
//! the goal it predicts — and then speaks with the [`crate::speech`] decoder, one token at a time,
//! reading its plan and every context token. Every output is an agent action; a reply in a
//! conversation is the `ANSWER` action, so the same weights decide whether to search, calculate
//! or answer, and what to say.
//!
//! Training ([`UnifiedModel::loss`]) mixes browser states labelled by the teacher, dialogues,
//! grammar questions and text continuation ([`crate::dialog`]):
//!
//! ```text
//!   L = NLL_speech(y | plan, context) + λ_ptr L_pointer + L_JEPA (VICReg + goal + policy BC)
//! ```
//!
//! The decoder is conditioned on the plan the planner would produce: the policy rollout from
//! `s_0` (`s_{t+1} = P(s_t, π(s_t, ĝ))`), for a share of the examples on the teacher-forced
//! plan (which has seen the answer) instead, plus noise. The JEPA terms train the latent space as
//! in the browsing agent: the plan predicts the encodings of the question with more and more of
//! the answer.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{bail, DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::arena::Arena;
use crate::browser::action::{Action, ACTION_LEN};
use crate::browser::agent::Policy;
use crate::browser::goal::{Goal, Split};
use crate::browser::obs::{Note, OBS_LEN};
use crate::browser::{self, PageSnapshot};
use crate::config::{EngineConfig, PlannerKind};
use crate::dialog::{self, Example, LanguageData, Source};
use crate::jepa::planner::DepthStats;
use crate::jepa::{JEPAPlanner, Jepa, MppiWorkspace, PlanStats};
use crate::kernels::rng::Rng;
use crate::nn::ParamStore;
use crate::pipeline::{Generation, Reasoning, StageTimings};
use crate::speech::{self, Sampling, SpeechConfig, SpeechDecoder};
use crate::text;
use crate::ttt::{TttEncoder, SEGMENT_ANSWER, SEGMENT_PROMPT};
use crate::types::LatentState;

/// Architecture and training settings of the unified model.
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// Preset the configuration was built from (recorded in checkpoints).
    pub preset: String,
    /// Vocabulary, prompt length `N`, TTT encoder, JEPA and planner; `flow.seq_len` is the output
    /// length `L` (the flow decoder itself is not used).
    pub engine: EngineConfig,
    pub speech: SpeechConfig,
    /// Share of training examples whose decoder sees the teacher-forced plan instead of the
    /// policy rollout.
    pub teacher_plan: f64,
    /// Std of the noise added to the plan the decoder sees.
    pub plan_noise: f64,
}

impl UnifiedConfig {
    /// `tiny` (tests, ~1 M parameters) or `base` (the shipped model).
    pub fn preset(name: &str) -> Result<Self> {
        let vocab = text::ru().vocab_size();
        let base = match name {
            "tiny" => false,
            "base" => true,
            other => bail!("unknown unified preset '{other}' (tiny | base)"),
        };
        let mut e = EngineConfig::preset(if base { "small" } else { "tiny" }, vocab, OBS_LEN, ACTION_LEN)?;
        e.ttt.conv_width = browser::CONV_WIDTH;
        e.ttt.readout_last = browser::READOUT_LAST;
        e.ttt.readout_pools = browser::READOUT_POOLS;
        e.jepa.horizon = browser::HORIZON;
        e.jepa.probe_weight = 0.0;
        e.jepa.copy_dim = 0;
        e.jepa.copy_min_token = text::SPECIALS.len() as u32;
        e.planner.tree_beam = browser::TREE_BEAM;
        e.planner.tree_branch = browser::TREE_BRANCH;
        e.planner.kind = PlannerKind::Mppi;
        let speech = if base {
            SpeechConfig { d_model: 256, n_layers: 4, n_heads: 4, mlp_ratio: 4, max_len: ACTION_LEN, copy_dim: 32 }
        } else {
            SpeechConfig { d_model: 64, n_layers: 2, n_heads: 4, mlp_ratio: 2, max_len: ACTION_LEN, copy_dim: 16 }
        };
        Ok(Self { preset: name.to_string(), engine: e, speech, teacher_plan: 0.3, plan_noise: 0.05 })
    }

    /// `key=value` lines (written next to a checkpoint as `<ckpt>.cfg`).
    pub fn to_text(&self, seed: u64) -> String {
        let s = &self.speech;
        format!(
            "kind=unified\npreset={}\nseed={seed}\nd_model={}\nlayers={}\nheads={}\nmlp={}\ncopy={}\nteacher_plan={}\nplan_noise={}\n",
            self.preset, s.d_model, s.n_layers, s.n_heads, s.mlp_ratio, s.copy_dim, self.teacher_plan, self.plan_noise
        )
    }

    /// Parses [`UnifiedConfig::to_text`].
    pub fn from_text(text: &str) -> Result<Self> {
        let kv: HashMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
        if kv.get("kind") != Some(&"unified") {
            bail!("not a unified-model configuration")
        }
        let get = |k: &str| kv.get(k).copied().ok_or_else(|| candle_core::Error::Msg(format!("config: missing '{k}'")));
        let num = |k: &str| -> Result<usize> { get(k)?.parse().map_err(candle_core::Error::wrap) };
        let float = |k: &str| -> Result<f64> { get(k)?.parse().map_err(candle_core::Error::wrap) };
        let mut c = Self::preset(get("preset")?)?;
        c.engine.seed = num("seed")? as u64;
        c.speech.d_model = num("d_model")?;
        c.speech.n_layers = num("layers")?;
        c.speech.n_heads = num("heads")?;
        c.speech.mlp_ratio = num("mlp")?;
        c.speech.copy_dim = num("copy")?;
        c.teacher_plan = float("teacher_plan")?;
        c.plan_noise = float("plan_noise")?;
        Ok(c)
    }

    pub fn prompt_len(&self) -> usize {
        self.engine.max_prompt_len
    }

    pub fn answer_len(&self) -> usize {
        self.engine.answer_len()
    }
}

/// A batch of examples as tensors.
pub struct UnifiedBatch {
    /// `[B, N]` u32.
    pub prompt: Tensor,
    /// `[B, L]` u32.
    pub answer: Tensor,
    pub sources: Vec<Source>,
}

impl UnifiedBatch {
    pub fn new(examples: &[Example], device: &Device) -> Result<Self> {
        let b = examples.len();
        let prompt: Vec<u32> = examples.iter().flat_map(|e| e.prompt).collect();
        let answer: Vec<u32> = examples.iter().flat_map(|e| e.answer).collect();
        Ok(Self {
            prompt: Tensor::from_vec(prompt, (b, OBS_LEN), device)?,
            answer: Tensor::from_vec(answer, (b, ACTION_LEN), device)?,
            sources: examples.iter().map(|e| e.source).collect(),
        })
    }
}

/// Loss statistics of one source.
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceStats {
    pub nll: f64,
    pub tokens: usize,
    pub correct: usize,
    pub examples: usize,
}

impl SourceStats {
    /// Mean NLL per token.
    pub fn loss(&self) -> f64 {
        self.nll / self.tokens.max(1) as f64
    }
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.tokens.max(1) as f64
    }
    fn add(&mut self, o: &Self) {
        self.nll += o.nll;
        self.tokens += o.tokens;
        self.correct += o.correct;
        self.examples += o.examples;
    }
}

/// Scalar loss terms of a batch.
#[derive(Debug, Clone, Default)]
pub struct UnifiedReport {
    pub total: f32,
    pub nll: f32,
    pub pointer: f32,
    pub jepa: f32,
    pub goal: f32,
    pub by_source: Vec<(Source, SourceStats)>,
}

impl UnifiedReport {
    fn empty() -> Self {
        Self { by_source: Source::ALL.iter().map(|&s| (s, SourceStats::default())).collect(), ..Default::default() }
    }

    fn accumulate(&mut self, o: &Self) {
        self.total += o.total;
        self.nll += o.nll;
        self.pointer += o.pointer;
        self.jepa += o.jepa;
        self.goal += o.goal;
        for ((_, a), (_, b)) in self.by_source.iter_mut().zip(&o.by_source) {
            a.add(b);
        }
    }

    fn scaled(&self, k: f32) -> Self {
        Self {
            total: self.total * k,
            nll: self.nll * k,
            pointer: self.pointer * k,
            jepa: self.jepa * k,
            goal: self.goal * k,
            by_source: self.by_source.clone(),
        }
    }
}

impl std::fmt::Display for UnifiedReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "loss {:.3} | nll {:.3} ptr {:.3} jepa {:.3} goal {:.3} |",
            self.total, self.nll, self.pointer, self.jepa, self.goal
        )?;
        for (s, st) in &self.by_source {
            if st.tokens > 0 {
                write!(f, " {} {:.2} ({:.0}%)", s.name(), st.loss(), 100.0 * st.accuracy())?;
            }
        }
        Ok(())
    }
}

/// The unified model (graph path): parameters, loss, checkpoints.
pub struct UnifiedModel {
    pub cfg: UnifiedConfig,
    pub online: ParamStore,
    pub target: ParamStore,
    pub ttt: TttEncoder,
    pub jepa: Jepa,
    pub speech: SpeechDecoder,
    device: Device,
}

/// What the encoder made of one observation (batch of one).
pub struct Encoded {
    /// `s_0` and the goal `ĝ`, `[1, d_s]`.
    pub s0: Tensor,
    pub goal: Tensor,
    /// TTT features `[1, N, d_feat]`, pointer keys `[1, N, d_key]`, the tokens `[1, N]`.
    pub features: Tensor,
    pub keys: Option<Tensor>,
    pub context: Tensor,
}

impl UnifiedModel {
    pub fn new(cfg: UnifiedConfig, device: &Device) -> Result<Self> {
        cfg.engine.validate()?;
        let seed = cfg.engine.seed;
        let mut online = ParamStore::new(device, seed);
        let mut target = ParamStore::new(device, seed ^ 0x7A11);
        // the TTT encoder keeps copy keys for the speech decoder's pointer
        let mut tcfg = cfg.engine.clone();
        tcfg.jepa.copy_dim = cfg.speech.copy_dim;
        let ttt = TttEncoder::new(&mut online, &tcfg)?;
        let jepa = Jepa::new(&mut online, &mut target, &cfg.engine)?;
        let speech = SpeechDecoder::new(
            &mut online,
            &cfg.speech,
            cfg.engine.vocab_size,
            cfg.engine.jepa.d_state,
            cfg.engine.plan_len(),
            ttt.d_features(),
            cfg.engine.jepa.copy_min_token,
        )?;
        Ok(Self { cfg, online, target, ttt, jepa, speech, device: device.clone() })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_params(&self) -> usize {
        self.online.num_params()
    }

    pub fn trainable_vars(&self) -> Vec<Var> {
        self.online.vars()
    }

    /// The greedy chain of thoughts: `s_{t+1} = P(s_t, π(s_t, ĝ))` from `s_0: [B, d_s]` →
    /// `[B, H + 1, d_s]` (the actions carry no gradient).
    pub fn rollout(&self, s0: &Tensor, goal: &Tensor) -> Result<Tensor> {
        let mut states = vec![s0.clone()];
        for t in 0..self.cfg.engine.jepa.horizon {
            let a = self.jepa.policy_action(&states[t].detach(), &goal.detach())?.detach();
            let next = self.jepa.world.forward(&states[t], &a)?;
            states.push(next);
        }
        Tensor::stack(&states, 1)
    }

    /// Joint loss of a batch (see the module docs).
    pub fn loss(&self, batch: &UnifiedBatch, rng: &mut Rng) -> Result<(Tensor, UnifiedReport)> {
        self.loss_with(batch, rng, self.cfg.teacher_plan, self.cfg.plan_noise)
    }

    /// [`UnifiedModel::loss`] with another share of teacher-forced plans and plan noise.
    pub fn loss_with(
        &self,
        batch: &UnifiedBatch,
        rng: &mut Rng,
        teacher_plan: f64,
        plan_noise: f64,
    ) -> Result<(Tensor, UnifiedReport)> {
        let (b, l) = batch.answer.dims2()?;
        let n = batch.prompt.dims2()?.1;
        let h = self.cfg.engine.jepa.horizon;
        let chunk = l / h;
        let x = Tensor::cat(
            &[self.ttt.embed(&batch.prompt, 0, SEGMENT_PROMPT)?, self.ttt.embed(&batch.answer, n, SEGMENT_ANSWER)?],
            1,
        )?;
        let snaps: Vec<usize> = (0..=h).map(|i| n + i * chunk).collect();
        let (readouts, features, keys) = self.ttt.encode_with_features(&x, &snaps, n)?;
        let (teacher, terms) = self.jepa.forward_train(&readouts, &self.cfg.engine.jepa)?;
        let s0 = teacher.narrow(1, 0, 1)?.squeeze(1)?;
        let rollout = self.rollout(&s0, &self.jepa.goal_of(&s0)?)?;
        let m: Vec<f32> = (0..b).map(|_| (rng.uniform() < teacher_plan) as u8 as f32).collect();
        let m = Tensor::from_vec(m, (b, 1, 1), &self.device)?;
        let mut plan = (teacher.broadcast_mul(&m)? + rollout.broadcast_mul(&(m.neg()? + 1.0)?)?)?;
        if plan_noise > 0.0 {
            let mut v = vec![0f32; plan.elem_count()];
            rng.fill_normal(&mut v, plan_noise as f32);
            let noise = Tensor::from_vec(v, plan.dims(), &self.device)?;
            plan = (plan + noise)?;
        }
        let mem = self.speech.memory(&plan, &features, &batch.prompt, keys.as_ref())?;
        let sp = self.speech.loss(&batch.answer, &mem)?;
        let jepa = terms.weighted(&self.cfg.engine.jepa)?;
        let total = (&sp.total + &jepa)?;
        let mut report = UnifiedReport::empty();
        for (i, src) in batch.sources.iter().enumerate() {
            let st = &mut report.by_source.iter_mut().find(|(s, _)| s == src).expect("every source").1;
            st.nll += sp.per_example[i].0 as f64;
            st.tokens += sp.per_example[i].1;
            st.correct += sp.correct[i];
            st.examples += 1;
        }
        report.total = total.to_scalar::<f32>()?;
        report.nll = sp.nll;
        report.pointer = sp.pointer;
        report.jepa = jepa.to_scalar::<f32>()?;
        report.goal = terms.goal.to_scalar::<f32>()?;
        Ok((total, report))
    }

    /// Encodes one observation (inference).
    pub fn encode(&self, observation: &[u32]) -> Result<Encoded> {
        let n = observation.len();
        let context = Tensor::from_slice(observation, (1, n), &self.device)?;
        let x = self.ttt.embed(&context, 0, SEGMENT_PROMPT)?;
        let (readouts, features, keys) = self.ttt.encode_with_features(&x, &[n], n)?;
        let s0 = self.jepa.encode(&readouts[0])?;
        let goal = self.jepa.goal_of(&s0)?;
        Ok(Encoded { s0, goal, features, keys, context })
    }

    /// Says something given `plans: [K, H + 1, d_s]` (K utterances for the same observation).
    pub fn speak(
        &self,
        enc: &Encoded,
        plans: &Tensor,
        sampling: Sampling,
        rng: &mut Rng,
    ) -> Result<Vec<(Vec<u32>, f32)>> {
        let k = plans.dims()[0];
        let rep = |t: &Tensor| -> Result<Tensor> {
            let mut r = vec![1; t.rank()];
            r[0] = k;
            t.repeat(r)
        };
        let keys = enc.keys.as_ref().map(rep).transpose()?;
        let mem = self.speech.memory(plans, &rep(&enc.features)?, &rep(&enc.context)?, keys.as_ref())?;
        self.speech.generate(&mem, sampling, rng)
    }

    /// EMA step of the JEPA target encoder.
    pub fn ema_update(&self) -> Result<()> {
        Jepa::ema(&self.online, &self.target, self.cfg.engine.jepa.ema_decay)
    }

    /// Weights to `path` (safetensors) and the configuration to `path.cfg`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let mut map = HashMap::new();
        self.online.export("online", &mut map);
        self.target.export("target", &mut map);
        candle_core::safetensors::save(&map, path)?;
        std::fs::write(format!("{}.cfg", path.display()), self.cfg.to_text(self.cfg.engine.seed))
            .map_err(candle_core::Error::wrap)
    }

    /// Builds the model described by `path.cfg` and loads its weights.
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let cfg_text = std::fs::read_to_string(format!("{}.cfg", path.display()))
            .map_err(|e| candle_core::Error::Msg(format!("{}.cfg: {e}", path.display())))?;
        let model = Self::new(UnifiedConfig::from_text(&cfg_text)?, device)?;
        model.load_weights(path)?;
        Ok(model)
    }

    /// Overwrites the parameters from a checkpoint of the same architecture.
    pub fn load_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let map = candle_core::safetensors::load(path, &self.device)?;
        self.online.import("online", &map)?;
        self.target.import("target", &map)
    }

    /// Whether `path.cfg` describes a unified model.
    pub fn is_checkpoint<P: AsRef<Path>>(path: P) -> bool {
        std::fs::read_to_string(format!("{}.cfg", path.as_ref().display()))
            .is_ok_and(|t| t.lines().any(|l| l == "kind=unified"))
    }
}

/// Training settings.
#[derive(Debug, Clone)]
pub struct UnifiedTrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub warmup: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub log_every: usize,
    /// Threads the batch is split across (data parallelism within a step: candle runs
    /// element-wise ops on one core, so micro-batches on several cores add up).
    pub workers: usize,
}

impl Default for UnifiedTrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            steps: 20000,
            lr: 1e-3,
            min_lr: 5e-5,
            warmup: 300,
            weight_decay: 0.01,
            grad_clip: 1.0,
            log_every: 50,
            workers: std::thread::available_parallelism().map_or(1, |n| n.get()).min(4),
        }
    }
}

/// AdamW training of a [`UnifiedModel`] on the mixture of [`crate::dialog`].
pub struct UnifiedTrainer {
    pub model: UnifiedModel,
    pub tc: UnifiedTrainConfig,
    pub data: LanguageData,
    vars: Vec<Var>,
    opt: AdamW,
    rng: Rng,
    step: usize,
}

impl UnifiedTrainer {
    pub fn new(model: UnifiedModel, tc: UnifiedTrainConfig, data: LanguageData) -> Result<Self> {
        let vars = model.trainable_vars();
        let opt =
            AdamW::new(vars.clone(), ParamsAdamW { lr: tc.lr, weight_decay: tc.weight_decay, ..Default::default() })?;
        let rng = Rng::stream(model.cfg.engine.seed, 0x0A1F, 0);
        Ok(Self { model, tc, data, vars, opt, rng, step: 0 })
    }

    pub fn step(&self) -> usize {
        self.step
    }

    /// Continues a run from `step` (weights loaded from its checkpoint; AdamW moments restart).
    pub fn resume_at(&mut self, step: usize) {
        self.step = step;
        self.rng = Rng::stream(self.model.cfg.engine.seed, 0x0A1F, step as u64);
    }

    /// Linear warmup, then cosine decay to `min_lr`.
    pub fn lr_at(&self, step: usize) -> f64 {
        let tc = &self.tc;
        if step < tc.warmup {
            return tc.lr * (step + 1) as f64 / tc.warmup as f64;
        }
        let p = (step - tc.warmup) as f64 / (tc.steps.saturating_sub(tc.warmup)).max(1) as f64;
        tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos())
    }

    /// One optimisation step on a fresh mixed batch; returns the report and the gradient norm.
    /// The batch is split into `workers` micro-batches whose gradients are averaged.
    pub fn train_step(&mut self) -> Result<(UnifiedReport, f32)> {
        let examples = dialog::mixed(&mut self.rng, &self.data, self.tc.batch_size, Split::Train);
        let k = self.tc.workers.clamp(1, examples.len());
        let chunks: Vec<&[Example]> = examples.chunks(examples.len().div_ceil(k)).collect();
        let seeds: Vec<u64> = chunks.iter().map(|_| self.rng.below(1 << 30) as u64).collect();
        let model = &self.model;
        let results: Vec<Result<(candle_core::backprop::GradStore, UnifiedReport, usize)>> = std::thread::scope(|sc| {
            let handles: Vec<_> = chunks
                .iter()
                .zip(&seeds)
                .map(|(chunk, &seed)| {
                    sc.spawn(move || {
                        let batch = UnifiedBatch::new(chunk, model.device())?;
                        let (loss, report) = model.loss(&batch, &mut Rng::new(seed))?;
                        Ok((loss.backward()?, report, chunk.len()))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("training worker panicked")).collect()
        });
        let mut parts = Vec::with_capacity(results.len());
        for r in results {
            parts.push(r?);
        }
        let n = examples.len() as f64;
        let mut report = UnifiedReport::empty();
        for (_, r, len) in &parts {
            report.accumulate(&r.scaled(*len as f32 / n as f32));
        }
        let (mut grads, _, len0) = parts.remove(0);
        for v in &self.vars {
            let mut g = match grads.remove(v.as_tensor()) {
                Some(g) => (g * (len0 as f64 / n))?,
                None => continue,
            };
            for (other, _, len) in &parts {
                if let Some(o) = other.get(v.as_tensor()) {
                    g = (g + (o * (*len as f64 / n))?)?;
                }
            }
            grads.insert(v.as_tensor(), g);
        }
        let mut sq = 0f64;
        for v in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        let norm = sq.sqrt();
        if self.tc.grad_clip > 0.0 && norm > self.tc.grad_clip {
            let s = self.tc.grad_clip / norm;
            for v in &self.vars {
                if let Some(g) = grads.remove(v.as_tensor()) {
                    grads.insert(v.as_tensor(), (g * s)?);
                }
            }
        }
        self.opt.set_learning_rate(self.lr_at(self.step));
        self.opt.step(&grads)?;
        self.model.ema_update()?;
        self.step += 1;
        Ok((report, norm as f32))
    }

    /// Trains up to step `until`, logging every `log_every` steps.
    pub fn run_until(&mut self, until: usize, mut log: impl FnMut(&str)) -> Result<()> {
        let until = until.min(self.tc.steps);
        let (t0, start) = (Instant::now(), self.step);
        let mut acc = UnifiedReport::empty();
        let mut n = 0usize;
        while self.step < until {
            let (r, g) = self.train_step()?;
            acc.accumulate(&r);
            n += 1;
            let s = self.step;
            if s.is_multiple_of(self.tc.log_every) || s == until {
                log(&format!(
                    "step {s:>6} | {} | |g| {g:.2} | lr {:.2e} | {:.2} step/s",
                    acc.scaled(1.0 / n as f32),
                    self.lr_at(s - 1),
                    (s - start) as f64 / t0.elapsed().as_secs_f64()
                ));
                acc = UnifiedReport::empty();
                n = 0;
            }
        }
        Ok(())
    }
}

/// Teacher-forced loss per source on held-out data (the decoder sees the policy rollout, as at
/// inference without search).
pub fn validation(model: &UnifiedModel, data: &LanguageData, per_source: usize, seed: u64) -> Result<UnifiedReport> {
    let mut rng = Rng::stream(seed, 0x7A1D, 0);
    let mut total = UnifiedReport::empty();
    for source in Source::ALL {
        let mut left = per_source;
        while left > 0 {
            let k = left.min(32);
            let ex: Vec<Example> = (0..k).map(|_| dialog::example(&mut rng, data, source, Split::HeldOut)).collect();
            let batch = UnifiedBatch::new(&ex, model.device())?;
            let (_, r) = model.loss_with(&batch, &mut rng, 0.0, 0.0)?;
            total.accumulate(&r);
            left -= k;
        }
    }
    Ok(total)
}

/// The latent reasoning of one decision.
#[derive(Debug, Clone)]
pub struct Thought {
    pub stats: PlanStats,
    /// The chosen plan `[1, H + 1, d_s]`.
    pub plan: Tensor,
    /// Surviving hypotheses of the tree search: energy and trajectory `[H + 1, d_s]` (best first).
    pub hypotheses: Vec<(f32, Tensor)>,
    pub depths: Vec<DepthStats>,
}

/// Inference: encode → think (tree search + MPPI over the latent plan) → speak.
pub struct UnifiedEngine {
    pub model: UnifiedModel,
    planner: JEPAPlanner,
    ws: MppiWorkspace,
    _arena: Arena,
    /// How replies are sampled (greedy by default).
    pub sampling: Sampling,
    /// Run the latent search; off = the greedy chain of thoughts (policy rollout) only.
    pub search: bool,
    rng: Rng,
}

impl UnifiedEngine {
    pub fn new(model: UnifiedModel) -> Result<Self> {
        let e = &model.cfg.engine;
        let planner = JEPAPlanner::new(model.jepa.world.pack(DType::F32)?, e.jepa.horizon, &e.planner)
            .with_policy(model.jepa.policy.pack(DType::F32)?);
        let mut arena = Arena::new(&Device::Cpu);
        let ws = planner.workspace(&mut arena)?;
        Ok(Self { model, planner, ws, _arena: arena, sampling: Sampling::GREEDY, search: true, rng: Rng::new(7) })
    }

    /// Reseeds the sampler.
    pub fn seed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// Latent reasoning for an encoded observation.
    pub fn think(&mut self, enc: &Encoded, seed: u64) -> Result<Thought> {
        let e = &self.model.cfg.engine;
        let (h, ds) = (e.jepa.horizon, e.jepa.d_state);
        let dev = self.model.device().clone();
        if !self.search {
            let plan = self.model.rollout(&enc.s0, &enc.goal)?;
            return Ok(Thought { stats: PlanStats::default(), plan, hypotheses: Vec::new(), depths: Vec::new() });
        }
        let host = |t: &Tensor| -> Result<LatentState> {
            LatentState::new(t.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?, ds)
        };
        let stats = self.planner.plan_into(&host(&enc.s0)?, &host(&enc.goal)?, &mut self.ws, seed)?;
        let plan = Tensor::from_vec(self.ws.plan.flatten_all()?.to_vec1::<f32>()?, (1, h + 1, ds), &dev)?;
        let hypotheses = self
            .ws
            .hypotheses()
            .map(|(c, traj)| Ok((c, Tensor::from_slice(traj, (h + 1, ds), &dev)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Thought { stats, plan, hypotheses, depths: self.ws.tree_depths().collect() })
    }

    /// One decision: the output tokens (`L`, padded) and the reasoning behind them.
    pub fn respond(&mut self, observation: &[u32], seed: u64) -> Result<(Vec<u32>, Thought, Encoded)> {
        let enc = self.model.encode(observation)?;
        let thought = self.think(&enc, seed)?;
        let out = self.model.speak(&enc, &thought.plan, self.sampling, &mut self.rng)?.remove(0).0;
        Ok((out, thought, enc))
    }

    /// What each surviving hypothesis of the tree search would say (greedy), best first.
    pub fn alternatives(&mut self, enc: &Encoded, thought: &Thought) -> Result<Vec<(f32, Vec<u32>)>> {
        if thought.hypotheses.is_empty() {
            return Ok(Vec::new());
        }
        let plans = Tensor::stack(&thought.hypotheses.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>(), 0)?;
        let said = self.model.speak(enc, &plans, Sampling::GREEDY, &mut self.rng)?;
        Ok(thought.hypotheses.iter().zip(said).map(|((e, _), (t, _))| (*e, t)).collect())
    }

    /// `k` sampled utterances (temperature `t`), most likely first.
    pub fn samples(&mut self, enc: &Encoded, thought: &Thought, k: usize, t: f32) -> Result<Vec<(Vec<u32>, f32)>> {
        let plans = thought.plan.repeat((k, 1, 1))?;
        let s = Sampling { temperature: t, top_k: 40, greedy_prefix: 2 };
        let mut out = self.model.speak(enc, &plans, s, &mut self.rng)?;
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }
}

/// The unified model as a browsing policy (see [`crate::browser::agent`]).
pub struct UnifiedPolicy {
    pub engine: UnifiedEngine,
    seed: u64,
    trace: bool,
    last: Option<Generation>,
    reasoning: Option<Reasoning>,
    previous: Option<(Vec<u32>, Option<Action>)>,
    no_effect: Vec<Action>,
}

impl UnifiedPolicy {
    pub fn new(engine: UnifiedEngine) -> Self {
        Self { engine, seed: 0, trace: false, last: None, reasoning: None, previous: None, no_effect: Vec::new() }
    }

    /// Also decode what every surviving hypothesis would say (for display).
    pub fn with_trace(mut self) -> Self {
        self.trace = true;
        self
    }
}

impl Policy for UnifiedPolicy {
    fn act(&mut self, observation: &[u32], _: &Goal, _: &PageSnapshot, _: Option<&Note>) -> Result<Vec<u32>> {
        self.seed += 1;
        let t0 = Instant::now();
        let (mut out, thought, enc) = self.engine.respond(observation, self.seed)?;
        let think = t0.elapsed();
        match self.previous.take() {
            Some((obs, Some(action))) if obs == observation => self.no_effect.push(action),
            Some((obs, _)) if obs == observation => {}
            _ => self.no_effect.clear(),
        }
        let bpe = text::ru();
        let usable = |t: &[u32]| Action::decode(t, bpe).filter(|a| !self.no_effect.contains(a));
        let mut alternatives = Vec::new();
        if usable(&out).is_none() || self.trace {
            alternatives = self.engine.alternatives(&enc, &thought)?;
        }
        if usable(&out).is_none() {
            // the other hypotheses of the search, then a few samples
            let mut pool: Vec<Vec<u32>> = alternatives.iter().map(|(_, t)| t.clone()).collect();
            if !pool.iter().any(|t| usable(t).is_some()) {
                pool.extend(self.engine.samples(&enc, &thought, 4, 0.8)?.into_iter().map(|(t, _)| t));
            }
            if let Some(p) = pool.into_iter().find(|t| usable(t).is_some()) {
                out = p;
            }
        }
        self.previous = Some((observation.to_vec(), Action::decode(&out, bpe)));
        self.last = Some(Generation {
            plan: thought.stats,
            timings: StageTimings { encode: Duration::ZERO, plan: think, decode: Duration::ZERO },
            refined: false,
        });
        self.reasoning = self.trace.then(|| Reasoning {
            stats: thought.stats,
            refined: false,
            depths: thought.depths.clone(),
            hypotheses: alternatives,
            chain: Vec::new(),
        });
        Ok(out)
    }

    fn last_generation(&self) -> Option<Generation> {
        self.last
    }

    fn last_reasoning(&self) -> Option<Reasoning> {
        self.reasoning.clone()
    }
}

/// Greedy answers to grammar questions: exact-match accuracy per question kind.
pub fn grammar_accuracy(
    engine: &mut UnifiedEngine,
    questions: &[dialog::Question],
    n: usize,
    seed: u64,
) -> Result<Vec<(String, usize, usize)>> {
    let mut rng = Rng::stream(seed, 0x6A77, 0);
    let mut by_kind: Vec<(String, usize, usize)> = Vec::new();
    for _ in 0..n {
        let q = &questions[rng.below(questions.len())];
        let prompt = browser::obs::encode_dialog(&dialog::background_page(&mut rng), &[], &q.question, None);
        let (out, _, _) = engine.respond(&prompt, rng.below(1 << 30) as u64)?;
        let ok =
            matches!(Action::decode(&out, text::ru()), Some(Action::Answer { text }) if text.trim() == q.answer.trim());
        match by_kind.iter_mut().find(|(k, _, _)| *k == q.kind) {
            Some(e) => {
                e.1 += 1;
                e.2 += ok as usize;
            }
            None => by_kind.push((q.kind.clone(), 1, ok as usize)),
        }
    }
    by_kind.sort();
    Ok(by_kind)
}

/// Rows of an example batch (for tests and tools).
pub fn batch_of(examples: &[Example]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    (examples.iter().map(|e| e.prompt.to_vec()).collect(), examples.iter().map(|e| e.answer.to_vec()).collect())
}

pub use speech::rows_tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_and_tiny_model_trains() -> Result<()> {
        let cfg = UnifiedConfig::preset("tiny")?;
        let back = UnifiedConfig::from_text(&cfg.to_text(cfg.engine.seed))?;
        assert_eq!((back.speech.d_model, back.speech.copy_dim), (cfg.speech.d_model, cfg.speech.copy_dim));
        let model = UnifiedModel::new(cfg, &Device::Cpu)?;
        let tc =
            UnifiedTrainConfig { batch_size: 8, steps: 30, lr: 3e-3, warmup: 5, log_every: 10, ..Default::default() };
        let mut tr = UnifiedTrainer::new(model, tc, LanguageData::builtin())?;
        let mut first = None;
        let mut last = 0.0;
        tr.run_until(30, |line| {
            let nll: f32 =
                line.split("nll ").nth(1).and_then(|s| s.split_whitespace().next()).unwrap().parse().unwrap();
            first.get_or_insert(nll);
            last = nll;
        })?;
        assert!(last < first.unwrap(), "nll {first:?} → {last}");
        // inference: think, speak, alternatives
        let mut engine = UnifiedEngine::new(tr.model)?;
        let prompt = browser::obs::encode_dialog(&browser::data::snapshot("/w/1/", None), &[], "Привет!", None);
        let (out, thought, enc) = engine.respond(&prompt, 1)?;
        assert_eq!(out.len(), ACTION_LEN);
        assert!(thought.stats.tree_nodes > 0);
        assert_eq!(engine.alternatives(&enc, &thought)?.len(), thought.hypotheses.len());
        Ok(())
    }
}
