//! Training loop: AdamW on f32 master weights, warmup + cosine LR, global-norm gradient
//! clipping, EMA target update, and evaluation of both the teacher-forced decoder and the
//! full planning pipeline.

use std::time::Instant;

use candle_core::{Result, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::config::TrainConfig;
use crate::data::TaskSampler;
use crate::kernels::rng::Rng;
use crate::model::{CogModel, LossReport};
use crate::pipeline::CognitiveEngine;

/// Accuracy of the full engine under one planner setting.
#[derive(Debug, Clone, Default)]
pub struct EngineScore {
    pub mode: &'static str,
    pub token_acc: f32,
    pub exact: f32,
    pub mean_plan_energy: f32,
    pub mean_initial_energy: f32,
    pub mean_latency_us: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EvalReport {
    pub samples: usize,
    /// Decoder conditioned on the oracle (teacher-forced) plan — an upper bound.
    pub teacher_token_acc: f32,
    pub teacher_exact: f32,
    /// Full engine (packed weights, TTT → planner → CFM) under several planner settings;
    /// the first entry is the configured default.
    pub engine: Vec<EngineScore>,
    /// A few `(prompt, target, prediction)` triples of the default engine.
    pub examples: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
}

impl std::fmt::Display for EvalReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "eval n={} | oracle plan: tok {:.1}% seq {:.1}%",
            self.samples,
            100.0 * self.teacher_token_acc,
            100.0 * self.teacher_exact
        )?;
        for e in &self.engine {
            write!(
                f,
                "\n           {:<16} tok {:>5.1}% seq {:>5.1}% | energy {:.4} (warm start {:.4}) | {:>6.0} µs/query",
                e.mode,
                100.0 * e.token_acc,
                100.0 * e.exact,
                e.mean_plan_energy,
                e.mean_initial_energy,
                e.mean_latency_us
            )?;
        }
        Ok(())
    }
}

fn score(preds: &[Vec<u32>], targets: &[Vec<u32>]) -> (f32, f32) {
    let (mut tok, mut n, mut exact) = (0usize, 0usize, 0usize);
    for (p, t) in preds.iter().zip(targets) {
        let ok = p.iter().zip(t).filter(|(a, b)| a == b).count();
        tok += ok;
        n += t.len();
        exact += (ok == t.len()) as usize;
    }
    (tok as f32 / n.max(1) as f32, exact as f32 / preds.len().max(1) as f32)
}

pub struct Trainer {
    pub model: CogModel,
    pub tc: TrainConfig,
    pub sampler: TaskSampler,
    vars: Vec<Var>,
    opt: AdamW,
    rng: Rng,
    step: usize,
}

impl Trainer {
    pub fn new(model: CogModel, tc: TrainConfig) -> Result<Self> {
        let cfg = &model.cfg;
        let sampler = TaskSampler::new(tc.task, cfg.vocab_size, cfg.max_prompt_len, cfg.answer_len())?;
        let vars = model.trainable_vars();
        let opt =
            AdamW::new(vars.clone(), ParamsAdamW { lr: tc.lr, weight_decay: tc.weight_decay, ..Default::default() })?;
        let rng = Rng::stream(cfg.seed, 0x7EA1, 0);
        Ok(Self { model, tc, sampler, vars, opt, rng, step: 0 })
    }

    pub fn step(&self) -> usize {
        self.step
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

    /// One optimisation step; returns the loss report and the pre-clip gradient norm.
    pub fn train_step(&mut self) -> Result<(LossReport, f32)> {
        let batch = self.sampler.batch(&mut self.rng, self.tc.batch_size, self.model.device())?;
        let (loss, report) = self.model.loss(&batch, &mut self.rng, &self.tc)?;
        let mut grads = loss.backward()?;

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

    /// Evaluates on `n` fresh examples (held-out RNG stream).
    pub fn evaluate(&self, n: usize, seed: u64) -> Result<EvalReport> {
        let mut rng = Rng::stream(seed, 0xE7A1, 0);
        let batch = self.sampler.batch(&mut rng, n, self.model.device())?;
        let cd = self.tc.resolved_compute_dtype(self.model.device());

        let plan = self.model.teacher_plan(&batch, cd)?;
        let teacher = self.model.decode_graph(&plan, &mut rng)?;
        let (teacher_token_acc, teacher_exact) = score(&teacher, &batch.answers);

        let mut engine = CognitiveEngine::from_model(&self.model)?;
        let iterations = self.model.cfg.planner.iterations;
        let modes: [(&'static str, bool, usize); 3] =
            [("engine π+MPPI", true, iterations), ("  π only", true, 0), ("  MPPI (zero init)", false, iterations)];
        let mut scores = Vec::with_capacity(modes.len());
        let mut examples = Vec::new();
        for (mode, prior, iters) in modes {
            engine.set_policy_prior(prior);
            engine.set_mppi_iterations(iters);
            let mut preds = Vec::with_capacity(n);
            let (mut energy, mut e0, mut us) = (0f32, 0f32, 0f32);
            for (i, p) in batch.prompts.iter().enumerate() {
                let (out, g) = engine.generate(p, seed ^ (i as u64).wrapping_mul(0x9E37))?;
                energy += g.plan.energy;
                e0 += g.plan.initial_energy;
                us += g.timings.total().as_secs_f32() * 1e6;
                preds.push(out);
            }
            let (token_acc, exact) = score(&preds, &batch.answers);
            let k = n.max(1) as f32;
            if examples.is_empty() {
                examples = (0..n.min(4))
                    .map(|i| (batch.prompts[i].clone(), batch.answers[i].clone(), preds[i].clone()))
                    .collect();
            }
            scores.push(EngineScore {
                mode,
                token_acc,
                exact,
                mean_plan_energy: energy / k,
                mean_initial_energy: e0 / k,
                mean_latency_us: us / k,
            });
        }
        Ok(EvalReport { samples: n, teacher_token_acc, teacher_exact, engine: scores, examples })
    }

    /// Runs the configured number of steps, logging through `log`.
    pub fn run(&mut self, mut log: impl FnMut(&str)) -> Result<Option<EvalReport>> {
        let mut acc = LossReport::default();
        let mut acc_n = 0usize;
        let mut last_eval = None;
        let t0 = Instant::now();
        while self.step < self.tc.steps {
            let (r, gnorm) = self.train_step()?;
            acc.accumulate(&r);
            acc_n += 1;
            let s = self.step;
            if s.is_multiple_of(self.tc.log_every) || s == self.tc.steps {
                let mean = acc.scaled(1.0 / acc_n as f32);
                log(&format!(
                    "step {s:>5} | {mean} | |g| {gnorm:.3} | lr {:.2e} | {:.1} step/s",
                    self.lr_at(s - 1),
                    s as f64 / t0.elapsed().as_secs_f64()
                ));
                acc = LossReport::default();
                acc_n = 0;
            }
            if self.tc.eval_every > 0 && (s.is_multiple_of(self.tc.eval_every) || s == self.tc.steps) {
                let report = self.evaluate(self.tc.eval_samples, 1234)?;
                log(&format!("step {s:>5} | {report}"));
                last_eval = Some(report);
            }
        }
        Ok(last_eval)
    }
}
