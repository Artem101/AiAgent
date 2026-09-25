//! The browsing agent: observe → think → act, until the policy answers or runs out of steps.

use std::time::{Duration, Instant};

use candle_core::Result;

use super::obs::{self, OBS_LEN};
use super::vocab::{self, token_str, Action, Goal, ATTR_TOKENS, ITEMS, ITEM_TOKENS};
use super::world::{self, World};
use super::{data, expert, Browser, PageSnapshot, Role};
use crate::kernels::rng::Rng;
use crate::pipeline::CognitiveEngine;

/// Maps an observation to action tokens.
pub trait Policy {
    fn act(&mut self, observation: &[u32]) -> Result<Vec<u32>>;
}

/// The scripted teacher ([`expert::act`]).
pub struct ExpertPolicy;

impl Policy for ExpertPolicy {
    fn act(&mut self, observation: &[u32]) -> Result<Vec<u32>> {
        Ok(expert::act_tokens(observation).to_vec())
    }
}

/// The trained engine: every step is one `TTT → JEPA → CFM` generation.
pub struct EnginePolicy {
    pub engine: CognitiveEngine,
    seed: u64,
}

impl EnginePolicy {
    pub fn new(engine: CognitiveEngine) -> Self {
        Self { engine, seed: 0 }
    }
}

impl Policy for EnginePolicy {
    fn act(&mut self, observation: &[u32]) -> Result<Vec<u32>> {
        self.seed += 1;
        Ok(self.engine.generate(observation, self.seed)?.0)
    }
}

/// One step of an episode.
#[derive(Debug, Clone)]
pub struct Step {
    pub url: String,
    pub observation: [u32; OBS_LEN],
    /// Raw policy output.
    pub output: Vec<u32>,
    /// `None` when the output is not a well-formed action.
    pub action: Option<Action>,
    /// Why the action could not be carried out (missing element, malformed output).
    pub error: Option<String>,
    /// Time the policy took to decide.
    pub think_time: Duration,
}

/// A finished episode.
#[derive(Debug, Clone)]
pub struct Episode {
    pub world: u64,
    pub goal: Goal,
    pub steps: Vec<Step>,
    /// Value token the agent answered with, if it answered.
    pub answer: Option<u32>,
    /// Ground truth from the world's facts.
    pub truth: u32,
}

impl Episode {
    pub fn success(&self) -> bool {
        self.answer == Some(self.truth)
    }
}

/// Carries out `action` on the page of `snap`. `Ok(Some(reason))` is a policy mistake
/// (e.g. clicking text that is not on the page); `Err` is a browser failure.
fn execute(browser: &mut dyn Browser, snap: &PageSnapshot, action: Action) -> Result<Option<String>> {
    match action {
        Action::Click { role, word } => match snap.find(role, token_str(word)) {
            Some(id) => browser.click(id).map(|_| None),
            None => Ok(Some(format!("no {} \"{}\" on the page", role.name(), token_str(word)))),
        },
        Action::Type { word } => match snap.first(Role::Input) {
            Some(id) => browser.type_text(id, token_str(word)).map(|_| None),
            None => Ok(Some("no text field on the page".into())),
        },
        Action::Back => browser.back().map(|_| None),
        Action::Answer { .. } => Ok(None),
    }
}

/// Opens the search page of `world` and lets `policy` browse until it answers or
/// `max_steps` actions were taken. `on_step` sees every step as it happens.
pub fn run_episode(
    browser: &mut dyn Browser,
    policy: &mut dyn Policy,
    origin: &str,
    world: u64,
    goal: Goal,
    max_steps: usize,
    mut on_step: impl FnMut(&Step),
) -> Result<Episode> {
    browser.goto(&world::home_url(origin, world))?;
    let truth = World::new(world).value(goal.item, goal.attr);
    let mut steps = Vec::new();
    let mut answer = None;
    for _ in 0..max_steps {
        let snap = browser.snapshot()?;
        let observation = obs::encode(&snap, &goal);
        let t0 = Instant::now();
        let output = policy.act(&observation)?;
        let think_time = t0.elapsed();
        let action = Action::decode(&output);
        let error = match action {
            Some(a) => execute(browser, &snap, a)?,
            None => Some(format!("malformed action [{}]", vocab::describe(&output))),
        };
        let step = Step { url: snap.url, observation, output, action, error, think_time };
        on_step(&step);
        steps.push(step);
        if let Some(Action::Answer { word }) = action {
            answer = Some(word);
            break;
        }
    }
    Ok(Episode { world, goal, steps, answer, truth })
}

/// Aggregate over many episodes.
#[derive(Debug, Clone, Default)]
pub struct AgentReport {
    pub episodes: usize,
    pub successes: usize,
    /// Episodes that ended with a wrong answer (the rest of the failures ran out of steps).
    pub wrong_answers: usize,
    pub steps: usize,
    /// Steps whose action was malformed or referred to a missing element.
    pub failed_actions: usize,
    pub think_time: Duration,
    pub wall_time: Duration,
}

impl AgentReport {
    pub fn success_rate(&self) -> f32 {
        self.successes as f32 / self.episodes.max(1) as f32
    }
}

impl std::fmt::Display for AgentReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let steps = self.steps.max(1) as f64;
        write!(
            f,
            "success {:.1}% ({}/{}) | wrong answer {} | no answer {} | {:.2} steps/episode | failed actions {:.1}% | \
             think {:.1} ms/step | {:.0} ms/episode",
            100.0 * self.success_rate(),
            self.successes,
            self.episodes,
            self.wrong_answers,
            self.episodes - self.successes - self.wrong_answers,
            self.steps as f64 / self.episodes.max(1) as f64,
            100.0 * self.failed_actions as f64 / steps,
            1e3 * self.think_time.as_secs_f64() / steps,
            1e3 * self.wall_time.as_secs_f64() / self.episodes.max(1) as f64,
        )
    }
}

/// Random `(world, goal)` pairs for evaluation (deterministic in `seed`).
pub fn sample_tasks(n: usize, seed: u64) -> Vec<(u64, Goal)> {
    let mut rng = Rng::stream(seed, 0xA6E7, 0);
    (0..n)
        .map(|_| {
            let world = rng.below(1_000_000) as u64;
            let item = ITEM_TOKENS.start + rng.below(ITEMS.len()) as u32;
            let attr = ATTR_TOKENS.start + rng.below(ATTR_TOKENS.len()) as u32;
            (world, Goal { item, attr })
        })
        .collect()
}

/// Runs `episodes` random tasks and reports the success rate.
pub fn evaluate(
    browser: &mut dyn Browser,
    policy: &mut dyn Policy,
    origin: &str,
    episodes: usize,
    seed: u64,
    max_steps: usize,
) -> Result<AgentReport> {
    let mut r = AgentReport { episodes, ..Default::default() };
    let t0 = Instant::now();
    for (world, goal) in sample_tasks(episodes, seed) {
        let ep = run_episode(browser, policy, origin, world, goal, max_steps, |_| {})?;
        r.successes += ep.success() as usize;
        r.wrong_answers += (ep.answer.is_some() && !ep.success()) as usize;
        r.steps += ep.steps.len();
        r.failed_actions += ep.steps.iter().filter(|s| s.error.is_some()).count();
        r.think_time += ep.steps.iter().map(|s| s.think_time).sum::<Duration>();
    }
    r.wall_time = t0.elapsed();
    Ok(r)
}

/// Per-action-type accuracy on single browser states (open loop, expert labels).
#[derive(Debug, Clone, Default)]
pub struct StepReport {
    /// `(expert verb, states, exact action, right verb)` for CLICK, TYPE, BACK, ANSWER.
    pub by_verb: Vec<(&'static str, usize, usize, usize)>,
}

impl StepReport {
    pub fn exact(&self) -> f32 {
        let (n, ok) = self.by_verb.iter().fold((0, 0), |(n, ok), r| (n + r.1, ok + r.2));
        ok as f32 / n.max(1) as f32
    }
}

impl std::fmt::Display for StepReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "steps: exact {:.1}%", 100.0 * self.exact())?;
        for (verb, n, ok, verb_ok) in &self.by_verb {
            let pct = |x: usize| 100.0 * x as f32 / (*n).max(1) as f32;
            write!(f, " | {verb} {:.1}% (verb {:.1}%, n={n})", pct(*ok), pct(*verb_ok))?;
        }
        Ok(())
    }
}

/// Open-loop accuracy of `policy` on `n` random states of the training distribution.
pub fn step_accuracy(policy: &mut dyn Policy, n: usize, seed: u64) -> Result<StepReport> {
    const VERBS: [(u32, &str); 4] =
        [(vocab::CLICK, "CLICK"), (vocab::TYPE, "TYPE"), (vocab::BACK, "BACK"), (vocab::ANSWER, "ANSWER")];
    let mut rng = Rng::stream(seed, 0x57E9, 0);
    let mut by_verb: Vec<_> = VERBS.iter().map(|&(_, name)| (name, 0, 0, 0)).collect();
    for _ in 0..n {
        let (observation, target) = data::example(&mut rng);
        let out = policy.act(&observation)?;
        if let Some(i) = VERBS.iter().position(|&(v, _)| v == target[0]) {
            let r = &mut by_verb[i];
            r.1 += 1;
            r.2 += (Action::decode(&out) == Action::decode(&target)) as usize;
            r.3 += (out.first() == Some(&target[0])) as usize;
        }
    }
    Ok(StepReport { by_verb })
}
