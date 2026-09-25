//! The browsing agent: observe → think → act, until the policy answers or runs out of steps.
//!
//! Actions go to the browser (click, type, back) or to the calculator tool (`CALC`); the
//! calculator's result stays in every following observation (see [`obs::Note`]).

use std::time::{Duration, Instant};

use candle_core::Result;

use super::action::Action;
use super::goal::{self, Family, Goal, Split};
use super::obs::{self, Note, OBS_LEN};
use super::world::{self, World};
use super::{data, expert, Browser, PageSnapshot, Role};
use crate::kernels::rng::Rng;
use crate::pipeline::{ActionDecoder, CognitiveEngine, Generation, Reasoning};
use crate::text;
use crate::tools::Calculator;

/// Maps an observation to action tokens.
pub trait Policy {
    /// `goal`, `snapshot` and `note` are given to privileged policies (the teacher); the engine
    /// reads only `observation`, which already contains all of them in token form.
    fn act(
        &mut self,
        observation: &[u32],
        goal: &Goal,
        snapshot: &PageSnapshot,
        note: Option<&Note>,
    ) -> Result<Vec<u32>>;
    /// Diagnostics of the last decision (the engine's latent reasoning).
    fn last_generation(&self) -> Option<Generation> {
        None
    }
    /// Full trace of the last latent reasoning, when the policy records it.
    fn last_reasoning(&self) -> Option<Reasoning> {
        None
    }
}

/// The scripted teacher ([`expert::act`]); needs the structured goal.
pub struct ExpertPolicy;

impl Policy for ExpertPolicy {
    fn act(
        &mut self,
        _observation: &[u32],
        goal: &Goal,
        snapshot: &PageSnapshot,
        note: Option<&Note>,
    ) -> Result<Vec<u32>> {
        let Some(spec) = &goal.spec else { candle_core::bail!("the teacher needs a recognised question") };
        Ok(expert::act(spec, snapshot, note).encode(text::ru()).map(|a| a.to_vec()).unwrap_or_default())
    }
}

/// The trained engine: every step is one `TTT → latent reasoning → CFM` generation.
pub struct EnginePolicy {
    pub engine: CognitiveEngine,
    seed: u64,
    last: Option<Generation>,
    /// Record the full reasoning trace of every decision (allocates; for inspection).
    trace: bool,
    reasoning: Option<Reasoning>,
    /// The previous observation and action, and the actions that left this observation
    /// unchanged (not repeated: the agent takes its next hypothesis instead).
    previous: Option<(Vec<u32>, Option<Action>)>,
    no_effect: Vec<Action>,
}

impl EnginePolicy {
    pub fn new(engine: CognitiveEngine) -> Self {
        Self { engine, seed: 0, last: None, trace: false, reasoning: None, previous: None, no_effect: Vec::new() }
    }

    /// Also keep the decoded tree of hypotheses and chain of thoughts of each decision.
    pub fn with_trace(mut self) -> Self {
        self.trace = true;
        self
    }
}

impl Policy for EnginePolicy {
    fn act(
        &mut self,
        observation: &[u32],
        _goal: &Goal,
        _snapshot: &PageSnapshot,
        _note: Option<&Note>,
    ) -> Result<Vec<u32>> {
        self.seed += 1;
        let (mut out, g) = self.engine.generate(observation, self.seed)?;
        self.last = Some(g);
        self.reasoning = if self.trace { Some(self.engine.last_reasoning(g.plan)?) } else { None };
        // An action after which the agent sees exactly the same observation had no effect (a
        // click on a missing element, retyping the same query): it is not repeated.
        match self.previous.take() {
            Some((obs, Some(action))) if obs == observation => self.no_effect.push(action),
            Some((obs, _)) if obs == observation => {}
            _ => self.no_effect.clear(),
        }
        // The thoughts proposed alternatives: if the winner is not a well-formed action, or one
        // that had no effect here, act on the best proposal that is well-formed and new
        // (hypotheses that decode into nonsense or into a dead end are pruned).
        let usable = |t: &[u32]| Action::decode(t, text::ru()).filter(|a| !self.no_effect.contains(a));
        if self.engine.decoder() == ActionDecoder::ProbeConsensus && usable(&out).is_none() {
            if let Some((_, p)) = self.engine.ranked_proposals().into_iter().find(|(_, p)| usable(p).is_some()) {
                out = p;
            }
        }
        self.previous = Some((observation.to_vec(), Action::decode(&out, text::ru())));
        Ok(out)
    }

    fn last_generation(&self) -> Option<Generation> {
        self.last
    }

    fn last_reasoning(&self) -> Option<Reasoning> {
        self.reasoning.clone()
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
    /// What the calculator returned for a `CALC` action, and which one answered.
    pub tool: Option<(Note, String)>,
    /// Time the policy took to decide.
    pub think_time: Duration,
    /// Planner statistics and timings of this step (if the policy is the engine).
    pub plan: Option<Generation>,
    /// Decoded latent reasoning (if the policy records it, see [`EnginePolicy::with_trace`]).
    pub thoughts: Option<Reasoning>,
}

/// A finished episode.
#[derive(Debug, Clone)]
pub struct Episode {
    pub world: u64,
    pub goal: Goal,
    pub steps: Vec<Step>,
    /// What the agent answered, if it answered.
    pub answer: Option<String>,
}

impl Episode {
    /// Whether the answer is correct (`None` when the question was not recognised).
    pub fn success(&self) -> Option<bool> {
        let spec = self.goal.spec?;
        Some(self.answer.as_deref().is_some_and(|a| spec.accepts(&World::new(self.world), a)))
    }

    /// The expected answer, when the question is recognised.
    pub fn expected(&self) -> Option<String> {
        self.goal.spec.map(|s| s.expected(&World::new(self.world)))
    }
}

/// Carries out a browser `action` on the page of `snap`. `Ok(Some(reason))` is a policy
/// mistake (e.g. clicking text that is not on the page); `Err` is a browser failure.
fn execute(browser: &mut dyn Browser, snap: &PageSnapshot, action: &Action) -> Result<Option<String>> {
    match action {
        Action::Click { role, text } => match snap.find(*role, text) {
            Some(id) => browser.click(id).map(|_| None),
            None => Ok(Some(format!("no {} «{text}» on the page", role.name()))),
        },
        Action::Type { text } => match snap.first(Role::Input) {
            Some(id) => browser.type_text(id, text).map(|_| None),
            None => Ok(Some("no text field on the page".into())),
        },
        Action::Back => browser.back().map(|_| None),
        Action::Answer { .. } | Action::Calc { .. } => Ok(None),
    }
}

/// Opens the search page of `world` and lets `policy` browse (and calculate with `calc`)
/// until it answers or `max_steps` actions were taken. `on_step` sees every step as it
/// happens.
#[allow(clippy::too_many_arguments)]
pub fn run_episode(
    browser: &mut dyn Browser,
    policy: &mut dyn Policy,
    calc: &mut dyn Calculator,
    origin: &str,
    world: u64,
    goal: &Goal,
    max_steps: usize,
    mut on_step: impl FnMut(&Step),
) -> Result<Episode> {
    browser.goto(&world::home_url(origin, world))?;
    let mut steps = Vec::new();
    let mut answer = None;
    let mut note: Option<Note> = None;
    for _ in 0..max_steps {
        let snap = browser.snapshot()?;
        let observation = obs::encode(&snap, &goal.text, note.as_ref());
        let t0 = Instant::now();
        let output = policy.act(&observation, goal, &snap, note.as_ref())?;
        let think_time = t0.elapsed();
        let action = Action::decode(&output, text::ru());
        let mut tool = None;
        let error = match &action {
            Some(Action::Calc { text }) => {
                let reply = calc.eval(text);
                let n = Note::of(text, &reply);
                note = Some(n.clone());
                tool = Some((n, calc.name().to_string()));
                reply.err().map(|e| format!("calculator: {e}"))
            }
            Some(a) => execute(browser, &snap, a)?,
            None => Some(format!("malformed action [{}]", obs::describe(&output))),
        };
        let step = Step {
            url: snap.url,
            observation,
            output,
            action,
            error,
            tool,
            think_time,
            plan: policy.last_generation(),
            thoughts: policy.last_reasoning(),
        };
        on_step(&step);
        let done = matches!(&step.action, Some(Action::Answer { .. }));
        if let Some(Action::Answer { text }) = &step.action {
            answer = Some(text.trim().to_string());
        }
        steps.push(step);
        if done {
            break;
        }
    }
    Ok(Episode { world, goal: goal.clone(), steps, answer })
}

/// Success statistics of one group of episodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tally {
    pub episodes: usize,
    pub successes: usize,
    pub wrong_answers: usize,
    pub steps: usize,
}

impl Tally {
    fn add(&mut self, ep: &Episode) {
        self.episodes += 1;
        let ok = ep.success() == Some(true);
        self.successes += ok as usize;
        self.wrong_answers += (ep.answer.is_some() && !ok) as usize;
        self.steps += ep.steps.len();
    }

    pub fn rate(&self) -> f32 {
        self.successes as f32 / self.episodes.max(1) as f32
    }
}

/// Aggregate over many episodes.
#[derive(Debug, Clone, Default)]
pub struct AgentReport {
    pub all: Tally,
    pub by_family: Vec<(Family, Tally)>,
    /// Steps whose action was malformed or referred to a missing element.
    pub failed_actions: usize,
    pub think_time: Duration,
    pub wall_time: Duration,
    /// Sums over the engine's decisions of the plan energy: greedy chain, after the tree
    /// search (MPPI warm start), final (after MPPI and latent GD); and the decision count.
    pub energy: [f64; 3],
    pub decisions: usize,
}

impl AgentReport {
    pub fn success_rate(&self) -> f32 {
        self.all.rate()
    }
}

impl std::fmt::Display for AgentReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a = &self.all;
        let steps = a.steps.max(1) as f64;
        write!(
            f,
            "success {:.1}% ({}/{}) | wrong answer {} | no answer {} | {:.2} steps/episode | failed actions {:.1}% | \
             think {:.1} ms/step | {:.0} ms/episode",
            100.0 * a.rate(),
            a.successes,
            a.episodes,
            a.wrong_answers,
            a.episodes - a.successes - a.wrong_answers,
            a.steps as f64 / a.episodes.max(1) as f64,
            100.0 * self.failed_actions as f64 / steps,
            1e3 * self.think_time.as_secs_f64() / steps,
            1e3 * self.wall_time.as_secs_f64() / a.episodes.max(1) as f64,
        )?;
        if self.decisions > 0 {
            let k = self.decisions as f64;
            write!(
                f,
                "\n    latent energy per decision: greedy {:.4} → tree {:.4} → final {:.4}",
                self.energy[0] / k,
                self.energy[1] / k,
                self.energy[2] / k
            )?;
        }
        for (fam, t) in &self.by_family {
            write!(
                f,
                "\n    {:<8} {:>5.1}% ({}/{}), {:.2} steps",
                fam.name(),
                100.0 * t.rate(),
                t.successes,
                t.episodes,
                t.steps as f64 / t.episodes.max(1) as f64
            )?;
        }
        Ok(())
    }
}

/// Random `(world, goal)` pairs for evaluation (deterministic in `seed`).
pub fn sample_tasks(n: usize, seed: u64, split: Split) -> Vec<(u64, Goal)> {
    let mut rng = Rng::stream(seed, 0xA6E7, 0);
    (0..n)
        .map(|_| {
            let world = rng.below(1_000_000) as u64;
            let f = data::family(&mut rng);
            (world, goal::sample(&mut rng, &World::new(world), f, split))
        })
        .collect()
}

/// Runs `episodes` random tasks worded with the templates of `split` and reports success.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    browser: &mut dyn Browser,
    policy: &mut dyn Policy,
    calc: &mut dyn Calculator,
    origin: &str,
    episodes: usize,
    seed: u64,
    split: Split,
    max_steps: usize,
) -> Result<AgentReport> {
    let mut r =
        AgentReport { by_family: Family::ALL.iter().map(|&f| (f, Tally::default())).collect(), ..Default::default() };
    let t0 = Instant::now();
    for (world, goal) in sample_tasks(episodes, seed, split) {
        let ep = run_episode(browser, policy, calc, origin, world, &goal, max_steps, |_| {})?;
        r.all.add(&ep);
        let fam = goal.spec.expect("sampled").family();
        if let Some((_, t)) = r.by_family.iter_mut().find(|(f, _)| *f == fam) {
            t.add(&ep);
        }
        r.failed_actions += ep.steps.iter().filter(|s| s.error.is_some()).count();
        r.think_time += ep.steps.iter().map(|s| s.think_time).sum::<Duration>();
        for p in ep.steps.iter().filter_map(|s| s.plan) {
            r.energy[0] += p.plan.greedy_energy as f64;
            r.energy[1] += p.plan.initial_energy as f64;
            r.energy[2] += p.plan.energy as f64;
            r.decisions += 1;
        }
    }
    r.wall_time = t0.elapsed();
    Ok(r)
}

/// Per-action-type accuracy on single browser states (open loop, teacher labels).
#[derive(Debug, Clone, Default)]
pub struct StepReport {
    /// `(teacher verb, states, exact action, right verb)` for CLICK, TYPE, BACK, CALC, ANSWER.
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

/// Open-loop accuracy of `policy` on `n` random states worded with the templates of `split`.
pub fn step_accuracy(policy: &mut dyn Policy, n: usize, seed: u64, split: Split) -> Result<StepReport> {
    use text::{ANSWER, BACK, CALC, CLICK, TYPE};
    const VERBS: [(u32, &str); 5] =
        [(CLICK, "CLICK"), (TYPE, "TYPE"), (BACK, "BACK"), (CALC, "CALC"), (ANSWER, "ANSWER")];
    let bpe = text::ru();
    let mut rng = Rng::stream(seed, 0x57E9, 0);
    let mut by_verb: Vec<_> = VERBS.iter().map(|&(_, name)| (name, 0, 0, 0)).collect();
    for _ in 0..n {
        let data::Labelled { observation, action: target, goal, snapshot, note } = data::labelled(&mut rng, split);
        let out = policy.act(&observation, &goal, &snapshot, note.as_ref())?;
        if let Some(i) = VERBS.iter().position(|&(v, _)| v == target[0]) {
            let r = &mut by_verb[i];
            r.1 += 1;
            let got = Action::decode(&out, bpe);
            r.2 += (got.is_some() && got == Action::decode(&target, bpe)) as usize;
            r.3 += (out.first() == Some(&target[0])) as usize;
        }
    }
    Ok(StepReport { by_verb })
}
