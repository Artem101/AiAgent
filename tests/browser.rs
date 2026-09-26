//! Browsing agent: the teacher solves every task family, the simulator matches Chromium
//! element for element, and a (barely trained) engine drives both browsers end to end.
//!
//! Chromium tests run when a browser is found (`COG_CHROME`, Playwright's Chromium, or
//! `chromium` / `google-chrome` on `PATH`) and are skipped with a message otherwise.

use candle_core::{Device, Result};
use cog_engine::browser::agent::{self, EnginePolicy, ExpertPolicy, Policy};
use cog_engine::browser::goal::{self, Family, Split};
use cog_engine::browser::obs::Note;
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::world::home_url;
use cog_engine::browser::{
    expert, obs, Action, Browser, Chrome, Goal, PageSnapshot, Role, SimBrowser, SiteServer, ACTION_LEN,
};
use cog_engine::data::Task;
use cog_engine::tools::{Calculator, PythonCalc, RustCalc};
use cog_engine::train::Trainer;
use cog_engine::{text, CogModel, CognitiveEngine, TrainConfig};

fn chrome() -> Option<Chrome> {
    if Chrome::find_executable().is_none() {
        eprintln!("skipping: no Chromium found (set COG_CHROME)");
        return None;
    }
    Some(Chrome::launch_default().expect("Chromium found but failed to start"))
}

/// Executes `action` on `snap` in `browser` (a calculation updates `note`); panics on a
/// missing element.
fn apply(browser: &mut dyn Browser, snap: &PageSnapshot, action: &Action, note: &mut Option<Note>) -> Result<()> {
    match action {
        Action::Click { role, text } => browser.click(snap.find(*role, text).expect("element")),
        Action::Type { text } => browser.type_text(snap.first(Role::Input).expect("text field"), text),
        Action::Back => browser.back(),
        Action::Calc { text } => {
            *note = Some(Note::of(text, &RustCalc.eval(text)));
            Ok(())
        }
        Action::Answer { .. } | Action::Think { .. } | Action::Lookup { .. } => Ok(()),
    }
}

#[test]
fn teacher_solves_every_family_and_wording() -> Result<()> {
    let mut sim = SimBrowser::new();
    for split in [Split::Train, Split::HeldOut] {
        let r = agent::evaluate(&mut sim, &mut ExpertPolicy, &mut RustCalc, SIM_ORIGIN, 600, 7, split, 12)?;
        assert_eq!(r.all.successes, 600, "{r}");
        assert_eq!(r.failed_actions, 0, "{r}");
        for (fam, t) in &r.by_family {
            assert!(t.episodes > 25, "{fam:?}: {t:?}");
            let max_steps = match fam {
                Family::Lookup => 4.0,
                Family::Compare => 3.0,
                Family::Filter => 1.0 + 6.0 + 1.0,
                Family::Calc => 2.0,
                Family::Total => 4.0,
                Family::Chat => 1.0,
            };
            assert!(t.steps as f64 / t.episodes as f64 <= max_steps, "{fam:?}: {t:?}");
        }
    }
    Ok(())
}

#[test]
fn teacher_recovers_from_wrong_pages() -> Result<()> {
    let mut sim = SimBrowser::new();
    let goal = Goal {
        spec: goal::recognize("Что дешевле: лампа или стул?"),
        text: "Что дешевле: лампа или стул?".into(),
    };
    let spec = goal.spec.unwrap();
    // start on the catalogue and on a product page of the wrong task
    for start in ["/w/11/catalog?page=3", "/w/11/item/%D0%B4%D1%80%D0%BE%D0%BD"] {
        sim.goto(&format!("{SIM_ORIGIN}{start}"))?;
        let mut actions = Vec::new();
        let mut note = None;
        for _ in 0..8 {
            let snap = sim.snapshot()?;
            let a = expert::act(&spec, &snap, note.as_ref());
            apply(&mut sim, &snap, &a, &mut note)?;
            let done = matches!(a, Action::Answer { .. });
            actions.push(a);
            if done {
                break;
            }
        }
        assert_eq!(actions[0], Action::Click { role: Role::Link, text: "Главная".into() }, "{actions:?}");
        assert_eq!(actions[1], Action::Type { text: "лампа стул".into() });
        assert!(matches!(actions.last(), Some(Action::Answer { .. })), "{actions:?}");
    }
    Ok(())
}

/// «Сколько будет 5+5?»: the teacher calls the calculator, reads the result from its next
/// observation and answers; a mistyped calculation is redone; totals search, then calculate.
#[test]
fn teacher_calculates_with_the_tool() -> Result<()> {
    let mut sim = SimBrowser::new();
    let run =
        |sim: &mut SimBrowser, q: &str, first_note: Option<Note>| -> Result<(Vec<Action>, Vec<[u32; obs::OBS_LEN]>)> {
            let spec = goal::recognize(q).expect("recognised");
            sim.goto(&home_url(SIM_ORIGIN, 5))?;
            let (mut actions, mut seen, mut note) = (Vec::new(), Vec::new(), first_note);
            for _ in 0..8 {
                let snap = sim.snapshot()?;
                seen.push(obs::encode(&snap, q, note.as_ref()));
                let a = expert::act(&spec, &snap, note.as_ref());
                apply(sim, &snap, &a, &mut note)?;
                let done = matches!(a, Action::Answer { .. });
                actions.push(a);
                if done {
                    break;
                }
            }
            Ok((actions, seen))
        };
    let (acts, seen) = run(&mut sim, "Сколько будет 5+5?", None)?;
    assert_eq!(acts, [Action::Calc { text: "5+5".into() }, Action::Answer { text: "5+5 = 10".into() }]);
    // the result is in the second observation, after the question
    let tail = text::fragment(text::ru(), "5+5 = 10");
    assert!(seen[1].ends_with(&tail), "{}", obs::describe(&seen[1]));
    let (acts, _) = run(&mut sim, "Умножь 12 на 3.", Some(Note::of("12 * 4", &Ok("48".into()))))?;
    assert_eq!(acts, [Action::Calc { text: "12 * 3".into() }, Action::Answer { text: "12 * 3 = 36".into() }]);
    let (acts, _) = run(&mut sim, "Сколько стоят вместе лампа и стул?", None)?;
    assert_eq!(acts[0], Action::Type { text: "лампа стул".into() });
    assert!(matches!(&acts[2], Action::Calc { text } if text.contains('+')), "{acts:?}");
    let w = cog_engine::browser::World::new(5);
    assert_eq!(acts[3], Action::Answer { text: format!("Вместе {} ₽.", w.price[0] + w.price[1]) });
    Ok(())
}

#[test]
fn python_calculator_runs_in_the_episode() -> Result<()> {
    let Ok(mut py) = PythonCalc::start() else {
        eprintln!("skipping: python3 not found");
        return Ok(());
    };
    let goal = Goal {
        spec: goal::recognize("Сколько будет 7 умножить на 8?"),
        text: "Сколько будет 7 умножить на 8?".into(),
    };
    let ep = agent::run_episode(&mut SimBrowser::new(), &mut ExpertPolicy, &mut py, SIM_ORIGIN, 1, &goal, 4, |_| {})?;
    let (note, by) = ep.steps[0].tool.clone().expect("a calculator call");
    assert_eq!((note.expr.as_str(), note.result.as_str(), by.as_str()), ("7 * 8", "56", "python"));
    assert_eq!(ep.answer.as_deref(), Some("7 * 8 = 56"));
    assert_eq!(ep.success(), Some(true));
    Ok(())
}

/// Every snapshot along teacher episodes of all families (typing, catalogue paging, history
/// navigation, Cyrillic URLs) is identical in Chromium and in the simulator — so a policy
/// trained on the simulator sees the same inputs in the real browser.
#[test]
fn simulator_matches_chromium() -> Result<()> {
    let Some(mut chrome) = chrome() else { return Ok(()) };
    let server = SiteServer::start("127.0.0.1:0")?;
    let mut sim = SimBrowser::new();
    let mut seen = std::collections::HashSet::new();
    for (world, goal) in agent::sample_tasks(24, 3, Split::Train) {
        let spec = goal.spec.unwrap();
        seen.insert(spec.family());
        chrome.goto(&home_url(&server.origin(), world))?;
        sim.goto(&home_url(SIM_ORIGIN, world))?;
        let mut script = vec![];
        let mut note = None;
        for step in 0..12 {
            let (c, s) = (chrome.snapshot()?, sim.snapshot()?);
            assert_eq!(c.elements, s.elements, "world {world}, step {step}: {} vs {}", c.url, s.url);
            assert_eq!(c.title, s.title);
            assert_eq!(obs::encode(&c, &goal.text, note.as_ref()), obs::encode(&s, &goal.text, note.as_ref()));
            let action = expert::act(&spec, &s, note.as_ref());
            apply(&mut chrome, &c, &action, &mut note.clone())?;
            apply(&mut sim, &s, &action, &mut note)?;
            let done = matches!(action, Action::Answer { .. });
            script.push(action);
            if done {
                // Also exercise history: two steps back.
                for _ in 0..2 {
                    chrome.back()?;
                    sim.back()?;
                    assert_eq!(chrome.snapshot()?.elements, sim.snapshot()?.elements, "after back");
                }
                break;
            }
        }
        assert!(matches!(script.last(), Some(Action::Answer { .. })), "{script:?}");
    }
    assert!(seen.len() >= 5, "task families exercised: {seen:?}");
    Ok(())
}

#[test]
fn teacher_succeeds_in_chromium() -> Result<()> {
    let Some(mut chrome) = chrome() else { return Ok(()) };
    let server = SiteServer::start("127.0.0.1:0")?;
    let r =
        agent::evaluate(&mut chrome, &mut ExpertPolicy, &mut RustCalc, &server.origin(), 12, 11, Split::HeldOut, 12)?;
    assert_eq!(r.all.successes, 12, "{r}");
    Ok(())
}

/// A briefly trained engine produces well-formed observations → actions in both browsers and
/// reports its latent reasoning (quality is measured by `cog_engine agent-eval`, not here).
#[test]
fn engine_policy_drives_the_browser() -> Result<()> {
    let mut cfg = cog_engine::browser::engine_config("tiny")?;
    cfg.planner.num_samples = 16;
    cfg.planner.iterations = 2;
    cfg.flow.solver.steps = 4;
    let mut tc = TrainConfig::quick(Task::Browser);
    tc.steps = 20;
    tc.batch_size = 16;
    tc.warmup = 5;
    tc.eval_every = 0;
    let mut trainer = Trainer::new(CogModel::new(cfg, &Device::Cpu)?, tc)?;
    for _ in 0..20 {
        let r = trainer.train_step()?.0;
        assert!(r.total.is_finite());
        assert!(r.probe > 0.0, "the browser config trains the answer probe");
    }
    let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&trainer.model)?);
    let goal = Goal {
        spec: goal::recognize("Сколько стоит лампа?"), text: "Сколько стоит лампа?".into()
    };
    let mut sim = SimBrowser::new();
    let ep = agent::run_episode(&mut sim, &mut policy, &mut RustCalc, SIM_ORIGIN, 1, &goal, 3, |_| {})?;
    assert!(!ep.steps.is_empty() && ep.steps.len() <= 3);
    let vocab = text::ru().vocab_size() as u32;
    for s in &ep.steps {
        assert_eq!(s.output.len(), ACTION_LEN);
        assert!(s.output.iter().all(|&t| t < vocab));
        assert!(s.plan.is_some());
    }
    assert!(policy.last_generation().is_some());
    if let Some(mut chrome) = chrome() {
        let server = SiteServer::start("127.0.0.1:0")?;
        let ep = agent::run_episode(&mut chrome, &mut policy, &mut RustCalc, &server.origin(), 1, &goal, 3, |_| {})?;
        assert!(!ep.steps.is_empty());
    }
    Ok(())
}
