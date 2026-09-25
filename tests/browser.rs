//! Browsing agent: the expert solves the sandbox web, the simulator matches Chromium element
//! for element, and a (barely trained) engine drives both browsers end to end.
//!
//! Chromium tests run when a browser is found (`COG_CHROME`, Playwright's Chromium, or
//! `chromium` / `google-chrome` on `PATH`) and are skipped with a message otherwise.

use candle_core::{Device, Result};
use cog_engine::browser::agent::{self, EnginePolicy, ExpertPolicy};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::vocab::{word_id, Action};
use cog_engine::browser::{obs, Browser, Chrome, Goal, SimBrowser, SiteServer, ACTION_LEN, VOCAB_SIZE};
use cog_engine::data::Task;
use cog_engine::train::Trainer;
use cog_engine::{CogModel, CognitiveEngine, TrainConfig};

fn chrome() -> Option<Chrome> {
    if Chrome::find_executable().is_none() {
        eprintln!("skipping: no Chromium found (set COG_CHROME)");
        return None;
    }
    Some(Chrome::launch_default().expect("Chromium found but failed to start"))
}

#[test]
fn expert_solves_every_task_in_four_steps() -> Result<()> {
    let mut sim = SimBrowser::new();
    let r = agent::evaluate(&mut sim, &mut ExpertPolicy, SIM_ORIGIN, 300, 7, 10)?;
    assert_eq!(r.successes, 300, "{r}");
    assert_eq!(r.steps, 4 * 300);
    assert_eq!(r.failed_actions, 0);
    Ok(())
}

#[test]
fn expert_recovers_from_a_wrong_query_and_a_wrong_page() -> Result<()> {
    let mut sim = SimBrowser::new();
    let goal = Goal::parse("what is the rating of the drone")?;
    sim.goto(&format!("{SIM_ORIGIN}/w/11/search?q=sofa"))?;
    sim.goto(&format!("{SIM_ORIGIN}/w/11/item/sofa"))?;
    let mut policy = ExpertPolicy;
    let mut actions = Vec::new();
    for _ in 0..8 {
        let snap = sim.snapshot()?;
        let a = Action::decode(&agent::Policy::act(&mut policy, &obs::encode(&snap, &goal))?).unwrap();
        actions.push(a);
        match a {
            Action::Back => sim.back()?,
            Action::Type { word } => sim.type_text(
                snap.first(cog_engine::browser::Role::Input).unwrap(),
                cog_engine::browser::vocab::token_str(word),
            )?,
            Action::Click { role, word } => {
                sim.click(snap.find(role, cog_engine::browser::vocab::token_str(word)).unwrap())?
            }
            Action::Answer { .. } => break,
        }
    }
    let drone = word_id("drone").unwrap();
    assert_eq!(actions[0], Action::Back, "{actions:?}"); // wrong product page
    assert_eq!(actions[1], Action::Type { word: drone }); // wrong query on the results page
    assert!(matches!(actions.last(), Some(Action::Answer { .. })), "{actions:?}");
    Ok(())
}

/// Every snapshot along expert episodes (plus typing and history navigation) is identical in
/// Chromium and in the simulator — so a policy trained on the simulator sees the same inputs
/// in the real browser.
#[test]
fn simulator_matches_chromium() -> Result<()> {
    let Some(mut chrome) = chrome() else { return Ok(()) };
    let server = SiteServer::start("127.0.0.1:0")?;
    let mut sim = SimBrowser::new();
    for (world, goal) in agent::sample_tasks(6, 3) {
        chrome.goto(&cog_engine::browser::world::home_url(&server.origin(), world))?;
        sim.goto(&cog_engine::browser::world::home_url(SIM_ORIGIN, world))?;
        let mut script = vec![];
        for step in 0..8 {
            let (c, s) = (chrome.snapshot()?, sim.snapshot()?);
            assert_eq!(c.elements, s.elements, "world {world}, step {step}: {} vs {}", c.url, s.url);
            assert_eq!(c.title, s.title);
            let action = Action::decode(&agent::Policy::act(&mut ExpertPolicy, &obs::encode(&s, &goal))?).unwrap();
            script.push(action);
            let text = |w: u32| cog_engine::browser::vocab::token_str(w);
            match action {
                Action::Click { role, word } => {
                    let id = s.find(role, text(word)).unwrap();
                    chrome.click(id)?;
                    sim.click(id)?;
                }
                Action::Type { word } => {
                    let id = s.first(cog_engine::browser::Role::Input).unwrap();
                    chrome.type_text(id, text(word))?;
                    sim.type_text(id, text(word))?;
                }
                Action::Back => {
                    chrome.back()?;
                    sim.back()?;
                }
                Action::Answer { .. } => {
                    // Also exercise history: back to the results, then back to the search page.
                    for _ in 0..2 {
                        chrome.back()?;
                        sim.back()?;
                        assert_eq!(chrome.snapshot()?.elements, sim.snapshot()?.elements, "after back");
                    }
                    break;
                }
            }
        }
        assert!(matches!(script.last(), Some(Action::Answer { .. })), "{script:?}");
    }
    Ok(())
}

#[test]
fn expert_succeeds_in_chromium() -> Result<()> {
    let Some(mut chrome) = chrome() else { return Ok(()) };
    let server = SiteServer::start("127.0.0.1:0")?;
    let r = agent::evaluate(&mut chrome, &mut ExpertPolicy, &server.origin(), 5, 11, 10)?;
    assert_eq!(r.successes, 5, "{r}");
    Ok(())
}

/// A briefly trained engine produces well-formed observations → actions in both browsers
/// (quality is measured by `cog_engine agent-eval`, not here).
#[test]
fn engine_policy_drives_the_browser() -> Result<()> {
    let mut cfg = cog_engine::browser::engine_config("tiny")?;
    cfg.planner.num_samples = 16;
    cfg.planner.iterations = 2;
    cfg.flow.solver.steps = 4;
    let mut tc = TrainConfig::quick(Task::Browser);
    tc.steps = 30;
    tc.batch_size = 32;
    tc.warmup = 5;
    tc.eval_every = 0;
    let mut trainer = Trainer::new(CogModel::new(cfg, &Device::Cpu)?, tc)?;
    for _ in 0..30 {
        let r = trainer.train_step()?.0;
        assert!(r.total.is_finite());
        assert!(r.probe > 0.0, "the browser config trains the answer probe");
    }
    let mut policy = EnginePolicy::new(CognitiveEngine::from_model(&trainer.model)?);
    let goal = Goal::parse("price of lamp")?;
    let mut sim = SimBrowser::new();
    let ep = agent::run_episode(&mut sim, &mut policy, SIM_ORIGIN, 1, goal, 3, |_| {})?;
    assert!(!ep.steps.is_empty() && ep.steps.len() <= 3);
    assert!(ep
        .steps
        .iter()
        .all(|s| s.output.len() == ACTION_LEN && s.output.iter().all(|&t| (t as usize) < VOCAB_SIZE)));
    if let Some(mut chrome) = chrome() {
        let server = SiteServer::start("127.0.0.1:0")?;
        let ep = agent::run_episode(&mut chrome, &mut policy, &server.origin(), 1, goal, 3, |_| {})?;
        assert!(!ep.steps.is_empty());
    }
    Ok(())
}
