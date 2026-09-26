//! The unified model end to end: a briefly trained tiny model browses, calculates and talks in
//! the simulator (and in Chromium when one is found), with the conversation in its observation.

use candle_core::{Device, Result};
use cog_engine::browser::agent::{self, Policy};
use cog_engine::browser::goal::{self, Goal};
use cog_engine::browser::obs::{self, Turn};
use cog_engine::browser::sim::SIM_ORIGIN;
use cog_engine::browser::{Chrome, SimBrowser, SiteServer, ACTION_LEN};
use cog_engine::dialog::LanguageData;
use cog_engine::text;
use cog_engine::tools::RustCalc;
use cog_engine::unified::{
    UnifiedConfig, UnifiedEngine, UnifiedModel, UnifiedPolicy, UnifiedTrainConfig, UnifiedTrainer,
};

fn trained_tiny() -> Result<UnifiedModel> {
    let mut cfg = UnifiedConfig::preset("tiny")?;
    cfg.engine.planner.num_samples = 16;
    cfg.engine.planner.iterations = 2;
    let tc = UnifiedTrainConfig { batch_size: 8, steps: 10, warmup: 2, log_every: 5, workers: 2, ..Default::default() };
    let mut tr = UnifiedTrainer::new(UnifiedModel::new(cfg, &Device::Cpu)?, tc, LanguageData::builtin())?;
    tr.run_until(10, |l| assert!(!l.contains("NaN"), "{l}"))?;
    Ok(tr.model)
}

#[test]
fn one_model_browses_and_talks_with_history() -> Result<()> {
    let model = trained_tiny()?;
    // checkpoints round-trip with their configuration
    let path = std::env::temp_dir().join(format!("cog_unified_{}.safetensors", std::process::id()));
    model.save(&path)?;
    assert!(UnifiedModel::is_checkpoint(&path));
    let loaded = UnifiedModel::load(&path, &Device::Cpu)?;
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(format!("{}.cfg", path.display())).ok();
    assert_eq!(loaded.num_params(), model.num_params());

    let mut policy = UnifiedPolicy::new(UnifiedEngine::new(loaded)?).with_trace();
    let mut sim = SimBrowser::new();
    let history = vec![Turn::user("Привет!"), Turn::bot("Привет! Чем помочь?")];
    let vocab = text::ru().vocab_size() as u32;
    for q in ["Сколько стоит лампа?", "Сколько будет 5+5?", "Как дела?"] {
        let g = Goal { spec: goal::recognize(q), text: q.into() };
        let ep =
            agent::run_dialog_episode(&mut sim, &mut policy, &mut RustCalc, SIM_ORIGIN, 1, &history, &g, 3, |_| {})?;
        assert!(!ep.steps.is_empty() && ep.steps.len() <= 3);
        for s in &ep.steps {
            assert_eq!(s.output.len(), ACTION_LEN);
            assert!(s.output.iter().all(|&t| t < vocab));
            // the conversation is part of what the model sees
            assert!(s.observation.contains(&text::USER) && s.observation.contains(&text::BOT));
            let r = s.thoughts.as_ref().expect("traced reasoning");
            assert!(r.stats.tree_nodes > 0, "the latent search ran");
        }
    }
    assert!(policy.last_generation().is_some());
    if Chrome::find_executable().is_some() {
        let mut chrome = Chrome::launch_default()?;
        let server = SiteServer::start("127.0.0.1:0")?;
        let g = Goal {
            spec: goal::recognize("Сколько стоит лампа?"), text: "Сколько стоит лампа?".into()
        };
        let ep = agent::run_episode(&mut chrome, &mut policy, &mut RustCalc, &server.origin(), 1, &g, 2, |_| {})?;
        assert!(!ep.steps.is_empty());
    } else {
        eprintln!("skipping Chromium: no browser found (set COG_CHROME)");
    }
    Ok(())
}

#[test]
fn observation_keeps_page_history_question_and_tool_result() {
    let snap = cog_engine::browser::data::snapshot("/w/3/", None);
    let note = obs::Note { expr: "5+5".into(), result: "10".into() };
    let history = vec![Turn::user("Как тебя зовут?"), Turn::bot("Я ваш помощник.")];
    let o = obs::encode_dialog(&snap, &history, "Сколько будет 5+5?", Some(&note));
    let bpe = text::ru();
    let pos = |t: u32| o.iter().position(|&x| x == t).unwrap();
    // page … <user> … <bot> … <goal> … CALC …, newest last
    assert!(pos(text::USER) < pos(text::BOT) && pos(text::BOT) < pos(text::GOAL) && pos(text::GOAL) < pos(text::CALC));
    assert!(bpe.decode(&o).contains("Найти"), "the page is still there: {}", bpe.describe(&o));
    assert_eq!(obs::encode_dialog(&snap, &[], "Привет", None), obs::encode(&snap, "Привет", None));
}

/// The second observation format: the thinking teacher solves every family with a `THINK`
/// before each action, and a tiny scratchpad model runs school tasks with the calculator and
/// the dictionary.
#[test]
fn scratchpad_format_thinks_calculates_and_looks_up() -> Result<()> {
    use cog_engine::browser::agent::ThinkingExpertPolicy;
    use cog_engine::browser::goal::Split;
    use cog_engine::browser::obs::{Entry, Layout};
    use cog_engine::browser::Action;
    let mut sim = SimBrowser::new();
    let r = agent::evaluate(&mut sim, &mut ThinkingExpertPolicy, &mut RustCalc, SIM_ORIGIN, 120, 3, Split::Train, 12)?;
    assert_eq!(r.all.successes, 120, "{r}");
    // every action follows a THINK
    let g = Goal {
        spec: goal::recognize("Сколько стоят вместе лампа и стул?"),
        text: "Сколько стоят вместе лампа и стул?".into(),
    };
    let ep = agent::run_episode(&mut sim, &mut ThinkingExpertPolicy, &mut RustCalc, SIM_ORIGIN, 4, &g, 12, |_| {})?;
    assert_eq!(ep.success(), Some(true));
    for pair in ep.steps.chunks(2) {
        assert!(matches!(pair[0].action, Some(Action::Think { .. })), "{:?}", pair[0].action);
        assert!(!matches!(pair[1].action, Some(Action::Think { .. })));
    }
    let calc = ep.steps.iter().find(|s| matches!(s.action, Some(Action::Calc { .. }))).unwrap();
    assert!(matches!(calc.added, Some(Entry::Calc(_))));
    // the scratchpad of a later step holds the thought and the calculator's line
    let last = ep.steps.last().unwrap();
    assert!(last.observation.contains(&text::THINK) && last.observation.contains(&text::CALC));
    assert_eq!(last.observation.len(), Layout::V2.len);

    // a tiny model in the scratchpad format, trained a few steps on built-in data (with a school
    // chain): its episodes run and its steps have the V2 shapes
    let mut cfg = UnifiedConfig::preset("tiny2")?;
    cfg.engine.planner.num_samples = 16;
    cfg.engine.planner.iterations = 2;
    let tc = UnifiedTrainConfig { batch_size: 8, steps: 6, warmup: 2, log_every: 3, workers: 2, ..Default::default() };
    let mut tr = UnifiedTrainer::new(UnifiedModel::new(cfg, &Device::Cpu)?, tc, LanguageData::builtin())?;
    let mut lines = Vec::new();
    tr.run_until(6, |l| lines.push(l.to_string()))?;
    assert!(lines.iter().any(|l| l.contains("school")), "{lines:?}");
    let mut policy = UnifiedPolicy::new(UnifiedEngine::new(tr.model)?);
    let g = Goal {
        spec: None,
        text: "У Маши было 9 яблок. Маша отдала 4 яблока другу. Сколько яблок осталось?".into(),
    };
    let ep = agent::run_episode(&mut sim, &mut policy, &mut RustCalc, SIM_ORIGIN, 1, &g, 3, |_| {})?;
    assert!(!ep.steps.is_empty());
    for s in &ep.steps {
        assert_eq!((s.observation.len(), s.output.len()), (Layout::V2.len, Layout::V2.action_len));
    }
    Ok(())
}

#[test]
fn dictionary_tool_answers_look_ups() {
    use cog_engine::tools::dictionary::{Dictionary, NOT_FOUND};
    let d = Dictionary::from_entries([("книга", "книга — существительное, ж. р.")]);
    assert_eq!(d.lookup("Книга"), "книга — существительное, ж. р.");
    assert_eq!(d.lookup("абракадабра"), NOT_FOUND);
}
