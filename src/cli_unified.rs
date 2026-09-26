//! Commands of the unified model (`train-unified`, `chat`, `unified-eval`).

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Result};

use cog_engine::browser::action::Action;
use cog_engine::browser::agent::{self as browsing, Policy};
use cog_engine::browser::goal::{self, Goal, Split};
use cog_engine::browser::obs::{self, Turn};
use cog_engine::dialog::{self, LanguageData};
use cog_engine::speech::Sampling;
use cog_engine::text;
use cog_engine::unified::{
    grammar_accuracy, validation, UnifiedConfig, UnifiedEngine, UnifiedModel, UnifiedPolicy, UnifiedTrainConfig,
    UnifiedTrainer,
};

use super::{agent_browser, agent_calculator, print_reasoning, Args};

/// The shipped unified model.
pub const SHIPPED: &str = "models/agent.safetensors";

fn language_data(a: &Args) -> Result<LanguageData> {
    let dir = a.get("data", "data/ru20k");
    if dir == "builtin" {
        return Ok(LanguageData::builtin());
    }
    let ud = a.get("ud", "data/ru");
    let t0 = Instant::now();
    let d = LanguageData::load(Path::new(&dir), (ud != "none").then_some(Path::new(&ud)))?;
    println!(
        "data: {} dialogues (+{} valid), {} grammar questions (+{} on held-out lemmas), {} sentences (+{} valid) in {:.1?}",
        d.dialogs.len(),
        d.dialogs_valid.len(),
        d.questions.len(),
        d.questions_heldout.len(),
        d.sentences.len(),
        d.sentences_valid.len(),
        t0.elapsed()
    );
    Ok(d)
}

/// Loads a unified checkpoint as an inference engine (`--search 0` switches the latent search
/// off; `--temperature`, `--top-k` set the sampling of replies).
pub fn load_engine(a: &Args, ckpt: &str) -> Result<UnifiedEngine> {
    let model = UnifiedModel::load(ckpt, &Device::Cpu)?;
    let mut engine = UnifiedEngine::new(model)?;
    engine.search = a.num::<u8>("search", 1)? != 0;
    engine.sampling =
        Sampling { temperature: a.num("temperature", 0.0)?, top_k: a.num("top-k", 40)?, greedy_prefix: 2 };
    engine.seed(a.num("seed", 1)?);
    Ok(engine)
}

/// The unified model as a browsing policy.
pub fn policy(a: &Args, ckpt: &str, trace: bool) -> Result<Box<dyn Policy>> {
    let p = UnifiedPolicy::new(load_engine(a, ckpt)?);
    Ok(Box::new(if trace { p.with_trace() } else { p }))
}

pub fn cmd_train(a: &Args) -> Result<()> {
    let out = a.get("out", SHIPPED);
    let mut cfg = UnifiedConfig::preset(&a.get("preset", "base"))?;
    cfg.engine.seed = a.num("seed", 7)?;
    cfg.speech.d_model = a.num("d-model", cfg.speech.d_model)?;
    cfg.speech.n_layers = a.num("layers", cfg.speech.n_layers)?;
    cfg.speech.n_heads = a.num("heads", cfg.speech.n_heads)?;
    cfg.speech.copy_dim = a.num("copy", cfg.speech.copy_dim)?;
    cfg.teacher_plan = a.num("teacher-plan", cfg.teacher_plan)?;
    let model = UnifiedModel::new(cfg, &Device::Cpu)?;
    if let Some(init) = a.opts.get("init") {
        model.load_weights(init)?;
        println!("initialised from {init}");
    }
    println!("unified model: {} parameters", model.num_params());
    let d = UnifiedTrainConfig::default();
    let tc = UnifiedTrainConfig {
        batch_size: a.num("batch", d.batch_size)?,
        steps: a.num("steps", d.steps)?,
        lr: a.num("lr", d.lr)?,
        min_lr: a.num("min-lr", d.min_lr)?,
        warmup: a.num("warmup", d.warmup)?,
        log_every: a.num("log-every", d.log_every)?,
        ..d
    };
    let data = language_data(a)?;
    let mut tr = UnifiedTrainer::new(model, tc, data)?;
    if a.opts.contains_key("start-step") {
        tr.resume_at(a.num("start-step", 0)?);
    }
    let save_every: usize = a.num("save-every", 500)?;
    let eval_every: usize = a.num("eval-every", 0)?;
    let steps = tr.tc.steps;
    while tr.step() < steps {
        let next = (tr.step() / save_every + 1) * save_every;
        tr.run_until(next, |l| println!("{l}"))?;
        tr.model.save(&out)?;
        println!("saved {out} at step {}", tr.step());
        if eval_every > 0 && (tr.step().is_multiple_of(eval_every) || tr.step() == steps) {
            quick_eval(&tr.model, &tr.data, &out)?;
        }
    }
    Ok(())
}

/// Held-out loss per source and a few replies (during training).
fn quick_eval(model: &UnifiedModel, data: &LanguageData, ckpt: &str) -> Result<()> {
    let v = validation(model, data, 64, 99)?;
    println!("  valid | {v}");
    let mut engine = UnifiedEngine::new(UnifiedModel::load(ckpt, &Device::Cpu)?)?;
    for line in ["Привет!", "Как тебя зовут?", "Сколько будет 7+8?", "Какого рода слово «книга»?"]
    {
        let prompt = obs::encode_dialog(&cog_engine::browser::data::snapshot("/w/1/", None), &[], line, None);
        let (out, _, _) = engine.respond(&prompt, 1)?;
        println!("  {line} → {}", say(&out));
    }
    Ok(())
}

/// An output in readable form.
fn say(tokens: &[u32]) -> String {
    match Action::decode(tokens, text::ru()) {
        Some(Action::Answer { text }) => text,
        Some(a) => a.to_string(),
        None => format!("⟨{}⟩", obs::describe(tokens)),
    }
}

/// Conversation with the agent: every line of the user is one episode in the browser (the
/// agent may search, calculate or just answer); the dialogue so far is in every observation.
pub fn cmd_chat(a: &Args) -> Result<()> {
    let ckpt = a.get("ckpt", SHIPPED);
    let mut a2 = Args { cmd: a.cmd.clone(), opts: a.opts.clone() };
    a2.opts.entry("temperature".into()).or_insert_with(|| "0.7".into());
    let mut policy = UnifiedPolicy::new(load_engine(&a2, &ckpt)?);
    if a.opts.contains_key("think") {
        policy = policy.with_trace();
    }
    let mut calc = agent_calculator(a, "python")?;
    let (mut browser, origin, _server) = agent_browser(a, "sim")?;
    let world: u64 = a.num("world", 42)?;
    let steps = a.num("max-steps", 12)?;
    let think = a.opts.contains_key("think");
    let mut history: Vec<Turn> = Vec::new();
    let scripted: Option<Vec<String>> = a.opts.get("say").map(|s| s.split('|').map(|x| x.trim().to_string()).collect());
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut next = 0;
    println!("Чат с агентом (мир магазина {world}, {origin}). Пустая строка или Ctrl-D — выход.");
    loop {
        let line = match &scripted {
            Some(s) => match s.get(next) {
                Some(l) => {
                    next += 1;
                    println!("\nВы     : {l}");
                    l.clone()
                }
                None => break,
            },
            None => {
                print!("\nВы     : ");
                std::io::stdout().flush().ok();
                match lines.next() {
                    Some(Ok(l)) if !l.trim().is_empty() => l.trim().to_string(),
                    _ => break,
                }
            }
        };
        let goal = Goal { spec: goal::recognize(&line), text: line.clone() };
        let t0 = Instant::now();
        let ep = browsing::run_dialog_episode(
            browser.as_mut(),
            &mut policy,
            calc.as_mut(),
            &origin,
            world,
            &history,
            &goal,
            steps,
            |s| {
                let answered = matches!(s.action, Some(Action::Answer { .. }));
                if think {
                    if let Some(r) = &s.thoughts {
                        print_reasoning(r);
                    }
                }
                if !answered || think {
                    let action = s.action.as_ref().map_or_else(|| say(&s.output), |x| x.to_string());
                    let err = s.error.as_ref().map(|e| format!("  ✗ {e}")).unwrap_or_default();
                    println!("  · {action}{err}");
                    if let Some((note, _)) = &s.tool {
                        println!("    = {} = {}", note.expr, note.result);
                    }
                }
            },
        )?;
        let reply = ep.answer.clone().unwrap_or_else(|| "…".to_string());
        let check = match ep.success() {
            Some(true) => " ✓",
            Some(false) => " ✗",
            None => "",
        };
        println!("Агент  : {reply}{check}   ({} шаг., {:.1} с)", ep.steps.len(), t0.elapsed().as_secs_f64());
        history.push(Turn::user(line));
        history.push(Turn::bot(reply));
    }
    Ok(())
}

/// Held-out loss per source, grammar accuracy, replies to a fixed set of lines, and the agent's
/// success on browsing tasks.
pub fn cmd_eval(a: &Args) -> Result<()> {
    let ckpt = a.get("ckpt", SHIPPED);
    let model = UnifiedModel::load(&ckpt, &Device::Cpu)?;
    println!("{ckpt}: {} parameters", model.num_params());
    let data = language_data(a)?;
    let n = a.num("n", 256)?;
    println!("held-out loss per token (accuracy of the most likely token):\n  {}", validation(&model, &data, n, 99)?);
    let mut engine = UnifiedEngine::new(model)?;
    engine.search = a.num::<u8>("search", 1)? != 0;
    let q = a.num("grammar", 300)?;
    if q > 0 {
        for (name, qs) in [("held-out lemmas", &data.questions_heldout), ("training lemmas", &data.questions)] {
            let by_kind = grammar_accuracy(&mut engine, qs, q, 5)?;
            let (n, ok) = by_kind.iter().fold((0, 0), |(n, ok), k| (n + k.1, ok + k.2));
            let kinds: Vec<String> =
                by_kind.iter().map(|(k, n, ok)| format!("{k} {:.0}%", 100.0 * *ok as f64 / *n as f64)).collect();
            println!("grammar, {name}: {:.1}% exact ({ok}/{n}) | {}", 100.0 * ok as f64 / n as f64, kinds.join(", "));
        }
    }
    let lines = [
        "Привет!",
        "Как дела?",
        "Кто ты?",
        "Что ты умеешь?",
        "Как тебя зовут?",
        "Ты где сейчас?",
        "Что ты будешь делать завтра?",
        "Мне грустно.",
        "Спасибо!",
        "Какое множественное число у слова «друг»?",
        "Как будет «читать» в прошедшем времени?",
    ];
    engine.sampling = Sampling { temperature: a.num("temperature", 0.0)?, top_k: 40, greedy_prefix: 2 };
    println!("replies (on the start page, no history):");
    for line in lines {
        let prompt = obs::encode_dialog(&cog_engine::browser::data::snapshot("/w/1/", None), &[], line, None);
        let (out, _, _) = engine.respond(&prompt, 1)?;
        println!("  {line:<44} → {}", say(&out));
    }
    let dialogs = a.num("dialogs", 8)?;
    if dialogs > 0 {
        println!("held-out dialogue excerpts (context → reference | model):");
        let mut rng = cog_engine::kernels::rng::Rng::new(11);
        for _ in 0..dialogs {
            let d = &data.dialogs_valid[rng.below(data.dialogs_valid.len())];
            let (history, line) = dialog::dialog_turns(&d.turns);
            let prompt = obs::encode_dialog(&cog_engine::browser::data::snapshot("/w/1/", None), &history, &line, None);
            let (out, _, _) = engine.respond(&prompt, 1)?;
            println!("  {} → {} | {}", d.turns.join(" / "), d.reply, say(&out));
        }
    }
    let episodes = a.num("episodes", 200)?;
    if episodes > 0 {
        engine.sampling = Sampling::GREEDY;
        let mut policy = UnifiedPolicy::new(engine);
        let mut calc = agent_calculator(a, "rust")?;
        let (mut browser, origin, _server) = agent_browser(a, "sim")?;
        for (name, split) in [("train wordings", Split::Train), ("held-out wordings", Split::HeldOut)] {
            let r = browsing::evaluate(browser.as_mut(), &mut policy, calc.as_mut(), &origin, episodes, 1, split, 12)?;
            println!("agent | {} | {name}: {r}", a.get("browser", "sim"));
        }
    }
    Ok(())
}
