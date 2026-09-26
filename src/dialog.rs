//! Training data of the unified model ([`crate::unified`]): one model that browses and talks, so
//! every example is an agent observation and the action it should take.
//!
//! | source | share | observation | action |
//! |---|---|---|---|
//! | browser | 35% | page + (sometimes unrelated earlier turns) + task + tool result | the teacher's `CLICK` / `TYPE` / `BACK` / `CALC` / `ANSWER «Сейчас лампа стоит 12 ₽.»` |
//! | dialogue | 35% | a shop page + the earlier turns + the user's line | `ANSWER` + the reply from fiction or a joke |
//! | grammar | 15% | a shop page + a question about a word | `ANSWER` + its form(s): «Как будет «стол» во множественном числе?» → «столы» |
//! | text | 15% | `<text>` + the start of a sentence | its continuation (no verb) |
//!
//! The language sources come from the «ru20k» dataset (`scripts/fetch_ru20k.sh`,
//! `scripts/build_ru20k.py`): ~910 000 dialogue excerpts from fiction and jokes, example
//! sentences for the 20 000 most frequent Russian words (up to five per word, plus UD Russian),
//! and 37 000 grammar questions generated from the paradigms of openrussian.org. Dialogue
//! turns are placed exactly where the agent sees a conversation at inference (`<user>` /
//! `<bot>` before the current `<goal>` line, see [`crate::browser::obs`]), on top of a page of the
//! shop, so talking and browsing share one observation format and one output format.

use std::path::Path;

use candle_core::{bail, Error, Result};

use crate::browser::action::Action;
use crate::browser::data as bdata;
use crate::browser::expert;
use crate::browser::goal::Goal;
use crate::browser::goal::Split;
use crate::browser::obs::{self, Entry, Layout, Note, Turn};
use crate::browser::world::{url_encode, CATALOG_PAGES, ITEMS};
use crate::browser::PageSnapshot;
use crate::kernels::rng::Rng;
use crate::text::{self, fragment, END, PAD, TEXT};

/// Where an example comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Browser,
    Dialog,
    Grammar,
    Text,
    /// Reasoning chains of the school dataset (second observation format only).
    School,
}

impl Source {
    pub const ALL: [Source; 5] = [Self::Browser, Self::Dialog, Self::Grammar, Self::Text, Self::School];

    pub fn name(self) -> &'static str {
        match self {
            Self::Browser => "browser",
            Self::Dialog => "dialogue",
            Self::Grammar => "grammar",
            Self::Text => "text",
            Self::School => "school",
        }
    }
}

/// Share of each source in training batches (first observation format).
pub const MIX: [(Source, f64); 4] =
    [(Source::Browser, 0.35), (Source::Dialog, 0.35), (Source::Grammar, 0.15), (Source::Text, 0.15)];

/// Share of each source with the scratchpad format (docs/scaling.md).
pub const MIX_V2: [(Source, f64); 5] = [
    (Source::School, 0.35),
    (Source::Browser, 0.25),
    (Source::Dialog, 0.25),
    (Source::Grammar, 0.05),
    (Source::Text, 0.10),
];

/// The mixture for an observation format.
pub fn mix(layout: &Layout) -> &'static [(Source, f64)] {
    if layout.scratchpad {
        &MIX_V2
    } else {
        &MIX
    }
}

/// A step of a school reasoning chain.
#[derive(Debug, Clone)]
pub struct ChainStep {
    /// `THINK`, `CALC`, `LOOKUP` or `ANSWER`.
    pub act: String,
    pub text: String,
    /// The tool's reply (`CALC`, `LOOKUP`).
    pub result: Option<String>,
}

impl ChainStep {
    /// The step as the agent's action.
    pub fn action(&self) -> Option<Action> {
        let text = self.text.clone();
        Some(match self.act.as_str() {
            "THINK" => Action::Think { text },
            "CALC" => Action::Calc { text },
            "LOOKUP" => Action::Lookup { text },
            "ANSWER" => Action::Answer { text },
            _ => return None,
        })
    }

    /// What the step leaves in the scratchpad (nothing for the answer).
    pub fn entry(&self) -> Option<Entry> {
        let r = || self.result.clone().unwrap_or_default();
        match self.act.as_str() {
            "THINK" => Some(Entry::Think(self.text.clone())),
            "CALC" => Some(Entry::Calc(Note { expr: self.text.clone(), result: r() })),
            "LOOKUP" => Some(Entry::Lookup { word: self.text.clone(), entry: r() }),
            _ => None,
        }
    }
}

/// A school task: the question and the chain of steps that solves it (`scripts/build_school.py`).
#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    pub grade: u8,
    pub subject: String,
    pub topic: String,
    pub question: String,
    pub steps: Vec<ChainStep>,
    /// `number` or `text`, and the value the answer must state.
    pub check_type: String,
    pub check: String,
}

impl Chain {
    /// Whether `answer` states the expected value: the number among the answer's numbers, or
    /// the text in it (case and `ё` ignored).
    pub fn accepts(&self, answer: &str) -> bool {
        let norm = |t: &str| t.to_lowercase().replace('ё', "е");
        if self.check_type == "number" {
            let want = crate::tools::calc::run(&self.check);
            crate::browser::goal::numbers(answer).iter().any(|n| want.is_ok() && crate::tools::calc::run(n) == want)
        } else {
            let a = norm(answer);
            norm(&self.check).split('|').all(|part| a.contains(part.trim()))
        }
    }
}

/// A dialogue excerpt: the earlier turns (the last one is the user's line) and the reply.
#[derive(Debug, Clone)]
pub struct Dialog {
    pub turns: Vec<String>,
    pub reply: String,
}

/// A grammar question about a word.
#[derive(Debug, Clone)]
pub struct Question {
    pub kind: String,
    pub question: String,
    pub answer: String,
}

/// The language part of the data, split into training and held-out parts.
#[derive(Debug, Clone, Default)]
pub struct LanguageData {
    pub dialogs: Vec<Dialog>,
    pub dialogs_valid: Vec<Dialog>,
    pub questions: Vec<Question>,
    /// Questions about held-out lemmas (never trained on).
    pub questions_heldout: Vec<Question>,
    pub sentences: Vec<String>,
    pub sentences_valid: Vec<String>,
    /// School reasoning chains by split (empty unless [`LanguageData::load_school`] was called).
    pub school: Vec<Chain>,
    pub school_valid: Vec<Chain>,
    pub school_test: Vec<Chain>,
    /// Curriculum: school tasks above this grade are not sampled (0 = no limit).
    pub max_grade: u8,
}

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        Error::Msg(format!(
            "{}: {e} (download the sources with scripts/fetch_ru20k.sh and scripts/fetch_ru_corpus.sh, \
             then run scripts/build_ru20k.py)",
            path.display()
        ))
    })
}

impl LanguageData {
    /// Loads `dir/{dialog,qa,sentences}.tsv` (the «ru20k» build) and, if given, the UD Russian
    /// sentences `ud/{train,valid}.txt`.
    pub fn load(dir: &Path, ud: Option<&Path>) -> Result<Self> {
        let mut d = Self::default();
        for line in read(&dir.join("dialog.tsv"))?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 3 {
                continue;
            }
            let dialog = Dialog {
                turns: f[1..f.len() - 1].iter().map(|s| s.to_string()).collect(),
                reply: f[f.len() - 1].into(),
            };
            if f[0] == "valid" {
                d.dialogs_valid.push(dialog)
            } else {
                d.dialogs.push(dialog)
            }
        }
        for line in read(&dir.join("qa.tsv"))?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 4 {
                continue;
            }
            let q = Question { kind: f[1].into(), question: f[2].into(), answer: f[3].into() };
            if f[0] == "heldout" {
                d.questions_heldout.push(q)
            } else {
                d.questions.push(q)
            }
        }
        for line in read(&dir.join("sentences.tsv"))?.lines() {
            if let Some((split, s)) = line.split_once('\t') {
                if split == "valid" {
                    d.sentences_valid.push(s.into())
                } else {
                    d.sentences.push(s.into())
                }
            }
        }
        if let Some(ud) = ud {
            d.sentences.extend(
                read(&ud.join("train.txt"))?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from),
            );
            d.sentences_valid.extend(
                read(&ud.join("valid.txt"))?.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from),
            );
        }
        if d.dialogs.is_empty() || d.questions.is_empty() || d.sentences.is_empty() {
            bail!("{}: incomplete language data", dir.display())
        }
        Ok(d)
    }

    /// Loads the school chains (`chains.jsonl` of `scripts/build_school.py`).
    pub fn load_school(&mut self, path: &Path) -> Result<()> {
        for line in read(path)?.lines() {
            let v: serde_json::Value = serde_json::from_str(line).map_err(candle_core::Error::wrap)?;
            let st = |k: &str| v[k].as_str().unwrap_or_default().to_string();
            let steps = v["steps"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| ChainStep {
                            act: x["act"].as_str().unwrap_or_default().to_string(),
                            text: x["text"].as_str().unwrap_or_default().to_string(),
                            result: x["result"].as_str().map(String::from),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let chain = Chain {
                id: st("id"),
                grade: v["grade"].as_u64().unwrap_or(0) as u8,
                subject: st("subject"),
                topic: st("topic"),
                question: st("question"),
                steps,
                check_type: v["check"]["type"].as_str().unwrap_or("text").to_string(),
                check: v["check"]["value"].as_str().unwrap_or_default().to_string(),
            };
            match v["split"].as_str() {
                Some("valid") => self.school_valid.push(chain),
                Some("test") => self.school_test.push(chain),
                _ => self.school.push(chain),
            }
        }
        if self.school.is_empty() {
            bail!("{}: no school chains", path.display())
        }
        Ok(())
    }

    /// A tiny built-in data set (tests, smoke runs without the downloads).
    pub fn builtin() -> Self {
        let dialog = |t: &[&str], r: &str| Dialog { turns: t.iter().map(|s| s.to_string()).collect(), reply: r.into() };
        let q = |k: &str, q: &str, a: &str| Question { kind: k.into(), question: q.into(), answer: a.into() };
        let dialogs = vec![
            dialog(&["Привет!"], "Привет! Как дела?"),
            dialog(&["Как тебя зовут?"], "Меня зовут Коля."),
            dialog(&["Привет!", "Привет! Как дела?", "Хорошо. А у тебя?"], "Тоже хорошо, спасибо."),
            dialog(&["Ты куда?"], "Домой."),
            dialog(&["Который час?"], "Уже поздно."),
        ];
        let questions = vec![
            q("plural", "Как будет «стол» во множественном числе?", "столы"),
            q("gender", "Какого рода слово «книга»?", "женского рода"),
            q("past", "Прошедшее время глагола «читать»?", "читал, читала, читало, читали"),
        ];
        let sentences = vec![
            "Мама мыла раму.".to_string(),
            "Москва — столица России.".to_string(),
            "Кошка спит на тёплом диване.".to_string(),
        ];
        let step = |act: &str, text: &str, result: Option<&str>| ChainStep {
            act: act.into(),
            text: text.into(),
            result: result.map(String::from),
        };
        let school = vec![Chain {
            id: "m1_story_00001".into(),
            grade: 1,
            subject: "математика".into(),
            topic: "задачи «было — стало»".into(),
            question: "У Маши было 9 яблок. Маша отдала 4 яблока другу. Сколько яблок осталось у Маши?".into(),
            steps: vec![
                step("THINK", "Было 9, 4 отдала — стало меньше, значит вычитаю.", None),
                step("CALC", "9 - 4", Some("5")),
                step("ANSWER", "Ответ: 5 яблок.", None),
            ],
            check_type: "number".into(),
            check: "5".into(),
        }];
        Self {
            dialogs_valid: dialogs[..2].to_vec(),
            dialogs,
            questions_heldout: questions[..1].to_vec(),
            questions,
            sentences_valid: sentences[..1].to_vec(),
            sentences,
            school_valid: school.clone(),
            school_test: school.clone(),
            school,
            max_grade: 0,
        }
    }
}

/// One training example.
#[derive(Debug, Clone)]
pub struct Example {
    pub source: Source,
    /// `layout.len` tokens.
    pub prompt: Vec<u32>,
    /// `layout.action_len` tokens.
    pub answer: Vec<u32>,
}

/// A random source according to the mixture of `layout`.
pub fn source(rng: &mut Rng, layout: &Layout) -> Source {
    let u = rng.uniform();
    let mut acc = 0.0;
    for &(s, p) in mix(layout) {
        acc += p;
        if u < acc {
            return s;
        }
    }
    Source::Text
}

/// A page of the shop the conversation happens on: mostly the start page, otherwise any page.
pub fn background_page(rng: &mut Rng) -> PageSnapshot {
    let seed = rng.below(1 << 30);
    let item = ITEMS[rng.below(ITEMS.len())].nom;
    let path = match rng.uniform() {
        x if x < 0.65 => format!("/w/{seed}/"),
        x if x < 0.8 => format!("/w/{seed}/search?q={}", url_encode(item)),
        x if x < 0.9 => format!("/w/{seed}/item/{}", url_encode(item)),
        _ => format!("/w/{seed}/catalog?page={}", 1 + rng.below(CATALOG_PAGES)),
    };
    bdata::snapshot(&path, None)
}

/// The turns of a dialogue as the agent's history: the last turn is the user's current line
/// (returned separately), the ones before alternate back to front (`… <user> <bot>`).
pub fn dialog_turns(turns: &[String]) -> (Vec<Turn>, String) {
    let (last, earlier) = turns.split_last().expect("a dialogue has at least one turn");
    let k = earlier.len();
    let history = earlier
        .iter()
        .enumerate()
        .map(|(i, t)| if (k - i) % 2 == 1 { Turn::bot(t.clone()) } else { Turn::user(t.clone()) })
        .collect();
    (history, last.clone())
}

/// Some unrelated earlier conversation (a random excerpt, one or two turns).
fn stray_history(rng: &mut Rng, dialogs: &[Dialog]) -> Vec<Turn> {
    if rng.uniform() < 0.5 {
        return agent_history(rng);
    }
    let d = &dialogs[rng.below(dialogs.len())];
    let mut all: Vec<String> = d.turns.clone();
    all.push(d.reply.clone());
    let keep = 1 + rng.below(2).min(all.len() - 1);
    let turns = &all[all.len() - keep..];
    // the agent spoke last
    turns
        .iter()
        .enumerate()
        .map(|(i, t)| if (turns.len() - i) % 2 == 1 { Turn::bot(t.clone()) } else { Turn::user(t.clone()) })
        .collect()
}

/// Earlier exchanges with the agent itself: one to three questions of the browsing families
/// (a phrase, a price, a sum…) with the teacher's answers — what a conversation with the agent
/// looks like after a few lines.
fn agent_history(rng: &mut Rng) -> Vec<Turn> {
    use crate::browser::world::World;
    let mut turns = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let world = World::new(rng.below(1 << 30) as u64);
        let f = bdata::family(rng);
        let g = crate::browser::goal::sample(rng, &world, f, Split::Train);
        let spec = g.spec.expect("sampled goals have a spec");
        turns.push(Turn::user(g.text));
        turns.push(Turn::bot(spec.answer_sentence(&world)));
    }
    turns
}

/// A user's line as people type it: now and then all in lower case or (a short one) in capitals.
fn vary_case(rng: &mut Rng, line: String) -> String {
    match rng.uniform() {
        x if x < 0.08 => line.to_lowercase(),
        x if x < 0.11 && line.chars().count() <= 30 => line.to_uppercase(),
        _ => line,
    }
}

/// The first letter inside «…» in upper case: «книга» → «Книга».
pub fn capitalize_quoted(text: &str) -> String {
    match text.find('«') {
        Some(i) => {
            let (head, tail) = text.split_at(i + '«'.len_utf8());
            let mut c = tail.chars();
            match c.next() {
                Some(f) => format!("{head}{}{}", f.to_uppercase(), c.as_str()),
                None => text.to_string(),
            }
        }
        None => text.to_string(),
    }
}

fn action_tokens(action: &Action, layout: &Layout) -> Option<Vec<u32>> {
    action.encode_len(text::ru(), layout.action_len)
}

fn answer_tokens(text: &str, layout: &Layout) -> Option<Vec<u32>> {
    action_tokens(&Action::Answer { text: text.to_string() }, layout)
}

/// A text-continuation example: `<text>` + the first tokens of a sentence (left-padded), and
/// the next tokens up to `action_len − 1`, then `<end>`.
pub fn continuation(rng: &mut Rng, sentence: &str, layout: &Layout) -> Option<(Vec<u32>, Vec<u32>)> {
    let (n_obs, n_act) = (layout.len, layout.action_len);
    let s = fragment(text::ru(), sentence);
    if s.len() < 2 {
        return None;
    }
    let cut = 1 + rng.below(s.len() - 1);
    let ctx = &s[cut.saturating_sub(n_obs - 1)..cut];
    let mut prompt = vec![PAD; n_obs];
    prompt[n_obs - 1 - ctx.len()] = TEXT;
    prompt[n_obs - ctx.len()..].copy_from_slice(ctx);
    let mut answer = vec![PAD; n_act];
    let rest = &s[cut..];
    let n = rest.len().min(n_act - 1);
    answer[..n].copy_from_slice(&rest[..n]);
    if n == rest.len() {
        answer[n] = END;
    }
    Some((prompt, answer))
}

/// A step of a thinking teacher's episode in the simulator (scratchpad format): the page, the
/// scratchpad so far (thoughts, actions, calculator lines) and the teacher's next step.
fn trajectory_step(rng: &mut Rng, split: Split) -> Option<(Goal, PageSnapshot, Vec<Entry>, Action)> {
    use crate::browser::world::{home_url, World};
    use crate::browser::{sim::SimBrowser, Browser, Role};
    use crate::tools::{Calculator, RustCalc};
    let world = rng.below(1 << 30) as u64;
    let f = bdata::family(rng);
    let goal = crate::browser::goal::sample(rng, &World::new(world), f, split);
    let spec = goal.spec?;
    let mut sim = SimBrowser::new();
    sim.goto(&home_url(crate::browser::sim::SIM_ORIGIN, world)).ok()?;
    let mut entries: Vec<Entry> = Vec::new();
    let mut steps = Vec::new();
    for _ in 0..24 {
        let snap = sim.snapshot().ok()?;
        let note = obs::last_calc(&entries).cloned();
        let action = match spec {
            crate::browser::Spec::Chat(c) => {
                Action::Answer { text: c.replies()[rng.below(c.replies().len())].to_string() }
            }
            _ => expert::act(&spec, &snap, note.as_ref()),
        };
        let thought = matches!(entries.last(), Some(Entry::Think(_)));
        let step = if thought { action.clone() } else { Action::Think { text: expert::explain(&spec, &action) } };
        steps.push((snap.clone(), entries.clone(), step.clone()));
        match &step {
            Action::Think { text } => entries.push(Entry::Think(text.clone())),
            Action::Calc { text } => entries.push(Entry::Calc(Note::of(text, &RustCalc.eval(text)))),
            Action::Answer { .. } => break,
            Action::Click { role, text } => {
                sim.click(snap.find(*role, text)?).ok()?;
                entries.push(Entry::Act(step.clone()));
            }
            Action::Type { text } => {
                sim.type_text(snap.first(Role::Input)?, text).ok()?;
                entries.push(Entry::Act(step.clone()));
            }
            Action::Back => {
                sim.back().ok()?;
                entries.push(Entry::Act(step.clone()));
            }
            Action::Lookup { .. } => return None,
        }
    }
    let (snap, entries, step) = steps.swap_remove(rng.below(steps.len()));
    Some((goal, snap, entries, step))
}

/// A browser state labelled by the teacher. With the scratchpad format the teacher thinks
/// before every action: half of the states have its reason already in the scratchpad (the
/// action follows), half do not (the reason — a `THINK` — is the target).
fn browser_example(rng: &mut Rng, dialogs: &[Dialog], split: Split, layout: &Layout) -> Option<(Vec<u32>, Vec<u32>)> {
    if layout.scratchpad && rng.uniform() < 0.5 {
        // half of the states come from whole teacher episodes (realistic scratchpads)
        let (goal, snap, entries, step) = trajectory_step(rng, split)?;
        let history = if rng.uniform() < 0.25 { stray_history(rng, dialogs) } else { Vec::new() };
        let prompt = obs::encode_with(layout, &snap, &history, &goal.text, &entries);
        return action_tokens(&step, layout).map(|a| (prompt, a));
    }
    let (goal, snapshot, note, action) = bdata::raw(rng, split);
    let spec = goal.spec.expect("sampled goals have a spec");
    let mut entries = Vec::new();
    let mut target = action.clone();
    if let Some(n) = &note {
        if layout.scratchpad {
            entries.push(Entry::Think(expert::explain(&spec, &Action::Calc { text: n.expr.clone() })));
        }
        entries.push(Entry::Calc(n.clone()));
    }
    if layout.scratchpad {
        let reason = expert::explain(&spec, &action);
        if rng.uniform() < 0.5 {
            target = Action::Think { text: reason };
        } else {
            entries.push(Entry::Think(reason));
        }
    }
    let history = if rng.uniform() < 0.25 { stray_history(rng, dialogs) } else { Vec::new() };
    let prompt = obs::encode_with(layout, &snapshot, &history, &goal.text, &entries);
    action_tokens(&target, layout).map(|a| (prompt, a))
}

/// A step of a school reasoning chain: the question, the steps before it in the scratchpad, and
/// the step itself as the target.
fn school_example(rng: &mut Rng, data: &LanguageData, held: bool, layout: &Layout) -> Option<(Vec<u32>, Vec<u32>)> {
    let pool = if held { &data.school_valid } else { &data.school };
    if pool.is_empty() {
        return None;
    }
    let mut c = &pool[rng.below(pool.len())];
    for _ in 0..20 {
        if data.max_grade == 0 || c.grade <= data.max_grade {
            break;
        }
        c = &pool[rng.below(pool.len())];
    }
    let k = rng.below(c.steps.len());
    let entries: Vec<Entry> = c.steps[..k].iter().filter_map(ChainStep::entry).collect();
    let action = c.steps[k].action()?;
    let dialogs = if held { &data.dialogs_valid } else { &data.dialogs };
    let history = if rng.uniform() < 0.15 { stray_history(rng, dialogs) } else { Vec::new() };
    let question = vary_case(rng, c.question.clone());
    let prompt = obs::encode_with(layout, &background_page(rng), &history, &question, &entries);
    action_tokens(&action, layout).map(|a| (prompt, a))
}

/// One example of `source` in `layout`; `split` picks training or held-out data (held-out
/// browser tasks use the held-out wordings).
pub fn example(rng: &mut Rng, data: &LanguageData, source: Source, split: Split, layout: &Layout) -> Example {
    let held = split == Split::HeldOut;
    let dialogs = if held { &data.dialogs_valid } else { &data.dialogs };
    // without school chains (the first format, or no data) their share goes to dialogues
    let source = if source == Source::School && (data.school.is_empty() || !layout.scratchpad) {
        Source::Dialog
    } else {
        source
    };
    loop {
        let made = match source {
            Source::Browser => browser_example(rng, dialogs, split, layout),
            Source::Dialog => {
                let d = &dialogs[rng.below(dialogs.len())];
                let (history, line) = dialog_turns(&d.turns);
                let line = vary_case(rng, line);
                answer_tokens(&d.reply, layout)
                    .map(|a| (obs::encode_with(layout, &background_page(rng), &history, &line, &[]), a))
            }
            Source::Grammar => {
                let qs = if held { &data.questions_heldout } else { &data.questions };
                let q = &qs[rng.below(qs.len())];
                let history = if rng.uniform() < 0.3 { stray_history(rng, dialogs) } else { Vec::new() };
                // the word asked about is sometimes capitalised («Книга»); its forms stay lower-case
                let question = if rng.uniform() < 0.2 { capitalize_quoted(&q.question) } else { q.question.clone() };
                let question = vary_case(rng, question);
                answer_tokens(&q.answer, layout)
                    .map(|a| (obs::encode_with(layout, &background_page(rng), &history, &question, &[]), a))
            }
            Source::Text => {
                let ss = if held { &data.sentences_valid } else { &data.sentences };
                let i = rng.below(ss.len());
                continuation(rng, &ss[i], layout)
            }
            Source::School => school_example(rng, data, held, layout),
        };
        if let Some((prompt, answer)) = made {
            return Example { source, prompt, answer };
        }
    }
}

/// `n` examples mixed according to the mixture of `layout`.
pub fn mixed(rng: &mut Rng, data: &LanguageData, n: usize, split: Split, layout: &Layout) -> Vec<Example> {
    (0..n)
        .map(|_| {
            let s = source(rng, layout);
            example(rng, data, s, split, layout)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{ANSWER, BOT, GOAL, USER};

    #[test]
    fn examples_have_the_agent_format() {
        let data = LanguageData::builtin();
        let mut rng = Rng::new(4);
        for layout in [Layout::V1, Layout::V2] {
            for source in Source::ALL {
                for _ in 0..40 {
                    let e = example(&mut rng, &data, source, Split::Train, &layout);
                    assert_eq!((e.prompt.len(), e.answer.len()), (layout.len, layout.action_len));
                    match e.source {
                        Source::Text => {
                            assert!(e.prompt.contains(&TEXT));
                            assert!(!e.prompt.contains(&GOAL));
                        }
                        Source::Dialog | Source::Grammar => {
                            assert_eq!(e.answer[0], ANSWER);
                            assert!(e.prompt.contains(&GOAL));
                            assert!(Action::decode(&e.answer, text::ru()).is_some());
                        }
                        Source::Browser | Source::School => {
                            assert!(Action::decode(&e.answer, text::ru()).is_some());
                            assert!(layout.scratchpad || source != Source::School);
                        }
                    }
                    // THINK and LOOKUP only exist in the scratchpad format
                    assert!(layout.scratchpad || !e.answer.contains(&text::THINK));
                }
            }
        }
        // a three-turn dialogue: <user> Привет! <bot> Привет! Как дела? <goal> Хорошо. А у тебя?
        let (h, line) = dialog_turns(&data.dialogs[2].turns);
        assert_eq!(line, "Хорошо. А у тебя?");
        assert_eq!(h, vec![Turn::user("Привет!"), Turn::bot("Привет! Как дела?")]);
        let t = obs::history_tokens(&h, text::ru());
        assert_eq!((t[0], t.iter().filter(|&&x| x == BOT).count()), (USER, 1));
        assert_eq!(capitalize_quoted("Какого рода слово «книга»?"), "Какого рода слово «Книга»?");
    }

    #[test]
    fn school_steps_see_the_steps_before_them() {
        let data = LanguageData::builtin();
        let c = &data.school[0];
        assert!(c.accepts("Ответ: 5 яблок.") && !c.accepts("Ответ: 6 яблок."));
        let mut rng = Rng::new(2);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..60 {
            let e = example(&mut rng, &data, Source::School, Split::Train, &Layout::V2);
            let verb = e.answer[0];
            seen.insert(verb);
            // the calculator step sees the thought before it; the answer sees the result
            if verb == text::CALC {
                assert!(e.prompt.contains(&text::THINK));
            }
            if verb == ANSWER {
                let tail = text::ru().decode(&e.prompt);
                assert!(tail.contains("9 - 4 = 5"), "{tail}");
            }
        }
        assert_eq!(seen.len(), 3, "THINK, CALC and ANSWER are all targets");
    }

    #[test]
    fn long_history_keeps_the_newest_turns() {
        let bpe = text::ru();
        let turns: Vec<Turn> = (0..30).map(|i| Turn::user(format!("реплика номер {i} о погоде и планах"))).collect();
        let t = obs::history_tokens(&turns, bpe);
        assert!(t.len() <= obs::HISTORY_MAX && t.len() > obs::HISTORY_MAX / 2);
        assert!(bpe.decode(&t).contains("номер 29"), "{}", bpe.describe(&t));
        assert!(!bpe.decode(&t).contains("номер 1 "), "{}", bpe.describe(&t));
    }
}
