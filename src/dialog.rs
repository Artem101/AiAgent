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

use crate::browser::action::{Action, ACTION_LEN};
use crate::browser::data as bdata;
use crate::browser::goal::Split;
use crate::browser::obs::{self, Turn, OBS_LEN};
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
}

impl Source {
    pub const ALL: [Source; 4] = [Self::Browser, Self::Dialog, Self::Grammar, Self::Text];

    pub fn name(self) -> &'static str {
        match self {
            Self::Browser => "browser",
            Self::Dialog => "dialogue",
            Self::Grammar => "grammar",
            Self::Text => "text",
        }
    }
}

/// Share of each source in training batches.
pub const MIX: [(Source, f64); 4] =
    [(Source::Browser, 0.35), (Source::Dialog, 0.35), (Source::Grammar, 0.15), (Source::Text, 0.15)];

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
        Self {
            dialogs_valid: dialogs[..2].to_vec(),
            dialogs,
            questions_heldout: questions[..1].to_vec(),
            questions,
            sentences_valid: sentences[..1].to_vec(),
            sentences,
        }
    }
}

/// One training example.
#[derive(Debug, Clone)]
pub struct Example {
    pub source: Source,
    pub prompt: [u32; OBS_LEN],
    pub answer: [u32; ACTION_LEN],
}

/// A random source according to [`MIX`].
pub fn source(rng: &mut Rng) -> Source {
    let u = rng.uniform();
    let mut acc = 0.0;
    for (s, p) in MIX {
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

fn answer_tokens(text: &str) -> Option<[u32; ACTION_LEN]> {
    Action::Answer { text: text.to_string() }.encode(text::ru())
}

/// A text-continuation example: `<text>` + the first tokens of a sentence (left-padded), and
/// the next tokens up to [`ACTION_LEN`] − 1, then `<end>`.
pub fn continuation(rng: &mut Rng, sentence: &str) -> Option<([u32; OBS_LEN], [u32; ACTION_LEN])> {
    let s = fragment(text::ru(), sentence);
    if s.len() < 2 {
        return None;
    }
    let cut = 1 + rng.below(s.len() - 1);
    let ctx = &s[cut.saturating_sub(OBS_LEN - 1)..cut];
    let mut prompt = [PAD; OBS_LEN];
    prompt[OBS_LEN - 1 - ctx.len()] = TEXT;
    prompt[OBS_LEN - ctx.len()..].copy_from_slice(ctx);
    let mut answer = [PAD; ACTION_LEN];
    let rest = &s[cut..];
    let n = rest.len().min(ACTION_LEN - 1);
    answer[..n].copy_from_slice(&rest[..n]);
    if n == rest.len() {
        answer[n] = END;
    }
    Some((prompt, answer))
}

/// One example of `source`; `split` picks training or held-out data (held-out browser tasks use
/// the held-out wordings).
pub fn example(rng: &mut Rng, data: &LanguageData, source: Source, split: Split) -> Example {
    let held = split == Split::HeldOut;
    let dialogs = if held { &data.dialogs_valid } else { &data.dialogs };
    loop {
        let made = match source {
            Source::Browser => {
                let l = bdata::labelled(rng, split);
                let prompt = if rng.uniform() < 0.25 {
                    let history = stray_history(rng, dialogs);
                    obs::encode_dialog(&l.snapshot, &history, &l.goal.text, l.note.as_ref())
                } else {
                    l.observation
                };
                Some((prompt, l.action))
            }
            Source::Dialog => {
                let d = &dialogs[rng.below(dialogs.len())];
                let (history, line) = dialog_turns(&d.turns);
                let line = vary_case(rng, line);
                answer_tokens(&d.reply).map(|a| (obs::encode_dialog(&background_page(rng), &history, &line, None), a))
            }
            Source::Grammar => {
                let qs = if held { &data.questions_heldout } else { &data.questions };
                let q = &qs[rng.below(qs.len())];
                let history = if rng.uniform() < 0.3 { stray_history(rng, dialogs) } else { Vec::new() };
                // the word asked about is sometimes capitalised («Книга»); its forms stay lower-case
                let question = if rng.uniform() < 0.2 { capitalize_quoted(&q.question) } else { q.question.clone() };
                let question = vary_case(rng, question);
                answer_tokens(&q.answer)
                    .map(|a| (obs::encode_dialog(&background_page(rng), &history, &question, None), a))
            }
            Source::Text => {
                let ss = if held { &data.sentences_valid } else { &data.sentences };
                let i = rng.below(ss.len());
                continuation(rng, &ss[i])
            }
        };
        if let Some((prompt, answer)) = made {
            return Example { source, prompt, answer };
        }
    }
}

/// `n` examples mixed according to [`MIX`].
pub fn mixed(rng: &mut Rng, data: &LanguageData, n: usize, split: Split) -> Vec<Example> {
    (0..n)
        .map(|_| {
            let s = source(rng);
            example(rng, data, s, split)
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
        for source in Source::ALL {
            for _ in 0..40 {
                let e = example(&mut rng, &data, source, Split::Train);
                match source {
                    Source::Text => {
                        assert!(e.prompt.contains(&TEXT));
                        assert!(!e.prompt.contains(&GOAL));
                    }
                    Source::Dialog | Source::Grammar => {
                        assert_eq!(e.answer[0], ANSWER);
                        assert!(e.prompt.contains(&GOAL));
                        assert!(Action::decode(&e.answer, text::ru()).is_some());
                    }
                    Source::Browser => assert!(Action::decode(&e.answer, text::ru()).is_some()),
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
    fn long_history_keeps_the_newest_turns() {
        let bpe = text::ru();
        let turns: Vec<Turn> = (0..30).map(|i| Turn::user(format!("реплика номер {i} о погоде и планах"))).collect();
        let t = obs::history_tokens(&turns, bpe);
        assert!(t.len() <= obs::HISTORY_MAX && t.len() > obs::HISTORY_MAX / 2);
        assert!(bpe.decode(&t).contains("номер 29"), "{}", bpe.describe(&t));
        assert!(!bpe.decode(&t).contains("номер 1 "), "{}", bpe.describe(&t));
    }
}
