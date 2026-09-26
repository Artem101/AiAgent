//! Observation encoding: page snapshot + conversation + instruction (+ the last tool result) →
//! a fixed-length prompt of BPE tokens.
//!
//! ```text
//!  [head] ␣Результаты [input] ␣лампа [button] ␣Найти … <pad>… <user> ␣Привет! <bot> ␣Привет!·␣Чем·␣помочь? <goal> ␣Сколько·␣будет·␣1·2·+·3·0·? CALC ␣1·2·+·3·0·␣=·␣4·2
//!  └──────────── page: role token + text, document order ────────────┘       └──────── earlier turns ────────┘ └──────── instruction ───────┘ └── tool result ──┘
//! ```
//!
//! An empty text field is `<empty>`. The earlier turns of the conversation (at most
//! [`HISTORY_MAX`] tokens, the newest kept), the instruction — the user's current message, at
//! most [`GOAL_MAX`] tokens — and the result of the last calculator call (at most [`NOTE_MAX`])
//! end the observation, the newest information last, next to the readout of the TTT encoder;
//! the page is truncated to the remaining space. Every text is a [`fragment`] (leading space),
//! so a number reads the same on the page, in the question, in the tool result and in the
//! answer.

use super::{PageSnapshot, Role};
use crate::text::{self, fragment, Bpe, BOT, CALC, EMPTY, GOAL, PAD, USER};

/// Prompt length `N` of the browsing task.
pub const OBS_LEN: usize = 160;
/// Longest instruction kept, in tokens (including `<goal>`).
pub const GOAL_MAX: usize = 40;
/// Longest tool result kept, in tokens (including `CALC`).
pub const NOTE_MAX: usize = 24;
/// Longest conversation history kept, in tokens (including the `<user>` / `<bot>` markers).
pub const HISTORY_MAX: usize = 56;

/// Who said a turn of the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    User,
    Bot,
}

/// An earlier turn of the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    pub speaker: Speaker,
    pub text: String,
}

impl Turn {
    pub fn user(text: impl Into<String>) -> Self {
        Self { speaker: Speaker::User, text: text.into() }
    }
    pub fn bot(text: impl Into<String>) -> Self {
        Self { speaker: Speaker::Bot, text: text.into() }
    }
}

/// What the last calculator call returned: the expression the agent sent and the printed
/// result (a number, or «ошибка»). It stays in the observation for the rest of the episode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub expr: String,
    pub result: String,
}

impl Note {
    /// The note of a calculator reply.
    pub fn of(expr: &str, reply: &Result<String, crate::tools::CalcError>) -> Self {
        let result = match reply {
            Ok(v) => v.clone(),
            Err(_) => "ошибка".to_string(),
        };
        Self { expr: expr.trim().to_string(), result }
    }

    /// Whether the note is the result of `expr` (spaces ignored).
    pub fn is_for(&self, expr: &str) -> bool {
        let squash = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        squash(&self.expr) == squash(expr)
    }
}

/// Page part of an observation (no truncation).
pub fn page_tokens(snapshot: &PageSnapshot, bpe: &Bpe) -> Vec<u32> {
    let mut page = Vec::with_capacity(8 * snapshot.elements.len());
    for e in &snapshot.elements {
        page.push(e.role.token());
        if e.role == Role::Input {
            let v = fragment(bpe, &e.value);
            page.extend(if v.is_empty() { vec![EMPTY] } else { v });
        } else {
            page.extend(fragment(bpe, &e.text));
        }
    }
    page
}

/// `<goal>` + instruction tokens (at most [`GOAL_MAX`]).
pub fn goal_tokens(instruction: &str, bpe: &Bpe) -> Vec<u32> {
    let mut g = vec![GOAL];
    g.extend(fragment(bpe, instruction));
    g.truncate(GOAL_MAX);
    g
}

/// `CALC` + `expr = result` (at most [`NOTE_MAX`]; empty without a note).
pub fn note_tokens(note: Option<&Note>, bpe: &Bpe) -> Vec<u32> {
    let Some(n) = note else { return Vec::new() };
    let mut t = vec![CALC];
    t.extend(fragment(bpe, &format!("{} = {}", n.expr, n.result)));
    t.truncate(NOTE_MAX);
    t
}

/// `<user> … <bot> …` for the earlier turns, oldest first, at most [`HISTORY_MAX`] tokens: the
/// newest turns are kept, and an older turn that does not fit whole is dropped (the newest one
/// is cut instead).
pub fn history_tokens(history: &[Turn], bpe: &Bpe) -> Vec<u32> {
    let mut parts: Vec<Vec<u32>> = Vec::new();
    let mut used = 0;
    for turn in history.iter().rev() {
        let mut t = vec![if turn.speaker == Speaker::User { USER } else { BOT }];
        t.extend(fragment(bpe, &turn.text));
        if used + t.len() > HISTORY_MAX {
            if parts.is_empty() {
                t.truncate(HISTORY_MAX);
                parts.push(t);
            }
            break;
        }
        used += t.len();
        parts.push(t);
    }
    parts.into_iter().rev().flatten().collect()
}

/// Encodes what the agent sees (no conversation history).
pub fn encode(snapshot: &PageSnapshot, instruction: &str, note: Option<&Note>) -> [u32; OBS_LEN] {
    encode_dialog(snapshot, &[], instruction, note)
}

/// Encodes what the agent sees, with the earlier turns of the conversation.
pub fn encode_dialog(
    snapshot: &PageSnapshot,
    history: &[Turn],
    instruction: &str,
    note: Option<&Note>,
) -> [u32; OBS_LEN] {
    let bpe = text::ru();
    let page = page_tokens(snapshot, bpe);
    let mut tail = history_tokens(history, bpe);
    tail.extend(goal_tokens(instruction, bpe));
    tail.extend(note_tokens(note, bpe));
    let mut out = [PAD; OBS_LEN];
    let n = page.len().min(OBS_LEN - tail.len());
    out[..n].copy_from_slice(&page[..n]);
    out[OBS_LEN - tail.len()..].copy_from_slice(&tail);
    out
}

/// Readable form of an observation or action (BPE pieces separated by `·`).
pub fn describe(tokens: &[u32]) -> String {
    text::ru().describe(tokens)
}
