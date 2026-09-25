//! Observation encoding: page snapshot + instruction (+ the last tool result) → a fixed-length
//! prompt of BPE tokens.
//!
//! ```text
//!  [head] ␣Результаты [input] ␣лампа [button] ␣Найти [link] ␣стул [value] ␣1·2·␣₽ … <pad>… <goal> ␣Сколько·␣будет·␣1·2·+·3·0·? CALC ␣1·2·+·3·0·␣=·␣4·2
//!  └──────────────── page: role token + text, document order ─────────────────┘         └──────────── instruction ───────────┘ └──── tool result ────┘
//! ```
//!
//! An empty text field is `<empty>`. The instruction (at most [`GOAL_MAX`] tokens) and the
//! result of the last calculator call (at most [`NOTE_MAX`]) always end the observation, the
//! newest information last, next to the readout of the TTT encoder; the page is truncated to
//! the remaining space. Every text is a [`fragment`] (leading space), so a number reads the
//! same on the page, in the question, in the tool result and in the answer.

use super::{PageSnapshot, Role};
use crate::text::{self, fragment, Bpe, CALC, EMPTY, GOAL, PAD};

/// Prompt length `N` of the browsing task.
pub const OBS_LEN: usize = 128;
/// Longest instruction kept, in tokens (including `<goal>`).
pub const GOAL_MAX: usize = 32;
/// Longest tool result kept, in tokens (including `CALC`).
pub const NOTE_MAX: usize = 24;

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

/// Encodes what the agent sees.
pub fn encode(snapshot: &PageSnapshot, instruction: &str, note: Option<&Note>) -> [u32; OBS_LEN] {
    let bpe = text::ru();
    let page = page_tokens(snapshot, bpe);
    let mut tail = goal_tokens(instruction, bpe);
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
