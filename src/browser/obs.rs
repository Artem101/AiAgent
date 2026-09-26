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

use super::action::ACTION_LEN;
use super::{PageSnapshot, Role};
use crate::text::{self, fragment, Bpe, BOT, CALC, EMPTY, GOAL, LOOKUP, PAD, THINK, USER};

/// Prompt length `N` of the browsing task.
pub const OBS_LEN: usize = 160;
/// Longest instruction kept, in tokens (including `<goal>`).
pub const GOAL_MAX: usize = 40;
/// Longest tool result kept, in tokens (including `CALC`).
pub const NOTE_MAX: usize = 24;
/// Longest conversation history kept, in tokens (including the `<user>` / `<bot>` markers).
pub const HISTORY_MAX: usize = 56;

/// Sizes of an observation format. [`Layout::V1`] is the format of the shipped model (the
/// constants above: the last calculator result is the only tool output kept); [`Layout::V2`]
/// is the format of the scaled-up models of docs/scaling.md: longer observation and action,
/// and a scratchpad with the agent's recent steps (`THINK`, `CALC`, `LOOKUP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Observation length `N`.
    pub len: usize,
    pub goal_max: usize,
    pub history_max: usize,
    /// Tokens of tool results (V1) or of the scratchpad (V2).
    pub scratch_max: usize,
    /// Action length `L`.
    pub action_len: usize,
    /// Keep every recent step (V2), or the last calculator result only (V1).
    pub scratchpad: bool,
}

impl Layout {
    pub const V1: Self = Self {
        len: OBS_LEN,
        goal_max: GOAL_MAX,
        history_max: HISTORY_MAX,
        scratch_max: NOTE_MAX,
        action_len: ACTION_LEN,
        scratchpad: false,
    };
    pub const V2: Self =
        Self { len: 320, goal_max: 64, history_max: 56, scratch_max: 128, action_len: 64, scratchpad: true };

    pub fn name(&self) -> &'static str {
        if self.scratchpad {
            "v2"
        } else {
            "v1"
        }
    }

    pub fn parse(s: &str) -> candle_core::Result<Self> {
        match s {
            "v1" => Ok(Self::V1),
            "v2" => Ok(Self::V2),
            other => candle_core::bail!("unknown observation layout '{other}' (v1 | v2)"),
        }
    }
}

/// A step the agent took in this episode, kept in its scratchpad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// A step of reasoning (`THINK`).
    Think(String),
    /// A calculator call and its result (`CALC`).
    Calc(Note),
    /// A dictionary look-up and the entry found (`LOOKUP`).
    Lookup { word: String, entry: String },
    /// A browser action taken (`CLICK [button] Найти`, `TYPE [input] лампа`, `BACK`): the agent
    /// remembers what it already did.
    Act(super::Action),
}

impl Entry {
    /// `THINK …`, `CALC expr = result`, `LOOKUP word → entry`, `CLICK [role] text`.
    pub fn tokens(&self, bpe: &Bpe) -> Vec<u32> {
        let (marker, text) = match self {
            Self::Think(t) => (vec![THINK], t.clone()),
            Self::Calc(n) => (vec![CALC], format!("{} = {}", n.expr, n.result)),
            Self::Lookup { word, entry } => (vec![LOOKUP], format!("{word} → {entry}")),
            Self::Act(a) => {
                let t = a.encode_len(bpe, 3 + 64).unwrap_or_default();
                let end = t.iter().position(|&x| x == crate::text::END).unwrap_or(t.len());
                return t[..end].to_vec();
            }
        };
        let mut t = marker;
        t.extend(fragment(bpe, &text));
        t
    }
}

/// The result of the last calculator call among `entries`.
pub fn last_calc(entries: &[Entry]) -> Option<&Note> {
    entries.iter().rev().find_map(|e| if let Entry::Calc(n) = e { Some(n) } else { None })
}

/// Tool results in the observation: in [`Layout::V1`] the last calculator result (at most
/// `scratch_max` tokens); in [`Layout::V2`] the newest steps that fit into `scratch_max`,
/// oldest first (the newest one is cut when it alone is too long).
pub fn scratch_tokens(entries: &[Entry], layout: &Layout, bpe: &Bpe) -> Vec<u32> {
    if !layout.scratchpad {
        let mut t = last_calc(entries).map(|n| Entry::Calc(n.clone()).tokens(bpe)).unwrap_or_default();
        t.truncate(layout.scratch_max);
        return t;
    }
    let mut parts: Vec<Vec<u32>> = Vec::new();
    let mut used = 0;
    for e in entries.iter().rev() {
        let mut t = e.tokens(bpe);
        if used + t.len() > layout.scratch_max {
            if parts.is_empty() {
                t.truncate(layout.scratch_max);
                parts.push(t);
            }
            break;
        }
        used += t.len();
        parts.push(t);
    }
    parts.into_iter().rev().flatten().collect()
}

/// Encodes what the agent sees in `layout`: the page, the earlier turns, the instruction and
/// the scratchpad (see the module docs).
pub fn encode_with(
    layout: &Layout,
    snapshot: &PageSnapshot,
    history: &[Turn],
    instruction: &str,
    entries: &[Entry],
) -> Vec<u32> {
    let bpe = text::ru();
    let page = page_tokens(snapshot, bpe);
    let mut tail = history_tokens_max(history, bpe, layout.history_max);
    tail.extend(goal_tokens_max(instruction, bpe, layout.goal_max));
    tail.extend(scratch_tokens(entries, layout, bpe));
    tail.truncate(layout.len);
    let mut out = vec![PAD; layout.len];
    let n = page.len().min(layout.len - tail.len());
    out[..n].copy_from_slice(&page[..n]);
    out[layout.len - tail.len()..].copy_from_slice(&tail);
    out
}

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
    goal_tokens_max(instruction, bpe, GOAL_MAX)
}

/// `<goal>` + instruction tokens (at most `max`).
pub fn goal_tokens_max(instruction: &str, bpe: &Bpe, max: usize) -> Vec<u32> {
    let mut g = vec![GOAL];
    g.extend(fragment(bpe, instruction));
    g.truncate(max);
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
    history_tokens_max(history, bpe, HISTORY_MAX)
}

/// [`history_tokens`] with at most `max` tokens.
pub fn history_tokens_max(history: &[Turn], bpe: &Bpe, max: usize) -> Vec<u32> {
    let mut parts: Vec<Vec<u32>> = Vec::new();
    let mut used = 0;
    for turn in history.iter().rev() {
        let mut t = vec![if turn.speaker == Speaker::User { USER } else { BOT }];
        t.extend(fragment(bpe, &turn.text));
        if used + t.len() > max {
            if parts.is_empty() {
                t.truncate(max);
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
    let entries: Vec<Entry> = note.map(|n| Entry::Calc(n.clone())).into_iter().collect();
    encode_with(&Layout::V1, snapshot, history, instruction, &entries).try_into().expect("OBS_LEN tokens")
}

/// Readable form of an observation or action (BPE pieces separated by `·`).
pub fn describe(tokens: &[u32]) -> String {
    text::ru().describe(tokens)
}
