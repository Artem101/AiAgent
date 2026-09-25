//! Web-browsing agent: the engine reads a web page and a Russian instruction, reasons in
//! latent space, decides on an action, and a real browser (headless Chromium over the DevTools
//! protocol) executes it — until the model answers.
//!
//! ```text
//!   ┌──────────── Browser (Chromium via CDP, or the DOM-identical simulator) ─────────────────┐
//!   │  page ──snapshot──► [role] BPE-text … <goal> «Что дешевле: лампа или стул?»             │
//!   │   ▲                       (observation, 96 tokens)                                      │
//!   │   │                                  ▼                                                  │
//!   │   │        CognitiveEngine: TTT → latent reasoning (tree + MPPI + latent GD, H = 8) → CFM│
//!   │   └──── click / type / back ◄── Action [verb, role, text…, <end>] (8 tokens) ◄──┘       │
//!   └──────────────────────────────── … until ANSWER ─────────────────────────────────────────┘
//! ```
//!
//! * [`world`] — the sandbox web: a Russian shop with search and a paged catalogue whose facts
//!   are drawn from a per-world seed, so answers cannot be memorised and must be looked up;
//! * [`goal`] — task specs (lookup, compare, filter, calc, total, chat) and their Russian wordings;
//! * [`server`] — serves the sandbox web over HTTP for the real browser;
//! * [`chrome`] / [`cdp`] — Chromium launcher and a dependency-free DevTools client;
//! * [`sim`] — an in-process browser over the same pages (fast training data, tests);
//! * [`obs`] / [`action`] — observations and actions as BPE token sequences ([`crate::text::ru`]);
//!   `CALC` actions go to the calculator tool ([`crate::tools`]), its result shows up in the next
//!   observation;
//! * [`expert`] — scripted teacher used for behaviour cloning;
//! * [`agent`] — the observe → act loop, episodes and success-rate evaluation;
//! * [`data`] — random browser states labelled by the teacher (`--task browser`).
//!
//! The same [`Browser`] trait is implemented by [`chrome::Chrome`] and [`sim::SimBrowser`];
//! `tests/browser.rs` checks that both produce identical snapshots along whole episodes.

pub mod action;
pub mod agent;
pub mod cdp;
pub mod chrome;
pub mod data;
pub mod expert;
pub mod goal;
pub mod obs;
pub mod server;
pub mod sim;
pub mod world;

use candle_core::Result;

use crate::config::{ActionDecoder, EngineConfig, PlannerKind};

pub use action::{Action, ACTION_LEN};
pub use agent::{run_episode, EnginePolicy, Episode, ExpertPolicy, Policy};
pub use chrome::Chrome;
pub use goal::{Family, Goal, Spec, Split};
pub use obs::OBS_LEN;
pub use server::SiteServer;
pub use sim::SimBrowser;
pub use world::World;

/// TTT causal window used for browsing: at a value cell the window covers the value, its role
/// token and the label before it, so one fast-weight update binds the label to the value.
pub const CONV_WIDTH: usize = 4;
/// Final TTT outputs in the readout: the last instruction tokens act as queries into the page
/// stored in `W_fast`.
pub const READOUT_LAST: usize = 4;
/// Gated readout pools: position-independent selection of the informative words of the
/// instruction (and the page).
pub const READOUT_POOLS: usize = 4;
/// Weight of the answer probe on `s_0` (see `JepaConfig::probe_weight`).
pub const PROBE_WEIGHT: f64 = 1.0;
/// Key width of the probe's copy mechanism: thoughts point at the words and digits of the page,
/// the question and the tool result instead of spelling them out (see [`crate::copy`]).
pub const COPY_DIM: usize = 32;
/// Latent planning horizon: one latent thought step per two action tokens.
pub const HORIZON: usize = 8;
/// Thoughts the probe is trained on per step: `s_0`, `s_H` and one random intermediate one
/// (see `TrainConfig::probe_states`).
pub const PROBE_STATES: usize = 3;
/// Latent tree search: hypotheses kept per depth (beam) and proposals per hypothesis.
pub const TREE_BEAM: usize = 4;
pub const TREE_BRANCH: usize = 4;

/// Vocabulary size of the browsing task (the Russian BPE tokenizer).
pub fn vocab_size() -> usize {
    crate::text::ru().vocab_size()
}

/// Engine configuration for the browsing task (`preset` = `tiny` | `base` | `small`).
pub fn engine_config(preset: &str) -> Result<EngineConfig> {
    let mut cfg = EngineConfig::preset(preset, vocab_size(), OBS_LEN, ACTION_LEN)?;
    cfg.ttt.conv_width = CONV_WIDTH;
    cfg.ttt.readout_last = READOUT_LAST;
    cfg.ttt.readout_pools = READOUT_POOLS;
    cfg.jepa.probe_weight = PROBE_WEIGHT;
    cfg.jepa.copy_dim = COPY_DIM;
    cfg.jepa.copy_min_token = crate::text::SPECIALS.len() as u32;
    cfg.jepa.horizon = HORIZON;
    // Mandatory latent reasoning before every action: tree of hypotheses → MPPI → latent GD.
    cfg.planner.tree_beam = TREE_BEAM;
    cfg.planner.tree_branch = TREE_BRANCH;
    cfg.planner.kind = PlannerKind::MppiThenGradient;
    // Actions are read from the thoughts: the first thought decides, and the rest of the plan and
    // the surviving tree hypotheses, ranked by how much all thoughts agree on them, are the
    // alternatives `EnginePolicy` acts on when that action is malformed or had no effect.
    cfg.decoder = ActionDecoder::ProbeFirst;
    Ok(cfg)
}

/// Kind of a page element the agent can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// `<h1>`
    Heading,
    /// `<a>`
    Link,
    /// `<input>`
    Input,
    /// `<button>`
    Button,
    /// `<th>`
    Label,
    /// `<td>`
    Value,
    /// `<p>`
    Text,
}

impl Role {
    pub const ALL: [Role; 7] =
        [Self::Heading, Self::Link, Self::Input, Self::Button, Self::Label, Self::Value, Self::Text];

    /// Observation token of the role (`[head]`, `[link]`, …).
    pub fn token(self) -> u32 {
        6 + Self::ALL.iter().position(|&r| r == self).expect("role in ALL") as u32
    }

    pub fn from_token(token: u32) -> Option<Self> {
        token.checked_sub(6).and_then(|i| Self::ALL.get(i as usize).copied())
    }

    /// Role of an HTML tag (as reported by the DOM snapshot script).
    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "h1" => Self::Heading,
            "a" => Self::Link,
            "input" => Self::Input,
            "button" => Self::Button,
            "th" => Self::Label,
            "td" => Self::Value,
            "p" => Self::Text,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Heading => "heading",
            Self::Link => "link",
            Self::Input => "input",
            Self::Button => "button",
            Self::Label => "label",
            Self::Value => "value",
            Self::Text => "text",
        }
    }
}

/// One visible element, in document order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Element {
    pub role: Role,
    /// Visible text (`innerText`), empty for inputs.
    pub text: String,
    /// Current value of an input, empty otherwise.
    pub value: String,
}

/// What the agent sees of a page. Element indices are the ids accepted by
/// [`Browser::click`] and [`Browser::type_text`] until the next snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSnapshot {
    pub url: String,
    pub title: String,
    pub elements: Vec<Element>,
}

impl PageSnapshot {
    /// Index of the first element with `role` whose text equals `text` (case-insensitive,
    /// surrounding whitespace ignored).
    pub fn find(&self, role: Role, text: &str) -> Option<usize> {
        let text = text.trim().to_lowercase();
        self.elements.iter().position(|e| e.role == role && e.text.trim().to_lowercase() == text)
    }

    /// Index of the first element with `role`.
    pub fn first(&self, role: Role) -> Option<usize> {
        self.elements.iter().position(|e| e.role == role)
    }
}

/// A web browser the agent can drive.
pub trait Browser {
    /// Loads `url` and waits until the page has loaded.
    fn goto(&mut self, url: &str) -> Result<()>;
    /// Reads the current page.
    fn snapshot(&mut self) -> Result<PageSnapshot>;
    /// Clicks element `id` of the last snapshot (follows links, submits forms).
    fn click(&mut self, id: usize) -> Result<()>;
    /// Replaces the content of text field `id` of the last snapshot with `text`.
    fn type_text(&mut self, id: usize, text: &str) -> Result<()>;
    /// Goes one step back in history.
    fn back(&mut self) -> Result<()>;
}
