//! Web-browsing agent: the engine reads a web page, decides on an action, and a real browser
//! (headless Chromium over the DevTools protocol) executes it — until the model answers.
//!
//! ```text
//!   ┌──────────── Browser (Chromium via CDP, or the DOM-identical simulator) ───────────┐
//!   │  page ──snapshot──► [role, words]… + <goal> item attr ──► CognitiveEngine          │
//!   │   ▲                        (observation, 24 tokens)          (TTT → JEPA → CFM)    │
//!   │   └──── click / type / back ◄── Action  [verb, role, word, <end>] ◄─────┘          │
//!   └─────────────────────────────────── … until ANSWER ─────────────────────────────────┘
//! ```
//!
//! * [`world`] — the sandbox web: a search engine and product pages whose facts are drawn
//!   from a per-world seed, so answers cannot be memorised and must be looked up;
//! * [`server`] — serves the sandbox web over HTTP for the real browser;
//! * [`chrome`] / [`cdp`] — Chromium launcher and a dependency-free DevTools client
//!   (WebSocket + JSON-RPC);
//! * [`sim`] — an in-process browser over the same pages (fast training data, tests);
//! * [`obs`] — page snapshot + goal → observation tokens (and back);
//! * [`expert`] — scripted teacher used for behaviour cloning;
//! * [`agent`] — the observe → act loop, episodes and success-rate evaluation;
//! * [`data`] — random browser states labelled by the expert (`--task browser`).
//!
//! The same [`Browser`] trait is implemented by [`chrome::Chrome`] and [`sim::SimBrowser`];
//! `tests/browser.rs` checks that both produce identical snapshots along whole episodes.

pub mod agent;
pub mod cdp;
pub mod chrome;
pub mod data;
pub mod expert;
pub mod obs;
pub mod server;
pub mod sim;
pub mod vocab;
pub mod world;

use candle_core::Result;

use crate::config::EngineConfig;

pub use agent::{run_episode, EnginePolicy, Episode, ExpertPolicy, Policy};
pub use chrome::Chrome;
pub use obs::OBS_LEN;
pub use server::SiteServer;
pub use sim::SimBrowser;
pub use vocab::{Action, Goal, ACTION_LEN, VOCAB_SIZE};
pub use world::World;

/// TTT causal window used for browsing: at a value cell the window `[value, [value], label,
/// [label]]` lets one fast-weight update bind the attribute name to its value.
pub const CONV_WIDTH: usize = 4;
/// Final TTT outputs in the readout: the three goal tokens at the end of the observation act as
/// queries into the page stored in `W_fast`.
pub const READOUT_LAST: usize = 3;

/// Weight of the answer probe on `s_0` (see `JepaConfig::probe_weight`).
pub const PROBE_WEIGHT: f64 = 1.0;

/// Engine configuration for the browsing task (`preset` = `tiny` | `small`).
pub fn engine_config(preset: &str) -> Result<EngineConfig> {
    let mut cfg = EngineConfig::preset(preset, VOCAB_SIZE, OBS_LEN, ACTION_LEN)?;
    cfg.ttt.conv_width = CONV_WIDTH;
    cfg.ttt.readout_last = READOUT_LAST;
    cfg.jepa.probe_weight = PROBE_WEIGHT;
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
    /// Index of the first element with `role` whose text equals `text` (case-insensitive).
    pub fn find(&self, role: Role, text: &str) -> Option<usize> {
        self.elements.iter().position(|e| e.role == role && e.text.eq_ignore_ascii_case(text))
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
