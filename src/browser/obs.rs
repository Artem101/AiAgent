//! Observation encoding: page snapshot + goal → a fixed-length token prompt, and back.
//!
//! ```text
//!  [head] results [input] lamp [button] search [link] sofa [link] lamp … <pad>… <goal> lamp price
//!  └────────────────── page: role token + words, document order ─────┘        └── last 3 slots ──┘
//! ```
//!
//! Words outside the vocabulary become `<unk>`, an empty text field is `<empty>`. The page part
//! is truncated to `OBS_LEN − 3` tokens; the goal always occupies the last three positions,
//! so it sits at the same positional embeddings on every page.

use super::vocab::{self, Goal, EMPTY, GOAL, PAD, UNK};
use super::{PageSnapshot, Role};

/// Prompt length `N` of the browsing task.
pub const OBS_LEN: usize = 24;
/// Tokens available for the page.
pub const PAGE_LEN: usize = OBS_LEN - 3;

fn words(text: &str) -> impl Iterator<Item = u32> + '_ {
    text.split_whitespace().map(|w| vocab::word_id(w).unwrap_or(UNK))
}

/// Encodes what the agent sees.
pub fn encode(snapshot: &PageSnapshot, goal: &Goal) -> [u32; OBS_LEN] {
    let mut page = Vec::with_capacity(2 * snapshot.elements.len());
    for e in &snapshot.elements {
        page.push(e.role.token());
        if e.role == Role::Input {
            let before = page.len();
            page.extend(words(&e.value));
            if page.len() == before {
                page.push(EMPTY);
            }
        } else {
            page.extend(words(&e.text));
        }
    }
    let mut out = [PAD; OBS_LEN];
    let n = page.len().min(PAGE_LEN);
    out[..n].copy_from_slice(&page[..n]);
    out[PAGE_LEN..].copy_from_slice(&goal.tokens());
    out
}

/// A decoded observation: elements as `(role, words)` plus the goal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObsView {
    pub elements: Vec<(Role, Vec<u32>)>,
    pub goal: Option<Goal>,
}

impl ObsView {
    /// Words of the first element with `role`.
    pub fn first(&self, role: Role) -> Option<&[u32]> {
        self.elements.iter().find(|(r, _)| *r == role).map(|(_, w)| w.as_slice())
    }

    /// Whether some element with `role` consists of exactly `word`.
    pub fn has(&self, role: Role, word: u32) -> bool {
        self.elements.iter().any(|(r, w)| *r == role && w.as_slice() == [word])
    }
}

/// Inverse of [`encode`] (up to truncation and `<unk>`).
pub fn parse(tokens: &[u32]) -> ObsView {
    let (page, goal) = match tokens.iter().rposition(|&t| t == GOAL) {
        Some(i) => (&tokens[..i], &tokens[i + 1..]),
        None => (tokens, &[][..]),
    };
    let goal = match goal {
        &[item, attr, ..] if vocab::is_item(item) && vocab::is_attr(attr) => Some(Goal { item, attr }),
        _ => None,
    };
    let mut elements: Vec<(Role, Vec<u32>)> = Vec::new();
    for &t in page {
        if let Some(role) = Role::from_token(t) {
            elements.push((role, Vec::new()));
        } else if let Some((_, w)) = elements.last_mut() {
            if t != PAD && t != EMPTY {
                w.push(t);
            }
        }
    }
    ObsView { elements, goal }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::world::World;

    #[test]
    fn encode_parse_round_trip() {
        let goal = Goal::parse("price of lamp").unwrap();
        for path in ["/w/1/", "/w/1/search?q=lamp", "/w/1/item/lamp", "/w/1/item/sofa", "/nope"] {
            let page = World::page_at(path);
            let snap = PageSnapshot {
                url: path.into(),
                title: page.title(),
                elements: page.nodes().into_iter().map(|n| n.element).collect(),
            };
            let obs = encode(&snap, &goal);
            assert_eq!(&obs[PAGE_LEN..], &goal.tokens());
            let view = parse(&obs);
            assert_eq!(view.goal, Some(goal));
            assert_eq!(view.elements.len(), snap.elements.len(), "{path}: {}", vocab::describe(&obs));
            for ((role, w), e) in view.elements.iter().zip(&snap.elements) {
                assert_eq!(*role, e.role);
                let text = if e.role == Role::Input { &e.value } else { &e.text };
                let expect: Vec<u32> = words(text).collect();
                assert_eq!(w, &expect);
            }
        }
        // every sandbox page fits into the observation
        let longest = World::page_at("/w/1/item/lamp").nodes().len() * 2;
        assert!(longest <= PAGE_LEN);
    }
}
