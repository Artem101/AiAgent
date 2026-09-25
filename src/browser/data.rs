//! Training data for `--task browser`: random browser states labelled by the teacher.
//!
//! Each example draws a world, a goal of one of the six families (worded with the training
//! templates), a page state and the last calculator result, if any. States are not only the
//! teacher's own trajectory: they also cover typed-but-not-submitted queries, wrong queries,
//! wrong product pages, stray pages of another task, error pages and calculator results for a
//! mistyped expression, so the policy learns to recover from its own mistakes (behaviour
//! cloning with DAgger-style state coverage).

use super::action::{Action, ACTION_LEN};
use super::goal::{self, Family, Goal, Spec, Split};
use super::obs::{self, Note, OBS_LEN};
use super::world::{url_encode, World, CATALOG_PAGES, ITEMS};
use super::{expert, PageSnapshot, Role};
use crate::kernels::rng::Rng;
use crate::text;
use crate::tools::calc;

/// Share of each task family in the training data.
pub const FAMILY_MIX: [(Family, f64); 6] = [
    (Family::Lookup, 0.25),
    (Family::Compare, 0.17),
    (Family::Filter, 0.17),
    (Family::Calc, 0.2),
    (Family::Total, 0.15),
    (Family::Chat, 0.06),
];

/// Out-of-vocabulary text a user could leave in the search box.
const JUNK: [&str; 4] = ["лампочка", "qwerty", "что-то", "товар"];

fn other_item(rng: &mut Rng, not: &[usize]) -> usize {
    loop {
        let i = rng.below(ITEMS.len());
        if !not.contains(&i) {
            return i;
        }
    }
}

fn wrong_text(rng: &mut Rng, not: &[usize]) -> String {
    if rng.uniform() < 0.2 {
        JUNK[rng.below(JUNK.len())].to_string()
    } else {
        ITEMS[other_item(rng, not)].nom.to_string()
    }
}

pub fn family(rng: &mut Rng) -> Family {
    let u = rng.uniform();
    let mut acc = 0.0;
    for (f, p) in FAMILY_MIX {
        acc += p;
        if u < acc {
            return f;
        }
    }
    Family::Chat
}

/// A calculator result for an expression close to `expr` but not it (a mistyped digit, or
/// something unrelated) — what the model sees after a wrong `CALC`.
fn wrong_note(rng: &mut Rng, expr: &str) -> Note {
    let mut e: Vec<char> = expr.chars().collect();
    let digits: Vec<usize> = (0..e.len()).filter(|&i| e[i].is_ascii_digit()).collect();
    if rng.uniform() < 0.7 && !digits.is_empty() {
        let i = digits[rng.below(digits.len())];
        let d = e[i].to_digit(10).unwrap_or(0);
        e[i] = char::from_digit((d + 1 + rng.below(9) as u32) % 10, 10).unwrap_or('0');
    } else {
        e = goal::sample_expr(rng).text(rng.uniform() < 0.5).chars().collect();
    }
    let e: String = e.into_iter().collect();
    if Note::of(&e, &Ok(String::new())).is_for(expr) {
        return Note::of("1 + 1", &calc::run("1 + 1"));
    }
    Note::of(&e, &calc::run(&e))
}

/// A page state for `spec`: `(path, text typed into the search box)`.
fn state(rng: &mut Rng, world: &World, spec: &Spec) -> (String, Option<String>) {
    let seed = world.seed;
    let home = format!("/w/{seed}/");
    let results = |q: &str| format!("/w/{seed}/search?q={}", url_encode(q));
    let item_page = |i: usize| format!("/w/{seed}/item/{}", url_encode(ITEMS[i].nom));
    let catalog = |p: usize| format!("/w/{seed}/catalog?page={p}");
    let u = rng.uniform();
    match *spec {
        Spec::Lookup { item, .. } => {
            let name = ITEMS[item].nom.to_string();
            let typed = |rng: &mut Rng| match rng.uniform() {
                x if x < 0.5 => None,
                x if x < 0.8 => Some(name.clone()),
                _ => Some(wrong_text(rng, &[item])),
            };
            if u < 0.2 {
                let t = typed(rng);
                (home, t)
            } else if u < 0.5 {
                let q = match rng.uniform() {
                    x if x < 0.6 => name.clone(),
                    x if x < 0.7 => ITEMS[item].gen.to_string(), // stem search still finds it
                    _ => wrong_text(rng, &[item]),
                };
                let t = if rng.uniform() < 0.2 { typed(rng) } else { None };
                (results(&q), t)
            } else if u < 0.85 {
                let i = if rng.uniform() < 0.75 { item } else { other_item(rng, &[item]) };
                (item_page(i), None)
            } else if u < 0.95 {
                (catalog(1 + rng.below(CATALOG_PAGES)), None)
            } else {
                (format!("/w/{seed}/item/404"), None)
            }
        }
        Spec::Compare { a, b, .. } => {
            let query = format!("{} {}", ITEMS[a].nom, ITEMS[b].nom);
            let partial = |rng: &mut Rng| ITEMS[if rng.uniform() < 0.5 { a } else { b }].nom.to_string();
            if u < 0.25 {
                let t = match rng.uniform() {
                    x if x < 0.4 => None,
                    x if x < 0.7 => Some(query.clone()),
                    x if x < 0.85 => Some(partial(rng)),
                    _ => Some(wrong_text(rng, &[a, b])),
                };
                (home, t)
            } else if u < 0.8 {
                let q = match rng.uniform() {
                    x if x < 0.65 => query.clone(),
                    x if x < 0.7 => format!("{} {}", ITEMS[b].nom, ITEMS[a].nom),
                    x if x < 0.85 => partial(rng),
                    _ => wrong_text(rng, &[a, b]),
                };
                let t = match rng.uniform() {
                    x if x < 0.85 => None,
                    x if x < 0.93 => Some(query.clone()),
                    _ => Some(partial(rng)),
                };
                (results(&q), t)
            } else if u < 0.9 {
                let i = [a, b, other_item(rng, &[a, b])][rng.below(3)];
                (item_page(i), None)
            } else {
                (catalog(1 + rng.below(CATALOG_PAGES)), None)
            }
        }
        Spec::Calc { .. } | Spec::Chat(_) => {
            // asked anywhere: mostly on the start page
            if u < 0.6 {
                (home, None)
            } else if u < 0.75 {
                (results(&wrong_text(rng, &[])), None)
            } else if u < 0.85 {
                (item_page(rng.below(ITEMS.len())), None)
            } else if u < 0.95 {
                (catalog(1 + rng.below(CATALOG_PAGES)), None)
            } else {
                (format!("/w/{seed}/item/404"), None)
            }
        }
        Spec::Total { a, b, .. } => {
            let query = format!("{} {}", ITEMS[a].nom, ITEMS[b].nom);
            if u < 0.2 {
                let t = match rng.uniform() {
                    x if x < 0.5 => None,
                    x if x < 0.8 => Some(query.clone()),
                    _ => Some(wrong_text(rng, &[a, b])),
                };
                (home, t)
            } else if u < 0.85 {
                let q = match rng.uniform() {
                    x if x < 0.75 => query.clone(),
                    x if x < 0.8 => format!("{} {}", ITEMS[b].nom, ITEMS[a].nom),
                    x if x < 0.9 => ITEMS[if rng.uniform() < 0.5 { a } else { b }].nom.to_string(),
                    _ => wrong_text(rng, &[a, b]),
                };
                (results(&q), None)
            } else if u < 0.95 {
                (item_page([a, b, other_item(rng, &[a, b])][rng.below(3)]), None)
            } else {
                (catalog(1 + rng.below(CATALOG_PAGES)), None)
            }
        }
        Spec::Filter(_) => {
            if u < 0.2 {
                (home, None)
            } else if u < 0.85 {
                (catalog(1 + rng.below(CATALOG_PAGES)), None)
            } else if u < 0.95 {
                if rng.uniform() < 0.5 {
                    (results(&wrong_text(rng, &[])), None)
                } else {
                    (item_page(rng.below(ITEMS.len())), None)
                }
            } else {
                (format!("/w/{seed}/item/404"), None)
            }
        }
    }
}

/// A snapshot of `path` with `typed` in the search box, as the simulator would show it.
pub fn snapshot(path: &str, typed: Option<String>) -> PageSnapshot {
    let page = World::page_at(path);
    let mut elements: Vec<_> = page.nodes().into_iter().map(|n| n.element).collect();
    if let (Some(text), Some(input)) = (typed, elements.iter_mut().find(|e| e.role == Role::Input)) {
        input.value = text;
    }
    PageSnapshot { url: path.to_string(), title: page.title(), elements }
}

/// The last calculator result in a state: for calculating tasks the right one (the teacher
/// answers), a wrong one (it recalculates) or none; rarely a stray one elsewhere (ignored).
fn note(rng: &mut Rng, world: &World, spec: &Spec) -> Option<Note> {
    let u = rng.uniform();
    match spec.calc_expr(world) {
        Some(expr) if u < 0.4 => Some(Note::of(&expr, &calc::run(&expr))),
        Some(expr) if u < 0.5 => Some(wrong_note(rng, &expr)),
        Some(_) => None,
        None if u < 0.03 => {
            let expr = goal::sample_expr(rng).text(false);
            Some(wrong_note(rng, &expr))
        }
        None => None,
    }
}

/// A labelled browser state.
#[derive(Debug, Clone)]
pub struct Labelled {
    pub observation: [u32; OBS_LEN],
    pub action: [u32; ACTION_LEN],
    pub goal: Goal,
    pub snapshot: PageSnapshot,
    pub note: Option<Note>,
}

/// A state and the teacher's action, in text form: `(goal, page, tool result, action)`. Uses
/// no tokenizer (the tokenizer itself is trained on these texts).
pub fn raw(rng: &mut Rng, split: Split) -> (Goal, PageSnapshot, Option<Note>, Action) {
    let world = World::new(rng.below(1 << 30) as u64);
    let f = family(rng);
    let goal = goal::sample(rng, &world, f, split);
    let spec = goal.spec.expect("sampled goals have a spec");
    let (path, typed) = state(rng, &world, &spec);
    let snapshot = snapshot(&path, typed);
    let note = note(rng, &world, &spec);
    let action = expert::act(&spec, &snapshot, note.as_ref());
    (goal, snapshot, note, action)
}

/// One labelled state for a goal worded with the templates of `split`.
pub fn labelled(rng: &mut Rng, split: Split) -> Labelled {
    loop {
        let (goal, snapshot, note, action) = raw(rng, split);
        let observation = obs::encode(&snapshot, &goal.text, note.as_ref());
        // Every generated action fits (checked by tests); skip the rare exception anyway.
        if let Some(action) = action.encode(text::ru()) {
            return Labelled { observation, action, goal, snapshot, note };
        }
    }
}

/// One training example (training templates only).
pub fn example(rng: &mut Rng) -> ([u32; OBS_LEN], [u32; ACTION_LEN]) {
    let l = labelled(rng, Split::Train);
    (l.observation, l.action)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::obs::{goal_tokens, note_tokens, page_tokens, NOTE_MAX};
    use crate::text::{ANSWER, BACK, CALC, CLICK, TYPE};

    #[test]
    fn covers_every_action_and_fits_the_observation() {
        let bpe = text::ru();
        let mut rng = Rng::new(0);
        let mut counts = [0usize; 5];
        let (mut max_page, mut max_goal, mut max_note) = (0, 0, 0);
        for _ in 0..3000 {
            let world = World::new(rng.below(1000) as u64);
            for split in [Split::Train, Split::HeldOut] {
                let f = family(&mut rng);
                let g = goal::sample(&mut rng, &world, f, split);
                let spec = g.spec.unwrap();
                let (path, typed) = state(&mut rng, &world, &spec);
                let snap = snapshot(&path, typed);
                max_page = max_page.max(page_tokens(&snap, bpe).len());
                max_goal = max_goal.max(goal_tokens(&g.text, bpe).len());
                let n = note(&mut rng, &world, &spec);
                max_note = max_note.max(note_tokens(n.as_ref(), bpe).len());
            }
            let act = labelled(&mut rng, Split::Train).action;
            assert!(Action::decode(&act, bpe).is_some(), "{}", bpe.describe(&act));
            counts[[CLICK, TYPE, BACK, ANSWER, CALC].iter().position(|&v| v == act[0]).unwrap()] += 1;
        }
        assert!(counts.iter().all(|&c| c > 50), "{counts:?}");
        assert!(max_goal < obs::GOAL_MAX, "instruction of {max_goal} tokens is truncated");
        assert!(max_note < NOTE_MAX, "tool result of {max_note} tokens is truncated");
        assert!(
            max_page + max_goal + max_note <= OBS_LEN,
            "page {max_page} + goal {max_goal} + tool {max_note} > {OBS_LEN}"
        );
    }
}
