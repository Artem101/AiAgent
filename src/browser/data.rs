//! Training data for `--task browser`: random browser states labelled by the expert.
//!
//! Rather than replaying only the expert's own 4-step trajectories, states are drawn from a
//! broader distribution — typed-but-not-submitted queries, wrong queries, wrong product pages,
//! error and blank pages — so the policy also learns to recover from its own mistakes
//! (behaviour cloning with DAgger-style state coverage).

use super::obs::{self, OBS_LEN};
use super::vocab::{token_str, Goal, ACTION_LEN, ATTR_TOKENS, ITEMS, ITEM_TOKENS};
use super::world::World;
use super::{expert, PageSnapshot, Role};
use crate::kernels::rng::Rng;

/// Out-of-vocabulary strings a user could leave in the search box.
const JUNK: [&str; 4] = ["lamps", "qwerty", "thing", "best"];

fn random_item(rng: &mut Rng) -> u32 {
    ITEM_TOKENS.start + rng.below(ITEMS.len()) as u32
}

fn other_item(rng: &mut Rng, not: u32) -> u32 {
    loop {
        let i = random_item(rng);
        if i != not {
            return i;
        }
    }
}

/// Text in the search box that is not the goal item.
fn wrong_text(rng: &mut Rng, goal: &Goal) -> String {
    if rng.uniform() < 0.2 {
        JUNK[rng.below(JUNK.len())].to_string()
    } else {
        token_str(other_item(rng, goal.item)).to_string()
    }
}

/// One `(observation, expert action)` pair.
pub fn example(rng: &mut Rng) -> ([u32; OBS_LEN], [u32; ACTION_LEN]) {
    let seed = rng.below(1 << 30) as u64;
    let goal = Goal { item: random_item(rng), attr: ATTR_TOKENS.start + rng.below(ATTR_TOKENS.len()) as u32 };
    let goal_name = token_str(goal.item).to_string();
    let r = rng.uniform();
    let (path, typed) = if r < 0.2 {
        // search page, possibly after typing
        let typed = match rng.uniform() {
            u if u < 0.5 => None,
            u if u < 0.8 => Some(goal_name.clone()),
            _ => Some(wrong_text(rng, &goal)),
        };
        (format!("/w/{seed}/"), typed)
    } else if r < 0.5 {
        // results of a right or wrong query, possibly with a new query typed in
        let query = if rng.uniform() < 0.7 { goal_name.clone() } else { wrong_text(rng, &goal) };
        let typed = match rng.uniform() {
            u if u < 0.8 => None,
            u if u < 0.9 => Some(goal_name.clone()),
            _ => Some(wrong_text(rng, &goal)),
        };
        (format!("/w/{seed}/search?q={query}"), typed)
    } else if r < 0.97 {
        // a product page: usually the right one
        let item = if rng.uniform() < 0.75 { goal.item } else { other_item(rng, goal.item) };
        (format!("/w/{seed}/item/{}", token_str(item)), None)
    } else {
        (format!("/w/{seed}/item/unknown"), None)
    };
    let page = World::page_at(&path);
    let mut elements: Vec<_> = page.nodes().into_iter().map(|n| n.element).collect();
    if let (Some(text), Some(input)) = (typed, elements.iter_mut().find(|e| e.role == Role::Input)) {
        input.value = text;
    }
    let snapshot = PageSnapshot { url: path, title: page.title(), elements };
    let observation = obs::encode(&snapshot, &goal);
    (observation, expert::act_tokens(&observation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::vocab::{Action, ANSWER, BACK, CLICK, TYPE};

    #[test]
    fn covers_every_kind_of_action() {
        let mut rng = Rng::new(0);
        let mut counts = [0usize; 4];
        for _ in 0..2000 {
            let (obs, act) = example(&mut rng);
            assert!(Action::decode(&act).is_some(), "{act:?}");
            assert_eq!(obs::parse(&obs).goal.map(|g| g.item), Some(obs[OBS_LEN - 2]));
            let k = [CLICK, TYPE, BACK, ANSWER].iter().position(|&v| v == act[0]).unwrap();
            counts[k] += 1;
        }
        assert!(counts.iter().all(|&c| c > 100), "{counts:?}");
    }
}
