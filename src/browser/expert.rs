//! Scripted teacher: the optimal next action as a function of the observation alone.
//!
//! ```text
//! product page of the goal item   → ANSWER <value in the row of the goal attribute>
//! product page of another item    → BACK
//! goal item among the links       → CLICK [link] <item>
//! search box already holds item   → CLICK [button] search
//! a search box on the page        → TYPE [input] <item>
//! otherwise                       → BACK
//! ```
//!
//! Because it only looks at observation tokens it is also the target function of
//! `Task::Browser` (`Task::apply`), and it recovers from any state the model can reach
//! (wrong query, wrong page, stray navigation).

use super::obs::{self, ObsView};
use super::vocab::{self, Action, ACTION_LEN, SEARCH};
use super::Role;

/// Expert action for a decoded observation.
pub fn act(view: &ObsView) -> Action {
    let Some(goal) = view.goal else { return Action::Back };
    if let Some(&[head]) = view.first(Role::Heading) {
        if vocab::is_item(head) {
            if head != goal.item {
                return Action::Back;
            }
            let row = view.elements.windows(2).find_map(|w| match (&w[0], &w[1]) {
                ((Role::Label, a), (Role::Value, v)) if a.as_slice() == [goal.attr] && v.len() == 1 => Some(v[0]),
                _ => None,
            });
            return row.map_or(Action::Back, |word| Action::Answer { word });
        }
    }
    if view.has(Role::Link, goal.item) {
        return Action::Click { role: Role::Link, word: goal.item };
    }
    match view.first(Role::Input) {
        Some(value) if value == [goal.item] && view.has(Role::Button, SEARCH) => {
            Action::Click { role: Role::Button, word: SEARCH }
        }
        Some(_) => Action::Type { word: goal.item },
        None => Action::Back,
    }
}

/// Expert action tokens for observation tokens.
pub fn act_tokens(observation: &[u32]) -> [u32; ACTION_LEN] {
    act(&obs::parse(observation)).encode()
}
