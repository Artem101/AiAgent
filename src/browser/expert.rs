//! Scripted teacher: the next action from the page snapshot and the structured goal.
//!
//! ```text
//! lookup  product page of the item          → ANSWER <value of the attribute row>
//!         value visible in a results / catalogue row → ANSWER it (price, rating / price, color)
//!         item among the links              → CLICK [link] <item>
//!         search box holds the item         → CLICK [button] Найти
//!         a search box                      → TYPE <item>
//! compare both products listed with values  → ANSWER <the cheaper / pricier / better / worse>
//!         search box holds «a b»            → CLICK [button] Найти
//!         a search box                      → TYPE «a b»
//! filter  a catalogue row matches           → ANSWER <product>
//!         otherwise                         → CLICK [link] Далее (last page: Главная, start over)
//!         search page                       → CLICK [link] Каталог
//! calc    the tool result is for the expression → ANSWER <result>
//!         otherwise                         → CALC <expression as written in the question>
//! total   both prices listed, result known  → ANSWER <result>
//!         both prices listed                → CALC «price a + price b» (or the difference)
//!         otherwise                         → search «a b», as for compare
//! chat                                      → ANSWER <the fixed reply>
//! any     wrong product page                → BACK
//!         page of another task              → CLICK [link] Главная
//!         no way forward                    → BACK
//! ```
//!
//! The teacher recovers from any state the model can reach (wrong query, wrong page, stray
//! navigation), which is what the training states in [`super::data`] cover.

use super::action::Action;
use super::goal::{CalcForm, Cond, Spec};
use super::obs::Note;
use super::world::{
    item_index, parse_number, Attr, COLORS, ITEMS, UI_CATALOG, UI_FIND, UI_HOME, UI_NEXT, UI_RESULTS, UI_SEARCH,
};
use super::{PageSnapshot, Role};

/// Kind of the current page, recognised from its heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    Search,
    Results,
    Catalog,
    Item(usize),
    Other,
}

/// What the teacher reads off a snapshot.
#[derive(Debug, Clone)]
pub struct View<'a> {
    pub kind: PageKind,
    pub input: Option<&'a str>,
    /// Product rows `(item, values)` of a results or catalogue table.
    pub rows: Vec<(usize, Vec<&'a str>)>,
    /// `attribute → value` rows of a product page.
    pub table: Vec<(Attr, &'a str)>,
    pub links: Vec<&'a str>,
    pub has_find: bool,
}

impl<'a> View<'a> {
    pub fn of(snap: &'a PageSnapshot) -> Self {
        let els = &snap.elements;
        let heading = els.iter().find(|e| e.role == Role::Heading).map_or("", |e| e.text.as_str());
        let kind = match heading {
            UI_SEARCH => PageKind::Search,
            UI_RESULTS => PageKind::Results,
            UI_CATALOG => PageKind::Catalog,
            h => item_index(h).map_or(PageKind::Other, PageKind::Item),
        };
        let mut rows = Vec::new();
        let mut table = Vec::new();
        for (i, e) in els.iter().enumerate() {
            let values = || els[i + 1..].iter().take_while(|v| v.role == Role::Value).map(|v| v.text.as_str());
            match e.role {
                Role::Link => {
                    if let Some(item) = item_index(&e.text) {
                        let vals: Vec<&str> = values().collect();
                        if !vals.is_empty() {
                            rows.push((item, vals));
                        }
                    }
                }
                Role::Label => {
                    if let (Some(attr), Some(v)) = (Attr::from_label(&e.text), values().next()) {
                        table.push((attr, v));
                    }
                }
                _ => {}
            }
        }
        Self {
            kind,
            input: els.iter().find(|e| e.role == Role::Input).map(|e| e.value.as_str()),
            rows,
            table,
            links: els.iter().filter(|e| e.role == Role::Link).map(|e| e.text.as_str()).collect(),
            has_find: els.iter().any(|e| e.role == Role::Button && e.text == UI_FIND),
        }
    }

    fn row(&self, item: usize) -> Option<&[&'a str]> {
        self.rows.iter().find(|(i, _)| *i == item).map(|(_, v)| v.as_slice())
    }

    fn has_link(&self, text: &str) -> bool {
        let text = text.to_lowercase();
        self.links.iter().any(|l| l.to_lowercase() == text)
    }
}

fn click(role: Role, text: &str) -> Action {
    Action::Click { role, text: text.to_string() }
}

fn answer(text: impl Into<String>) -> Action {
    Action::Answer { text: text.into() }
}

/// Answer text for a displayed value (`"12 ₽"` → `"12"`).
fn answer_value(attr: Attr, shown: &str) -> Action {
    match attr {
        Attr::Price | Attr::Rating => answer(parse_number(shown).map_or(shown.to_string(), |n| n.to_string())),
        _ => answer(shown),
    }
}

/// Search for `query`: submit if it is already typed, type it otherwise.
fn search(view: &View, query: &str) -> Action {
    match view.input {
        Some(v) if v.trim() == query && view.has_find => click(Role::Button, UI_FIND),
        Some(_) => Action::Type { text: query.to_string() },
        None => leave(view),
    }
}

/// Back to the start page (or history back when there is no link).
fn leave(view: &View) -> Action {
    if view.has_link(UI_HOME) {
        click(Role::Link, UI_HOME)
    } else {
        Action::Back
    }
}

/// Column of `attr` in a results row (price, rating) or a catalogue row (price, color).
fn column(kind: PageKind, attr: Attr) -> Option<usize> {
    match (kind, attr) {
        (_, Attr::Price) => Some(0),
        (PageKind::Results, Attr::Rating) | (PageKind::Catalog, Attr::Color) => Some(1),
        _ => None,
    }
}

/// `ANSWER` with the result when `note` is for `expr`, `CALC expr` otherwise.
fn calculate(expr: String, note: Option<&Note>) -> Action {
    match note {
        Some(n) if n.is_for(&expr) => answer(n.result.clone()),
        _ => Action::Calc { text: expr },
    }
}

/// The teacher's action for `spec` on the page of `snap`, given the last tool result `note`.
pub fn act(spec: &Spec, snap: &PageSnapshot, note: Option<&Note>) -> Action {
    let view = View::of(snap);
    match *spec {
        Spec::Calc { expr, form } => calculate(expr.text(form != CalcForm::Compact), note),
        Spec::Chat(c) => answer(c.reply()),
        Spec::Total { a, b, kind } => {
            let query = format!("{} {}", ITEMS[a].nom, ITEMS[b].nom);
            match view.kind {
                PageKind::Results => {
                    let price = |i: usize| view.row(i).and_then(|v| v.first().copied()).and_then(parse_number);
                    match (price(a), price(b)) {
                        (Some(x), Some(y)) => calculate(Spec::total_expr(kind, x, y), note),
                        _ => search(&view, &query),
                    }
                }
                PageKind::Search => search(&view, &query),
                PageKind::Item(_) | PageKind::Catalog => leave(&view),
                PageKind::Other => Action::Back,
            }
        }
        Spec::Lookup { item, attr } => {
            let name = ITEMS[item].nom;
            match view.kind {
                PageKind::Item(i) if i == item => {
                    view.table.iter().find(|(a, _)| *a == attr).map_or(Action::Back, |(_, v)| answer_value(attr, v))
                }
                PageKind::Item(_) => Action::Back,
                PageKind::Results | PageKind::Catalog => match (view.row(item), column(view.kind, attr)) {
                    (Some(vals), Some(c)) if c < vals.len() => answer_value(attr, vals[c]),
                    (Some(_), _) => click(Role::Link, name),
                    _ if view.kind == PageKind::Results => search(&view, name),
                    _ => leave(&view),
                },
                PageKind::Search => search(&view, name),
                PageKind::Other => Action::Back,
            }
        }
        Spec::Compare { a, b, attr, most } => {
            let query = format!("{} {}", ITEMS[a].nom, ITEMS[b].nom);
            match view.kind {
                PageKind::Results => {
                    let val = |i: usize| {
                        view.row(i).and_then(|v| column(PageKind::Results, attr).and_then(|c| v.get(c)).copied())
                    };
                    match (val(a).and_then(parse_number), val(b).and_then(parse_number)) {
                        (Some(x), Some(y)) => answer(ITEMS[if (x > y) == most { a } else { b }].nom),
                        _ => search(&view, &query),
                    }
                }
                PageKind::Search => search(&view, &query),
                PageKind::Item(_) | PageKind::Catalog => leave(&view),
                PageKind::Other => Action::Back,
            }
        }
        Spec::Filter(cond) => match view.kind {
            PageKind::Catalog => {
                let ok = |vals: &[&str]| match cond {
                    Cond::PriceBelow(x) => vals.first().and_then(|v| parse_number(v)).is_some_and(|p| p < x),
                    Cond::PriceAbove(x) => vals.first().and_then(|v| parse_number(v)).is_some_and(|p| p > x),
                    Cond::Color(c) => vals.get(1) == Some(&COLORS[c].0),
                };
                match view.rows.iter().find(|(_, v)| ok(v)) {
                    Some((i, _)) => answer(ITEMS[*i].nom),
                    None if view.has_link(UI_NEXT) => click(Role::Link, UI_NEXT),
                    None => leave(&view), // last page: start over from the first
                }
            }
            PageKind::Search if view.has_link(UI_CATALOG) => click(Role::Link, UI_CATALOG),
            PageKind::Search | PageKind::Other => Action::Back,
            PageKind::Results | PageKind::Item(_) => leave(&view),
        },
    }
}
