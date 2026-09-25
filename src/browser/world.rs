//! The sandbox web: a search engine over a product catalogue.
//!
//! Every world is generated from a seed: each product gets a random price, color, brand and
//! rating, and its page lists them in a random order. An agent therefore cannot memorise
//! answers — it has to search, open the right page and read it.
//!
//! URLs (`{origin}` is the server address, or `http://sim.local` in the simulator):
//!
//! | URL | page |
//! |---|---|
//! | `{origin}/w/{seed}/` | search box |
//! | `{origin}/w/{seed}/search?q=lamp` | results: 4 product links (the match, if any, among distractors) |
//! | `{origin}/w/{seed}/item/lamp` | product page: a table `attribute → value` |

use super::vocab::{self, token_str, BRANDS, COLORS, ITEMS, ITEM_TOKENS};
use super::{Element, Role};
use crate::kernels::rng::Rng;

/// Number of links on a results page.
pub const RESULTS_PER_PAGE: usize = 4;

/// Facts of one generated world.
#[derive(Debug, Clone)]
pub struct World {
    pub seed: u64,
    /// `values[item][attr]` — value token of each attribute.
    values: Vec<[u32; 4]>,
    /// Display order of the attribute rows on each product page.
    row_order: Vec<[usize; 4]>,
}

fn shuffle<T>(rng: &mut Rng, xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, rng.below(i + 1));
    }
}

impl World {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::stream(seed, 0x5EB, 0);
        let mut values = Vec::with_capacity(ITEMS.len());
        let mut row_order = Vec::with_capacity(ITEMS.len());
        for _ in 0..ITEMS.len() {
            let price = vocab::number_token(1 + rng.below(20));
            let color = vocab::COLOR_TOKENS.start + rng.below(COLORS.len()) as u32;
            let brand = vocab::BRAND_TOKENS.start + rng.below(BRANDS.len()) as u32;
            let rating = vocab::number_token(1 + rng.below(5));
            values.push([price, color, brand, rating]);
            let mut order = [0, 1, 2, 3];
            shuffle(&mut rng, &mut order);
            row_order.push(order);
        }
        Self { seed, values, row_order }
    }

    /// Value token of attribute `attr` of `item` (both tokens).
    pub fn value(&self, item: u32, attr: u32) -> u32 {
        self.values[(item - ITEM_TOKENS.start) as usize][(attr - vocab::ATTR_TOKENS.start) as usize]
    }

    /// Items returned for `query`: the exact match (if any) among random distractors, in an
    /// order that depends on the world and the query.
    pub fn search(&self, query: &str) -> Vec<u32> {
        let q = query.trim().to_lowercase();
        let hash = q.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01B3));
        let mut rng = Rng::stream(self.seed, 0x5EA7C, hash);
        let hit = ITEMS.iter().position(|&i| i == q);
        let mut pool: Vec<usize> = (0..ITEMS.len()).filter(|&i| Some(i) != hit).collect();
        shuffle(&mut rng, &mut pool);
        let mut out: Vec<usize> = hit.into_iter().chain(pool).take(RESULTS_PER_PAGE).collect();
        shuffle(&mut rng, &mut out);
        out.into_iter().map(|i| ITEM_TOKENS.start + i as u32).collect()
    }

    /// Page served at `path` (path and query of a URL, e.g. `/w/7/search?q=lamp`).
    pub fn page_at(path: &str) -> Page {
        let (path, query) = path.split_once('?').unwrap_or((path, ""));
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let seed = match parts.as_slice() {
            ["w", seed, ..] => seed.parse::<u64>().ok(),
            _ => None,
        };
        let Some(seed) = seed else { return Page::NotFound };
        let world = World::new(seed);
        match parts[2..] {
            [] => Page::Home { seed },
            ["search"] => {
                let q = query.split('&').find_map(|kv| kv.strip_prefix("q=")).map(url_decode).unwrap_or_default();
                let hits = world.search(&q);
                Page::Results { seed, query: q, hits }
            }
            ["item", name] => match vocab::word_id(name).filter(|&t| vocab::is_item(t)) {
                Some(item) => {
                    let i = (item - ITEM_TOKENS.start) as usize;
                    let rows = world.row_order[i]
                        .iter()
                        .map(|&a| (vocab::ATTR_TOKENS.start + a as u32, world.values[i][a]))
                        .collect();
                    Page::Item { seed, item, rows }
                }
                None => Page::NotFound,
            },
            _ => Page::NotFound,
        }
    }
}

/// URL of the search page of world `seed`.
pub fn home_url(origin: &str, seed: u64) -> String {
    format!("{origin}/w/{seed}/")
}

/// Splits `http://host:port/path?q` into `("http://host:port", "/path?q")`.
pub fn split_origin(url: &str) -> (&str, &str) {
    let after_scheme = url.find("://").map(|i| i + 3).unwrap_or(0);
    match url[after_scheme..].find('/') {
        Some(i) => url.split_at(after_scheme + i),
        None => (url, "/"),
    }
}

/// `application/x-www-form-urlencoded` encoding, as a browser submits a GET form.
pub fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'*' | b'-' | b'.' | b'_' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

pub fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                match u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("zz"), 16) {
                    Ok(v) => {
                        out.push(v);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A page of the sandbox web.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Page {
    Home {
        seed: u64,
    },
    Results {
        seed: u64,
        query: String,
        hits: Vec<u32>,
    },
    /// `rows` are `(attribute, value)` tokens in display order.
    Item {
        seed: u64,
        item: u32,
        rows: Vec<(u32, u32)>,
    },
    NotFound,
}

/// An element of a [`Page`] together with where clicking it leads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub element: Element,
    pub target: Target,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    None,
    /// A link to a path on the same origin.
    Link(String),
    /// The submit button of the search form of world `seed`.
    Submit(u64),
}

fn node(role: Role, text: &str, target: Target) -> Node {
    Node { element: Element { role, text: text.to_string(), value: String::new() }, target }
}

impl Page {
    pub fn title(&self) -> String {
        match self {
            Self::Home { .. } => "search".into(),
            Self::Results { query, .. } => format!("results: {query}"),
            Self::Item { item, .. } => token_str(*item).into(),
            Self::NotFound => "not found".into(),
        }
    }

    /// Visible elements in document order — exactly what the DOM snapshot script returns
    /// for [`Page::html`] (checked against Chromium by `tests/browser.rs`).
    pub fn nodes(&self) -> Vec<Node> {
        let search_form = |seed: u64, value: &str| {
            let mut input = node(Role::Input, "", Target::None);
            input.element.value = value.to_string();
            [input, node(Role::Button, "search", Target::Submit(seed))]
        };
        let home = |seed: u64| node(Role::Link, "home", Target::Link(format!("/w/{seed}/")));
        match self {
            Self::Home { seed } => {
                let mut v = vec![node(Role::Heading, "search", Target::None)];
                v.extend(search_form(*seed, ""));
                v
            }
            Self::Results { seed, query, hits } => {
                let mut v = vec![node(Role::Heading, "results", Target::None)];
                v.extend(search_form(*seed, query));
                for &h in hits {
                    let name = token_str(h);
                    v.push(node(Role::Link, name, Target::Link(format!("/w/{seed}/item/{name}"))));
                }
                v.push(home(*seed));
                v
            }
            Self::Item { seed, item, rows } => {
                let mut v = vec![node(Role::Heading, token_str(*item), Target::None), home(*seed)];
                for &(attr, value) in rows {
                    v.push(node(Role::Label, token_str(attr), Target::None));
                    v.push(node(Role::Value, token_str(value), Target::None));
                }
                v
            }
            Self::NotFound => vec![node(Role::Heading, "not found", Target::None)],
        }
    }

    /// HTML document of the page.
    pub fn html(&self) -> String {
        const STYLE: &str = "body{font-family:system-ui,sans-serif;max-width:640px;margin:32px auto;padding:0 16px}\
            a{display:block;margin:6px 0}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px 12px;text-align:left}";
        let mut body = String::new();
        let form = |seed: u64, value: &str| {
            format!(
                "<form action=\"/w/{seed}/search\" method=\"get\"><input name=\"q\" value=\"{}\" autocomplete=\"off\"> \
                 <button type=\"submit\">search</button></form>\n",
                html_escape(value)
            )
        };
        match self {
            Self::Home { seed } => {
                body.push_str("<h1>search</h1>\n");
                body.push_str(&form(*seed, ""));
            }
            Self::Results { seed, query, hits } => {
                body.push_str("<h1>results</h1>\n");
                body.push_str(&form(*seed, query));
                body.push_str("<nav>\n");
                for &h in hits {
                    let name = token_str(h);
                    body.push_str(&format!("<a href=\"/w/{seed}/item/{name}\">{name}</a>\n"));
                }
                body.push_str(&format!("</nav>\n<a href=\"/w/{seed}/\">home</a>\n"));
            }
            Self::Item { seed, item, rows } => {
                body.push_str(&format!("<h1>{}</h1>\n<a href=\"/w/{seed}/\">home</a>\n<table>\n", token_str(*item)));
                for &(attr, value) in rows {
                    body.push_str(&format!("<tr><th>{}</th><td>{}</td></tr>\n", token_str(attr), token_str(value)));
                }
                body.push_str("</table>\n");
            }
            Self::NotFound => body.push_str("<h1>not found</h1>\n"),
        }
        format!(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{}</title><style>{STYLE}</style></head>\n<body>\n{body}</body></html>\n",
            html_escape(&self.title())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worlds_are_deterministic_and_different() {
        let (a, b, c) = (World::new(1), World::new(1), World::new(2));
        let lamp = vocab::word_id("lamp").unwrap();
        let price = vocab::word_id("price").unwrap();
        assert_eq!(a.value(lamp, price), b.value(lamp, price));
        let differs = ITEM_TOKENS.into_iter().any(|i| a.value(i, price) != c.value(i, price));
        assert!(differs);
        // exact match always among the results, results are distinct items
        for (i, name) in ITEMS.iter().enumerate() {
            let hits = a.search(name);
            assert_eq!(hits.len(), RESULTS_PER_PAGE);
            assert!(hits.contains(&(ITEM_TOKENS.start + i as u32)));
            let mut dedup = hits.clone();
            dedup.sort();
            dedup.dedup();
            assert_eq!(dedup.len(), RESULTS_PER_PAGE);
        }
        assert_eq!(a.search("nothing").len(), RESULTS_PER_PAGE);
    }

    #[test]
    fn routes() {
        assert_eq!(World::page_at("/w/5/"), Page::Home { seed: 5 });
        assert_eq!(World::page_at("/w/5"), Page::Home { seed: 5 });
        match World::page_at("/w/5/search?q=Lamp") {
            Page::Results { query, hits, .. } => {
                assert_eq!(query, "Lamp");
                assert!(hits.contains(&vocab::word_id("lamp").unwrap()));
            }
            p => panic!("{p:?}"),
        }
        match World::page_at("/w/5/item/lamp") {
            Page::Item { rows, .. } => assert_eq!(rows.len(), 4),
            p => panic!("{p:?}"),
        }
        assert_eq!(World::page_at("/w/5/item/unicorn"), Page::NotFound);
        assert_eq!(World::page_at("/favicon.ico"), Page::NotFound);
        assert_eq!(split_origin("http://127.0.0.1:80/w/1/?x"), ("http://127.0.0.1:80", "/w/1/?x"));
        assert_eq!(split_origin("http://sim.local"), ("http://sim.local", "/"));
    }

    #[test]
    fn url_codec() {
        for s in ["lamp", "red lamp", "a&b=c", "лампа", "100%"] {
            assert_eq!(url_decode(&url_encode(s)), s);
        }
        assert_eq!(url_encode("red lamp"), "red+lamp");
        assert_eq!(url_decode("%zz"), "%zz");
    }
}
