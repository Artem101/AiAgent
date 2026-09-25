//! The sandbox web: a Russian-language shop with a search engine and a paged catalogue.
//!
//! Every world is generated from a seed: each product gets a random price, color, brand and
//! rating, its page lists them in a random order, and the catalogue lists products in a random
//! order. An agent therefore cannot memorise answers — it has to search, open pages and read.
//!
//! URLs (`{origin}` is the server address, or `http://sim.local` in the simulator):
//!
//! | URL | page |
//! |---|---|
//! | `{origin}/w/{seed}/` | search box, link to the catalogue |
//! | `{origin}/w/{seed}/search?q=лампа+стул` | results: products matched by word stem among distractors, with price and rating |
//! | `{origin}/w/{seed}/item/лампа` | product page: a table `attribute → value` |
//! | `{origin}/w/{seed}/catalog?page=2` | catalogue page: 4 products with price and color, link to the next page |

use super::{Element, Role};
use crate::kernels::rng::Rng;

/// A product with the case forms used in instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    /// Nominative (именительный) — how the shop names it.
    pub nom: &'static str,
    /// Genitive (родительный): «цена лампы».
    pub gen: &'static str,
    /// Accusative (винительный): «найди лампу».
    pub acc: &'static str,
}

const fn item(nom: &'static str, gen: &'static str, acc: &'static str) -> Item {
    Item { nom, gen, acc }
}

pub const ITEMS: [Item; 24] = [
    item("лампа", "лампы", "лампу"),
    item("стул", "стула", "стул"),
    item("стол", "стола", "стол"),
    item("телефон", "телефона", "телефон"),
    item("ноутбук", "ноутбука", "ноутбук"),
    item("камера", "камеры", "камеру"),
    item("часы", "часов", "часы"),
    item("велосипед", "велосипеда", "велосипед"),
    item("гитара", "гитары", "гитару"),
    item("чайник", "чайника", "чайник"),
    item("диван", "дивана", "диван"),
    item("рюкзак", "рюкзака", "рюкзак"),
    item("зонт", "зонта", "зонт"),
    item("зеркало", "зеркала", "зеркало"),
    item("подушка", "подушки", "подушку"),
    item("одеяло", "одеяла", "одеяло"),
    item("бутылка", "бутылки", "бутылку"),
    item("куртка", "куртки", "куртку"),
    item("шлем", "шлема", "шлем"),
    item("дрон", "дрона", "дрон"),
    item("принтер", "принтера", "принтер"),
    item("колонка", "колонки", "колонку"),
    item("мышь", "мыши", "мышь"),
    item("клавиатура", "клавиатуры", "клавиатуру"),
];

/// Colors: masculine nominative (as shown in the shop) and genitive («красного цвета»).
pub const COLORS: [(&str, &str); 8] = [
    ("красный", "красного"),
    ("синий", "синего"),
    ("зелёный", "зелёного"),
    ("чёрный", "чёрного"),
    ("белый", "белого"),
    ("жёлтый", "жёлтого"),
    ("оранжевый", "оранжевого"),
    ("фиолетовый", "фиолетового"),
];
pub const BRANDS: [&str; 8] = ["Acme", "Zenit", "Nova", "Orbit", "Pixel", "Vega", "Atlas", "Delta"];
pub const MAX_PRICE: u32 = 20;
pub const MAX_RATING: u32 = 5;
/// Products per catalogue page.
pub const CATALOG_PAGE: usize = 4;
pub const CATALOG_PAGES: usize = ITEMS.len() / CATALOG_PAGE;
/// Rows on a results page.
pub const RESULTS_PER_PAGE: usize = 4;

/// Interface texts.
pub const UI_SEARCH: &str = "Поиск";
pub const UI_FIND: &str = "Найти";
pub const UI_RESULTS: &str = "Результаты";
pub const UI_CATALOG: &str = "Каталог";
pub const UI_HOME: &str = "Главная";
pub const UI_NEXT: &str = "Далее";
pub const UI_NOT_FOUND: &str = "Страница не найдена";

/// A product attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Attr {
    Price,
    Color,
    Brand,
    Rating,
}

impl Attr {
    pub const ALL: [Attr; 4] = [Self::Price, Self::Color, Self::Brand, Self::Rating];

    /// Row label on a product page.
    pub fn label(self) -> &'static str {
        match self {
            Self::Price => "цена",
            Self::Color => "цвет",
            Self::Brand => "бренд",
            Self::Rating => "рейтинг",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.label() == label)
    }
}

pub fn fmt_price(p: u32) -> String {
    format!("{p} ₽")
}

pub fn fmt_rating(r: u32) -> String {
    format!("{r} ★")
}

/// The leading number of `"12 ₽"` / `"4 ★"`.
pub fn parse_number(text: &str) -> Option<u32> {
    text.split_whitespace().next()?.parse().ok()
}

/// Facts of one generated world.
#[derive(Debug, Clone)]
pub struct World {
    pub seed: u64,
    pub price: Vec<u32>,
    /// Index into [`COLORS`].
    pub color: Vec<usize>,
    /// Index into [`BRANDS`].
    pub brand: Vec<usize>,
    pub rating: Vec<u32>,
    /// Display order of the attribute rows on each product page.
    row_order: Vec<[Attr; 4]>,
    /// Product order in the catalogue.
    pub catalog: Vec<usize>,
}

pub fn shuffle<T>(rng: &mut Rng, xs: &mut [T]) {
    for i in (1..xs.len()).rev() {
        xs.swap(i, rng.below(i + 1));
    }
}

/// Index of the product named `name` (nominative, case-insensitive).
pub fn item_index(name: &str) -> Option<usize> {
    let name = name.trim().to_lowercase();
    ITEMS.iter().position(|i| i.nom == name)
}

/// Whether query word `w` names `item` by stem: at least 4 shared leading letters (3 for names
/// of up to 4 letters), e.g. «лампы» → лампа, «часов» → часы, «мыши» → мышь, but «стол» ≠ стул.
fn stem_match(w: &str, item: &Item) -> bool {
    let common = w.chars().zip(item.nom.chars()).take_while(|(x, y)| x == y).count();
    common >= if item.nom.chars().count() <= 4 { 3 } else { 4 }
}

impl World {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::stream(seed, 0x5EB, 0);
        let n = ITEMS.len();
        let mut w = Self {
            seed,
            price: Vec::with_capacity(n),
            color: Vec::with_capacity(n),
            brand: Vec::with_capacity(n),
            rating: Vec::with_capacity(n),
            row_order: Vec::with_capacity(n),
            catalog: (0..n).collect(),
        };
        for _ in 0..n {
            w.price.push(1 + rng.below(MAX_PRICE as usize) as u32);
            w.color.push(rng.below(COLORS.len()));
            w.brand.push(rng.below(BRANDS.len()));
            w.rating.push(1 + rng.below(MAX_RATING as usize) as u32);
            let mut order = Attr::ALL;
            shuffle(&mut rng, &mut order);
            w.row_order.push(order);
        }
        shuffle(&mut rng, &mut w.catalog);
        w
    }

    /// Displayed value of an attribute (`"12 ₽"`, `"красный"`, `"Vega"`, `"4 ★"`).
    pub fn display(&self, item: usize, attr: Attr) -> String {
        match attr {
            Attr::Price => fmt_price(self.price[item]),
            Attr::Color => COLORS[self.color[item]].0.to_string(),
            Attr::Brand => BRANDS[self.brand[item]].to_string(),
            Attr::Rating => fmt_rating(self.rating[item]),
        }
    }

    /// The answer an agent gives for an attribute (numbers without units).
    pub fn answer(&self, item: usize, attr: Attr) -> String {
        match attr {
            Attr::Price => self.price[item].to_string(),
            Attr::Rating => self.rating[item].to_string(),
            _ => self.display(item, attr),
        }
    }

    /// Products for `query`: every product named by a query word (in query order), then
    /// random distractors, shuffled — `RESULTS_PER_PAGE` rows.
    pub fn search(&self, query: &str) -> Vec<usize> {
        let q = query.trim().to_lowercase();
        let hash = q.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01B3));
        let mut rng = Rng::stream(self.seed, 0x5EA7C, hash);
        let mut hits: Vec<usize> = Vec::new();
        for w in q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            for (i, it) in ITEMS.iter().enumerate() {
                if stem_match(w, it) && !hits.contains(&i) {
                    hits.push(i);
                }
            }
        }
        hits.truncate(RESULTS_PER_PAGE);
        let mut pool: Vec<usize> = (0..ITEMS.len()).filter(|i| !hits.contains(i)).collect();
        shuffle(&mut rng, &mut pool);
        let fill = RESULTS_PER_PAGE - hits.len();
        hits.extend(pool.into_iter().take(fill));
        shuffle(&mut rng, &mut hits);
        hits
    }

    /// Page served at `path` (path and query of a URL, e.g. `/w/7/search?q=лампа`).
    pub fn page_at(path: &str) -> Page {
        let (path, query) = path.split_once('?').unwrap_or((path, ""));
        let path = url_decode(path);
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let seed = match parts.as_slice() {
            ["w", seed, ..] => seed.parse::<u64>().ok(),
            _ => None,
        };
        let Some(seed) = seed else { return Page::NotFound };
        let param = |key: &str| query.split('&').find_map(|kv| kv.strip_prefix(key)).map(url_decode);
        let world = World::new(seed);
        match parts[2..] {
            [] => Page::Home { seed },
            ["search"] => {
                let q = param("q=").unwrap_or_default();
                let hits = world.search(&q);
                Page::Results { world, query: q, hits }
            }
            ["item", name] => match item_index(name) {
                Some(item) => Page::Item { world, item },
                None => Page::NotFound,
            },
            ["catalog"] => match param("page=").unwrap_or_else(|| "1".into()).parse::<usize>() {
                Ok(page) if (1..=CATALOG_PAGES).contains(&page) => Page::Catalog { world, page },
                _ => Page::NotFound,
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

/// `application/x-www-form-urlencoded` encoding (UTF-8), as a browser submits a GET form.
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
                match std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(v) => {
                        out.push(v);
                        i += 2;
                    }
                    None => out.push(b'%'),
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
#[derive(Debug, Clone)]
pub enum Page {
    Home { seed: u64 },
    Results { world: World, query: String, hits: Vec<usize> },
    Item { world: World, item: usize },
    Catalog { world: World, page: usize },
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
    /// A link to a path on the same origin (already percent-encoded).
    Link(String),
    /// The submit button of the search form of world `seed`.
    Submit(u64),
}

fn node(role: Role, text: &str, target: Target) -> Node {
    Node { element: Element { role, text: text.to_string(), value: String::new() }, target }
}

fn item_href(seed: u64, item: usize) -> String {
    format!("/w/{seed}/item/{}", url_encode(ITEMS[item].nom))
}

fn catalog_href(seed: u64, page: usize) -> String {
    format!("/w/{seed}/catalog?page={page}")
}

impl Page {
    pub fn title(&self) -> String {
        match self {
            Self::Home { .. } => UI_SEARCH.into(),
            Self::Results { query, .. } => format!("{UI_RESULTS}: {query}"),
            Self::Item { item, .. } => ITEMS[*item].nom.into(),
            Self::Catalog { page, .. } => format!("{UI_CATALOG}, страница {page}"),
            Self::NotFound => UI_NOT_FOUND.into(),
        }
    }

    /// Product rows `(item, [value, value])` of a results page (price, rating) or a catalogue
    /// page (price, color).
    fn rows(&self) -> Vec<(usize, [String; 2])> {
        match self {
            Self::Results { world, hits, .. } => {
                hits.iter().map(|&i| (i, [world.display(i, Attr::Price), world.display(i, Attr::Rating)])).collect()
            }
            Self::Catalog { world, page } => world.catalog[(page - 1) * CATALOG_PAGE..page * CATALOG_PAGE]
                .iter()
                .map(|&i| (i, [world.display(i, Attr::Price), world.display(i, Attr::Color)]))
                .collect(),
            _ => vec![],
        }
    }

    /// Visible elements in document order — exactly what the DOM snapshot script returns for
    /// [`Page::html`] (checked against Chromium by `tests/browser.rs`). A table cell that
    /// holds a link is represented by the link.
    pub fn nodes(&self) -> Vec<Node> {
        let search_form = |seed: u64, value: &str| {
            let mut input = node(Role::Input, "", Target::None);
            input.element.value = value.to_string();
            [input, node(Role::Button, UI_FIND, Target::Submit(seed))]
        };
        let home = |seed: u64| node(Role::Link, UI_HOME, Target::Link(format!("/w/{seed}/")));
        let rows = |seed: u64, v: &mut Vec<Node>| {
            for (i, values) in self.rows() {
                v.push(node(Role::Link, ITEMS[i].nom, Target::Link(item_href(seed, i))));
                v.extend(values.iter().map(|x| node(Role::Value, x, Target::None)));
            }
        };
        match self {
            Self::Home { seed } => {
                let mut v = vec![node(Role::Heading, UI_SEARCH, Target::None)];
                v.extend(search_form(*seed, ""));
                v.push(node(Role::Link, UI_CATALOG, Target::Link(catalog_href(*seed, 1))));
                v
            }
            Self::Results { world, query, .. } => {
                let mut v = vec![node(Role::Heading, UI_RESULTS, Target::None)];
                v.extend(search_form(world.seed, query));
                rows(world.seed, &mut v);
                v.push(home(world.seed));
                v
            }
            Self::Item { world, item } => {
                let mut v = vec![node(Role::Heading, ITEMS[*item].nom, Target::None), home(world.seed)];
                for &attr in &world.row_order[*item] {
                    v.push(node(Role::Label, attr.label(), Target::None));
                    v.push(node(Role::Value, &world.display(*item, attr), Target::None));
                }
                v
            }
            Self::Catalog { world, page } => {
                let mut v = vec![node(Role::Heading, UI_CATALOG, Target::None)];
                rows(world.seed, &mut v);
                if *page < CATALOG_PAGES {
                    v.push(node(Role::Link, UI_NEXT, Target::Link(catalog_href(world.seed, page + 1))));
                }
                v.push(home(world.seed));
                v
            }
            Self::NotFound => vec![node(Role::Heading, UI_NOT_FOUND, Target::None)],
        }
    }

    /// HTML document of the page.
    pub fn html(&self) -> String {
        const STYLE: &str = "body{font-family:system-ui,sans-serif;max-width:640px;margin:32px auto;padding:0 16px}\
            body>a{display:inline-block;margin:8px 12px 8px 0}table{border-collapse:collapse;margin:12px 0}\
            th,td{border:1px solid #ccc;padding:4px 12px;text-align:left}";
        let form = |seed: u64, value: &str| {
            format!(
                "<form action=\"/w/{seed}/search\" method=\"get\"><input name=\"q\" value=\"{}\" autocomplete=\"off\"> \
                 <button type=\"submit\">{UI_FIND}</button></form>\n",
                html_escape(value)
            )
        };
        let table = |seed: u64| {
            let mut t = String::from("<table>\n");
            for (i, values) in self.rows() {
                t.push_str(&format!("<tr><td><a href=\"{}\">{}</a></td>", item_href(seed, i), ITEMS[i].nom));
                for x in values {
                    t.push_str(&format!("<td>{}</td>", html_escape(&x)));
                }
                t.push_str("</tr>\n");
            }
            t + "</table>\n"
        };
        let home = |seed: u64| format!("<a href=\"/w/{seed}/\">{UI_HOME}</a>\n");
        let body = match self {
            Self::Home { seed } => {
                format!(
                    "<h1>{UI_SEARCH}</h1>\n{}<a href=\"{}\">{UI_CATALOG}</a>\n",
                    form(*seed, ""),
                    catalog_href(*seed, 1)
                )
            }
            Self::Results { world, query, .. } => {
                format!("<h1>{UI_RESULTS}</h1>\n{}{}{}", form(world.seed, query), table(world.seed), home(world.seed))
            }
            Self::Item { world, item } => {
                let mut b = format!("<h1>{}</h1>\n{}<table>\n", ITEMS[*item].nom, home(world.seed));
                for &attr in &world.row_order[*item] {
                    let value = html_escape(&world.display(*item, attr));
                    b.push_str(&format!("<tr><th>{}</th><td>{value}</td></tr>\n", attr.label()));
                }
                b + "</table>\n"
            }
            Self::Catalog { world, page } => {
                let mut b = format!("<h1>{UI_CATALOG}</h1>\n{}", table(world.seed));
                if *page < CATALOG_PAGES {
                    b.push_str(&format!("<a href=\"{}\">{UI_NEXT}</a>\n", catalog_href(world.seed, page + 1)));
                }
                b + &home(world.seed)
            }
            Self::NotFound => format!("<h1>{UI_NOT_FOUND}</h1>\n"),
        };
        format!(
            "<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\"><title>{}</title><style>{STYLE}</style>\
             </head>\n<body>\n{body}</body></html>\n",
            html_escape(&self.title())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worlds_are_deterministic_and_searchable() {
        let (a, b, c) = (World::new(1), World::new(1), World::new(2));
        assert_eq!((&a.price, &a.catalog), (&b.price, &b.catalog));
        assert_ne!(a.price, c.price);
        for (i, it) in ITEMS.iter().enumerate() {
            for form in [it.nom, it.gen, it.acc] {
                let hits = a.search(form);
                assert_eq!(hits.len(), RESULTS_PER_PAGE);
                assert!(hits.contains(&i), "search «{form}» must find {}", it.nom);
            }
            // a product never matches another one by stem
            let own: Vec<usize> = (0..ITEMS.len()).filter(|&j| stem_match(it.nom, &ITEMS[j])).collect();
            assert_eq!(own, vec![i], "«{}» is ambiguous", it.nom);
        }
        let hits = a.search("Лампа, стул");
        assert!(hits.contains(&0) && hits.contains(&1));
        assert_eq!(a.search("ничего").len(), RESULTS_PER_PAGE);
    }

    #[test]
    fn routes() {
        assert!(matches!(World::page_at("/w/5/"), Page::Home { seed: 5 }));
        match World::page_at(&format!("/w/5/search?q={}", url_encode("Лампа стул"))) {
            Page::Results { query, hits, .. } => {
                assert_eq!(query, "Лампа стул");
                assert!(hits.contains(&0) && hits.contains(&1));
            }
            p => panic!("{p:?}"),
        }
        assert!(matches!(World::page_at(&format!("/w/5/item/{}", url_encode("лампа"))), Page::Item { item: 0, .. }));
        assert!(matches!(World::page_at("/w/5/item/лампа"), Page::Item { item: 0, .. }));
        assert!(matches!(World::page_at("/w/5/catalog"), Page::Catalog { page: 1, .. }));
        assert!(matches!(World::page_at("/w/5/catalog?page=7"), Page::NotFound));
        assert!(matches!(World::page_at("/w/5/item/единорог"), Page::NotFound));
        assert!(matches!(World::page_at("/favicon.ico"), Page::NotFound));
        assert_eq!(split_origin("http://127.0.0.1:80/w/1/?x"), ("http://127.0.0.1:80", "/w/1/?x"));
        assert_eq!(split_origin("http://sim.local"), ("http://sim.local", "/"));
        // catalogue pages chain via «Далее» and list every product exactly once
        let mut listed = vec![];
        for p in 1..=CATALOG_PAGES {
            let nodes = World::page_at(&format!("/w/5/catalog?page={p}")).nodes();
            listed.extend(
                nodes.iter().filter_map(|n| item_index(&n.element.text).filter(|_| n.element.role == Role::Link)),
            );
            assert_eq!(nodes.iter().any(|n| n.element.text == UI_NEXT), p < CATALOG_PAGES);
        }
        listed.sort();
        assert_eq!(listed, (0..ITEMS.len()).collect::<Vec<_>>());
    }

    #[test]
    fn url_codec_and_numbers() {
        for s in ["лампа", "red lamp", "a&b=c", "100%", "лампа стул"] {
            assert_eq!(url_decode(&url_encode(s)), s);
        }
        assert_eq!(url_encode("a b"), "a+b");
        assert_eq!(url_decode("%zz"), "%zz");
        assert_eq!(parse_number("12 ₽"), Some(12));
        assert_eq!(parse_number("4 ★"), Some(4));
        assert_eq!(parse_number("красный"), None);
    }
}
