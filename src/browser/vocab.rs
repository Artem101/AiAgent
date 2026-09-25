//! Closed word-level vocabulary shared by observations, actions and goals, plus the
//! [`Action`] and [`Goal`] types and a small English/Russian question parser.
//!
//! Token ids are positions in [`TOKENS`]: control tokens first, then element roles, action
//! verbs, and finally the words that can appear on pages of the sandbox web.

use std::ops::Range;

use candle_core::{bail, Result};

use super::Role;

/// Products listed in the sandbox catalogue.
pub const ITEMS: [&str; 24] = [
    "lamp", "chair", "table", "phone", "laptop", "camera", "watch", "bike", "guitar", "kettle", "sofa", "backpack",
    "umbrella", "mirror", "pillow", "blanket", "bottle", "jacket", "helmet", "drone", "printer", "speaker", "mouse",
    "keyboard",
];
/// Attributes shown on every product page.
pub const ATTRS: [&str; 4] = ["price", "color", "brand", "rating"];
pub const COLORS: [&str; 8] = ["red", "blue", "green", "black", "white", "yellow", "orange", "purple"];
pub const BRANDS: [&str; 8] = ["acme", "zenit", "nova", "orbit", "pixel", "vega", "atlas", "delta"];
/// Numeric values (prices 1…20, ratings 1…5 — the ranges overlap on purpose).
pub const NUMBERS: [&str; 20] =
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"];

/// Every token, indexed by id.
#[rustfmt::skip]
pub const TOKENS: [&str; VOCAB_SIZE] = [
    // control
    "<pad>", "<goal>", "<end>", "<none>", "<empty>", "<unk>",
    // element roles
    "[head]", "[link]", "[input]", "[button]", "[label]", "[value]", "[text]",
    // action verbs
    "CLICK", "TYPE", "BACK", "ANSWER",
    // interface words
    "search", "results", "home",
    // attributes
    "price", "color", "brand", "rating",
    // items
    "lamp", "chair", "table", "phone", "laptop", "camera", "watch", "bike", "guitar", "kettle", "sofa", "backpack",
    "umbrella", "mirror", "pillow", "blanket", "bottle", "jacket", "helmet", "drone", "printer", "speaker", "mouse",
    "keyboard",
    // colors
    "red", "blue", "green", "black", "white", "yellow", "orange", "purple",
    // brands
    "acme", "zenit", "nova", "orbit", "pixel", "vega", "atlas", "delta",
    // numbers
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
];

pub const VOCAB_SIZE: usize = 84;

pub const PAD: u32 = 0;
/// Separates the page from the goal in an observation.
pub const GOAL: u32 = 1;
pub const END: u32 = 2;
pub const NONE: u32 = 3;
/// Value of an empty text field.
pub const EMPTY: u32 = 4;
/// A word outside the vocabulary.
pub const UNK: u32 = 5;

pub const CLICK: u32 = 13;
pub const TYPE: u32 = 14;
pub const BACK: u32 = 15;
pub const ANSWER: u32 = 16;

pub const SEARCH: u32 = 17;
pub const RESULTS: u32 = 18;
pub const HOME: u32 = 19;

/// First word token (everything below is a control token, a role or a verb).
pub const WORDS_START: u32 = 17;
pub const ATTR_TOKENS: Range<u32> = 20..24;
pub const ITEM_TOKENS: Range<u32> = 24..48;
pub const COLOR_TOKENS: Range<u32> = 48..56;
pub const BRAND_TOKENS: Range<u32> = 56..64;
/// `NUMBER_TOKENS.start + (n − 1)` is the token of the number `n`.
pub const NUMBER_TOKENS: Range<u32> = 64..84;

/// Surface string of a token.
pub fn token_str(token: u32) -> &'static str {
    TOKENS.get(token as usize).copied().unwrap_or("<?>")
}

/// Id of a word (case-insensitive), `None` when it is not a word of the vocabulary.
pub fn word_id(word: &str) -> Option<u32> {
    let w = word.to_lowercase();
    (WORDS_START as usize..VOCAB_SIZE).find(|&i| TOKENS[i] == w).map(|i| i as u32)
}

/// Token of the number `n` (1 ≤ n ≤ 20).
pub fn number_token(n: usize) -> u32 {
    debug_assert!((1..=NUMBERS.len()).contains(&n));
    NUMBER_TOKENS.start + (n - 1) as u32
}

pub fn is_item(token: u32) -> bool {
    ITEM_TOKENS.contains(&token)
}

pub fn is_attr(token: u32) -> bool {
    ATTR_TOKENS.contains(&token)
}

/// Space-separated surface form of a token sequence (padding omitted).
pub fn describe(tokens: &[u32]) -> String {
    tokens.iter().filter(|&&t| t != PAD).map(|&t| token_str(t)).collect::<Vec<_>>().join(" ")
}

/// Number of tokens in an action.
pub const ACTION_LEN: usize = 4;

/// One browser action, encoded as `[verb, role, word, <end>]`:
///
/// | action | tokens |
/// |---|---|
/// | click a link / button by its text | `CLICK [link] lamp <end>`, `CLICK [button] search <end>` |
/// | type into the text field | `TYPE [input] lamp <end>` |
/// | go back in history | `BACK <none> <none> <end>` |
/// | finish with an answer | `ANSWER <none> 17 <end>` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Click { role: Role, word: u32 },
    Type { word: u32 },
    Back,
    Answer { word: u32 },
}

impl Action {
    pub fn encode(&self) -> [u32; ACTION_LEN] {
        match *self {
            Self::Click { role, word } => [CLICK, role.token(), word, END],
            Self::Type { word } => [TYPE, Role::Input.token(), word, END],
            Self::Back => [BACK, NONE, NONE, END],
            Self::Answer { word } => [ANSWER, NONE, word, END],
        }
    }

    /// Parses model output. The trailing `<end>` carries no information and is not checked;
    /// everything else must be well-formed.
    pub fn decode(tokens: &[u32]) -> Option<Self> {
        let &[verb, role, word, ..] = tokens else { return None };
        let is_word = word >= WORDS_START && (word as usize) < VOCAB_SIZE;
        match verb {
            CLICK if is_word => match Role::from_token(role)? {
                r @ (Role::Link | Role::Button) => Some(Self::Click { role: r, word }),
                _ => None,
            },
            TYPE if is_word && role == Role::Input.token() => Some(Self::Type { word }),
            BACK => Some(Self::Back),
            ANSWER if is_word => Some(Self::Answer { word }),
            _ => None,
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Click { role, word } => write!(f, "CLICK {} \"{}\"", role.name(), token_str(word)),
            Self::Type { word } => write!(f, "TYPE \"{}\"", token_str(word)),
            Self::Back => write!(f, "BACK"),
            Self::Answer { word } => write!(f, "ANSWER \"{}\"", token_str(word)),
        }
    }
}

/// What the agent has to find: attribute `attr` of product `item`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goal {
    pub item: u32,
    pub attr: u32,
}

/// Russian word stems for the items (same order as [`ITEMS`]).
const ITEM_STEMS_RU: [&[&str]; 24] = [
    &["ламп"],
    &["стул"],
    &["стол"],
    &["телефон", "смартфон"],
    &["ноутбук"],
    &["камер", "фотоаппарат"],
    &["часы", "часов", "часам"],
    &["велосипед", "байк"],
    &["гитар"],
    &["чайник"],
    &["диван"],
    &["рюкзак"],
    &["зонт"],
    &["зеркал"],
    &["подушк"],
    &["одеял", "плед"],
    &["бутыл"],
    &["куртк"],
    &["шлем"],
    &["дрон", "квадрокоптер"],
    &["принтер"],
    &["колонк", "динамик"],
    &["мыш"],
    &["клавиатур"],
];

/// Stems (English and Russian) that name an attribute, in [`ATTRS`] order.
const ATTR_STEMS: [&[&str]; 4] = [
    &["price", "cost", "цен", "стои"],
    &["color", "colour", "цвет"],
    &["brand", "make", "manufacturer", "бренд", "марк", "производ", "фирм"],
    &["rating", "stars", "score", "рейтинг", "оценк", "звезд"],
];

impl Goal {
    /// Extracts the goal from a free-form question in English or Russian, e.g.
    /// `"what is the price of the lamp?"` or `"сколько стоит лампа"`.
    pub fn parse(question: &str) -> Result<Self> {
        let words: Vec<String> = question
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect();
        let find = |exact: &dyn Fn(&str) -> Option<usize>, stems: &[&[&str]]| {
            words
                .iter()
                .find_map(|w| exact(w).or_else(|| stems.iter().position(|ss| ss.iter().any(|s| w.starts_with(s)))))
        };
        let item = find(&|w| ITEMS.iter().position(|&i| i == w), &ITEM_STEMS_RU);
        let attr = find(&|w| ATTRS.iter().position(|&a| a == w), &ATTR_STEMS);
        match (item, attr) {
            (Some(i), Some(a)) => Ok(Self { item: ITEM_TOKENS.start + i as u32, attr: ATTR_TOKENS.start + a as u32 }),
            (None, _) => bail!("no known product in the question (known: {})", ITEMS.join(", ")),
            (_, None) => bail!("no known attribute in the question (known: {})", ATTRS.join(", ")),
        }
    }

    /// Goal tokens as they appear at the end of an observation: `[<goal>, item, attr]`.
    pub fn tokens(&self) -> [u32; 3] {
        [GOAL, self.item, self.attr]
    }
}

impl std::fmt::Display for Goal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of {}", token_str(self.attr), token_str(self.item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_table_is_consistent() {
        assert_eq!(TOKENS.len(), VOCAB_SIZE);
        assert_eq!(token_str(CLICK), "CLICK");
        assert_eq!(token_str(ANSWER), "ANSWER");
        assert_eq!(token_str(SEARCH), "search");
        assert_eq!(token_str(HOME), "home");
        for (range, words) in [
            (ATTR_TOKENS, &ATTRS[..]),
            (ITEM_TOKENS, &ITEMS[..]),
            (COLOR_TOKENS, &COLORS[..]),
            (BRAND_TOKENS, &BRANDS[..]),
            (NUMBER_TOKENS, &NUMBERS[..]),
        ] {
            let got: Vec<&str> = range.map(token_str).collect();
            assert_eq!(got, words);
        }
        assert_eq!(NUMBER_TOKENS.end as usize, VOCAB_SIZE);
        for role in Role::ALL {
            assert_eq!(Role::from_token(role.token()), Some(role));
        }
        // words are unique, so word_id is a bijection on the word range
        for i in WORDS_START..VOCAB_SIZE as u32 {
            assert_eq!(word_id(token_str(i)), Some(i));
        }
        assert_eq!(word_id("Lamp"), Some(ITEM_TOKENS.start));
        assert_eq!(word_id("CLICK"), None);
        assert_eq!(number_token(17), word_id("17").unwrap());
    }

    #[test]
    fn actions_round_trip() {
        let lamp = word_id("lamp").unwrap();
        for a in [
            Action::Click { role: Role::Link, word: lamp },
            Action::Click { role: Role::Button, word: SEARCH },
            Action::Type { word: lamp },
            Action::Back,
            Action::Answer { word: number_token(3) },
        ] {
            assert_eq!(Action::decode(&a.encode()), Some(a));
        }
        assert_eq!(Action::decode(&[CLICK, Role::Input.token(), lamp, END]), None);
        assert_eq!(Action::decode(&[ANSWER, NONE, PAD, END]), None);
        assert_eq!(Action::decode(&[PAD, PAD, PAD, PAD]), None);
    }

    #[test]
    fn questions_in_english_and_russian() {
        let g = |q: &str| Goal::parse(q).unwrap().to_string();
        assert_eq!(g("What is the price of the lamp?"), "price of lamp");
        assert_eq!(g("Сколько стоит лампа?"), "price of lamp");
        assert_eq!(g("какого цвета велосипед"), "color of bike");
        assert_eq!(g("Кто производитель ноутбука"), "brand of laptop");
        assert_eq!(g("рейтинг клавиатуры"), "rating of keyboard");
        assert_eq!(g("brand of the drone"), "brand of drone");
        assert_eq!(g("who makes the drone?"), "brand of drone");
        assert_eq!(g("How much does the camera cost"), "price of camera");
        assert!(Goal::parse("what is the price").is_err());
        assert!(Goal::parse("tell me about the lamp").is_err());
    }
}
