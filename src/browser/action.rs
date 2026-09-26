//! Agent actions as fixed-length token sequences: `[verb, role, text…, <end>, <pad>…]`.
//!
//! | action | tokens (text in BPE pieces, see [`crate::text::fragment`]) |
//! |---|---|
//! | click a link / button by its text | `CLICK [link] ␣лампа <end>`, `CLICK [button] ␣Найти <end>` |
//! | type into the text field | `TYPE [input] ␣лампа·␣стул <end>` |
//! | go back in history | `BACK <none> <end>` |
//! | call the calculator | `CALC <none> ␣1·2·+·3·0 <end>` |
//! | finish with an answer | `ANSWER <none> ␣4·2 <end>` |
//! | think aloud (a step written into the scratchpad) | `THINK <none> ␣Цена·␣=·… <end>` |
//! | look a word up in the dictionary | `LOOKUP <none> ␣подснежник <end>` |
//!
//! `THINK` and `LOOKUP` belong to the second observation format ([`super::obs::Layout::V2`]).

use super::Role;
use crate::text::{fragment, Bpe, ANSWER, BACK, CALC, CLICK, END, LOOKUP, NONE, PAD, THINK, TYPE};

/// Tokens per action (`L`); must be divisible by the planning horizon.
pub const ACTION_LEN: usize = 32;
/// Longest text an action can carry, in tokens.
pub const MAX_TEXT: usize = ACTION_LEN - 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Click {
        role: Role,
        text: String,
    },
    Type {
        text: String,
    },
    Back,
    /// Evaluate an arithmetic expression with the calculator tool (see [`crate::tools`]).
    Calc {
        text: String,
    },
    Answer {
        text: String,
    },
    /// A step of reasoning, written into the agent's scratchpad.
    Think {
        text: String,
    },
    /// Look a word up in the dictionary (see [`crate::tools::dictionary`]).
    Lookup {
        text: String,
    },
}

impl Action {
    /// Token form of [`ACTION_LEN`] tokens; `None` if the text does not fit into [`MAX_TEXT`].
    pub fn encode(&self, bpe: &Bpe) -> Option<[u32; ACTION_LEN]> {
        self.encode_len(bpe, ACTION_LEN).map(|v| v.try_into().expect("ACTION_LEN tokens"))
    }

    /// Token form of `len` tokens; `None` if the text does not fit into `len − 3` tokens.
    pub fn encode_len(&self, bpe: &Bpe, len: usize) -> Option<Vec<u32>> {
        let (verb, role, text) = match self {
            Self::Click { role, text } => (CLICK, role.token(), text.as_str()),
            Self::Type { text } => (TYPE, Role::Input.token(), text.as_str()),
            Self::Back => (BACK, NONE, ""),
            Self::Calc { text } => (CALC, NONE, text.as_str()),
            Self::Answer { text } => (ANSWER, NONE, text.as_str()),
            Self::Think { text } => (THINK, NONE, text.as_str()),
            Self::Lookup { text } => (LOOKUP, NONE, text.as_str()),
        };
        let ids = fragment(bpe, text);
        if ids.len() + 3 > len {
            return None;
        }
        let mut out = vec![PAD; len];
        out[0] = verb;
        out[1] = role;
        out[2..2 + ids.len()].copy_from_slice(&ids);
        out[2 + ids.len()] = END;
        Some(out)
    }

    /// Parses model output: the text runs until `<end>` (or the last slot) and may not contain
    /// special tokens; clicks, typing, calculations and answers need a non-empty text.
    pub fn decode(tokens: &[u32], bpe: &Bpe) -> Option<Self> {
        let (&verb, &role) = (tokens.first()?, tokens.get(1)?);
        let body = &tokens[2.min(tokens.len())..];
        let body = &body[..body.iter().position(|&t| t == END).unwrap_or(body.len())];
        if body.iter().any(|&t| bpe.is_special(t)) {
            return None;
        }
        let text = bpe.decode(body).trim().to_string();
        let has_text = !text.trim().is_empty();
        match verb {
            CLICK if has_text => match Role::from_token(role)? {
                r @ (Role::Link | Role::Button) => Some(Self::Click { role: r, text }),
                _ => None,
            },
            TYPE if has_text && role == Role::Input.token() => Some(Self::Type { text }),
            BACK => Some(Self::Back),
            CALC if has_text && role == NONE => Some(Self::Calc { text }),
            ANSWER if has_text => Some(Self::Answer { text }),
            THINK if has_text && role == NONE => Some(Self::Think { text }),
            LOOKUP if has_text && role == NONE => Some(Self::Lookup { text }),
            _ => None,
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Click { role, text } => write!(f, "CLICK {} «{text}»", role.name()),
            Self::Type { text } => write!(f, "TYPE «{text}»"),
            Self::Back => write!(f, "BACK"),
            Self::Calc { text } => write!(f, "CALC «{text}»"),
            Self::Answer { text } => write!(f, "ANSWER «{text}»"),
            Self::Think { text } => write!(f, "THINK «{text}»"),
            Self::Lookup { text } => write!(f, "LOOKUP «{text}»"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::world::{ITEMS, UI_CATALOG, UI_FIND, UI_HOME, UI_NEXT};
    use crate::text;

    #[test]
    fn actions_round_trip_and_every_needed_text_fits() {
        let bpe = text::ru();
        let mut texts: Vec<String> = vec![UI_FIND.into(), UI_CATALOG.into(), UI_HOME.into(), UI_NEXT.into()];
        texts.extend(ITEMS.iter().map(|i| i.nom.to_string()));
        for a in ITEMS {
            for b in ITEMS {
                texts.push(format!("{} {}", a.nom, b.nom)); // comparison queries
            }
        }
        texts.extend((1..=21).map(|n| n.to_string()));
        texts.extend(crate::browser::world::COLORS.iter().map(|c| c.0.to_string()));
        texts.extend(crate::browser::world::BRANDS.iter().map(|b| b.to_string()));
        for t in &texts {
            for a in [
                Action::Click { role: Role::Link, text: t.clone() },
                Action::Type { text: t.clone() },
                Action::Answer { text: t.clone() },
            ] {
                let enc =
                    a.encode(bpe).unwrap_or_else(|| panic!("«{t}» does not fit: {}", bpe.describe(&bpe.encode(t))));
                assert_eq!(Action::decode(&enc, bpe), Some(a));
            }
        }
        for e in ["5+5", "999 * 999", "(12+30)*5", "2.5/0.5", "100 - 250"] {
            let a = Action::Calc { text: e.into() };
            assert_eq!(Action::decode(&a.encode(bpe).unwrap(), bpe), Some(a));
        }
        let replies = crate::browser::goal::Chat::ALL.map(|c| c.reply());
        for r in ["998001", "-150", "3.3333"].into_iter().chain(replies) {
            let a = Action::Answer { text: r.into() };
            assert_eq!(Action::decode(&a.encode(bpe).expect(r), bpe), Some(a));
        }
        assert_eq!(Action::decode(&Action::Back.encode(bpe).unwrap(), bpe), Some(Action::Back));
        for a in [
            Action::Think {
                text: "Было 11, 8 отдал — стало меньше, значит вычитаю.".into()
            },
            Action::Lookup { text: "подснежник".into() },
        ] {
            assert_eq!(Action::decode(&a.encode_len(bpe, 64).unwrap(), bpe), Some(a));
        }
        assert_eq!(Action::decode(&[THINK, Role::Link.token(), 300, END], bpe), None);
        assert_eq!(Action::decode(&[CALC, Role::Link.token(), 300, END], bpe), None);
        assert_eq!(Action::decode(&[CLICK, Role::Input.token(), 300, END], bpe), None);
        assert_eq!(Action::decode(&[ANSWER, NONE, END, PAD], bpe), None);
        assert_eq!(Action::decode(&[TYPE, Role::Input.token(), 300, CLICK, END], bpe), None);
        assert!(Action::Type {
            text: "очень длинный текст, который никак не помещается в одно действие агента, потому что в нём \
                   слишком много слов: даже самый щедрый словарь не уложит его в двадцать девять токенов"
                .into()
        }
        .encode(bpe)
        .is_none());
    }
}
