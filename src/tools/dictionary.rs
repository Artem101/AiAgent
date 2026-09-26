//! The dictionary tool: `LOOKUP «слово»` returns the word's entry — part of speech, grammar
//! (gender, declension, aspect, conjugation, forms) and structure (prefix, root, suffixes,
//! ending) — from `dictionary.jsonl`, built by `scripts/build_school.py` from openrussian.org,
//! A. N. Tikhonov's morpheme dictionary and UD SynTagRus.
//!
//! Look-ups ignore letter case and `ё`/`е`. The data is not in the repository; without it every
//! look-up answers «нет в словаре».

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

/// Answer for a word the dictionary does not have.
pub const NOT_FOUND: &str = "нет в словаре";

/// Word → entry.
#[derive(Debug, Clone, Default)]
pub struct Dictionary {
    entries: HashMap<String, String>,
}

fn key(word: &str) -> String {
    word.trim().trim_matches(|c: char| !c.is_alphanumeric() && c != '-').to_lowercase().replace('ё', "е")
}

impl Dictionary {
    /// Loads a `dictionary.jsonl` (one `{"word": …, "entry": …}` object per line).
    pub fn load<P: AsRef<Path>>(path: P) -> candle_core::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            candle_core::Error::Msg(format!(
                "{}: {e} (build it with scripts/fetch_school.sh and scripts/build_school.py)",
                path.as_ref().display()
            ))
        })?;
        let mut entries = HashMap::new();
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line).map_err(candle_core::Error::wrap)?;
            if let (Some(w), Some(e)) = (v["word"].as_str(), v["entry"].as_str()) {
                entries.insert(key(w), e.to_string());
            }
        }
        Ok(Self { entries })
    }

    /// A dictionary of a few entries (tests).
    pub fn from_entries<'a>(entries: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        Self { entries: entries.into_iter().map(|(w, e)| (key(w), e.to_string())).collect() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry of `word`, or [`NOT_FOUND`].
    pub fn lookup(&self, word: &str) -> String {
        self.entries.get(&key(word)).cloned().unwrap_or_else(|| NOT_FOUND.to_string())
    }
}

static SHARED: OnceLock<Dictionary> = OnceLock::new();

/// The dictionary the agent uses: `data/school/dictionary.jsonl` (or `$COG_DICTIONARY`), empty
/// when the file is missing. Loaded on first use.
pub fn shared() -> &'static Dictionary {
    SHARED.get_or_init(|| {
        let path = std::env::var("COG_DICTIONARY").unwrap_or_else(|_| "data/school/dictionary.jsonl".into());
        Dictionary::load(&path).unwrap_or_default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookups_ignore_case_and_yo() {
        let d = Dictionary::from_entries([("ёлка", "ёлка — существительное, ж. р."), ("стол", "стол — сущ.")]);
        assert_eq!(d.lookup("Елка"), "ёлка — существительное, ж. р.");
        assert_eq!(d.lookup(" СТОЛ. "), "стол — сущ.");
        assert_eq!(d.lookup("жираф"), NOT_FOUND);
    }
}
