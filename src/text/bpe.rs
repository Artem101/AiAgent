//! Byte-level BPE tokenizer with byte fallback: any UTF-8 text encodes without `<unk>`.
//!
//! Token ids: `[specials…][256 bytes][merges…]`. Text is first split into pre-tokens — a run
//! of letters, a single digit, a run of other symbols, or whitespace, where one leading space
//! sticks to the following pre-token (`" лампа"`, `" 1"`, `"7"`). Merges never cross
//! pre-token boundaries, so every number is spelled digit by digit (convenient for comparing
//! numbers) and words keep their leading space.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{bail, Error, Result};

/// Header line of the tokenizer file format.
const MAGIC: &str = "cog_engine-bpe 1";

/// A trained byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct Bpe {
    specials: Vec<String>,
    /// Special tokens appended after the merges (ids `vocab − extras.len() ..`): added to a
    /// trained tokenizer without renumbering it, so a model's embeddings stay valid.
    extras: Vec<String>,
    /// Merge `i` joins `merges[i]` into token `first_merge() + i`.
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    /// Bytes of every token (empty for specials).
    pieces: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Digit
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// Splits text into pre-tokens (see the module docs). Concatenating them gives the text back.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let glue = |j: usize| chars[j].1 == ' ' && j + 1 < chars.len() && class(chars[j + 1].1) != Class::Space;
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = chars[i].0;
        let mut j = i;
        if glue(j) {
            j += 1; // a single space sticks to what follows
        }
        let c = class(chars[j].1);
        j += 1;
        match c {
            Class::Digit => {}
            Class::Space => {
                // keep the last space for the next pre-token
                while j < chars.len() && class(chars[j].1) == Class::Space && !glue(j) {
                    j += 1;
                }
            }
            _ => {
                while j < chars.len() && class(chars[j].1) == c {
                    j += 1;
                }
            }
        }
        out.push(&text[start..chars.get(j).map_or(text.len(), |x| x.0)]);
        i = j;
    }
    out
}

impl Bpe {
    /// A tokenizer without merges (pure bytes).
    pub fn bytes_only(specials: &[&str]) -> Self {
        let mut pieces: Vec<Vec<u8>> = vec![Vec::new(); specials.len()];
        pieces.extend((0..=255u8).map(|b| vec![b]));
        Self {
            specials: specials.iter().map(|s| s.to_string()).collect(),
            extras: vec![],
            merges: vec![],
            ranks: HashMap::new(),
            pieces,
        }
    }

    /// Learns merges on `texts` until the vocabulary has `vocab_size` tokens (or no pair occurs
    /// twice). Deterministic: ties go to the smallest pair of ids.
    pub fn train<'a>(texts: impl IntoIterator<Item = &'a str>, specials: &[&str], vocab_size: usize) -> Result<Self> {
        let mut bpe = Self::bytes_only(specials);
        if vocab_size < bpe.pieces.len() {
            bail!("vocab_size {vocab_size} < {} specials + 256 bytes", specials.len())
        }
        let byte0 = specials.len() as u32;
        let mut freq: HashMap<&str, u64> = HashMap::new();
        for t in texts {
            for p in pretokenize(t) {
                *freq.entry(p).or_default() += 1;
            }
        }
        let mut entries: Vec<(&str, u64)> = freq.into_iter().collect();
        entries.sort_unstable(); // deterministic word order
        let mut words: Vec<Vec<u32>> =
            entries.iter().map(|(w, _)| w.bytes().map(|b| byte0 + b as u32).collect()).collect();
        let counts: Vec<i64> = entries.iter().map(|&(_, c)| c as i64).collect();

        let mut pair_count: HashMap<(u32, u32), i64> = HashMap::new();
        let mut pair_words: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (wi, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                *pair_count.entry((p[0], p[1])).or_default() += counts[wi];
                pair_words.entry((p[0], p[1])).or_default().push(wi as u32);
            }
        }
        let mut last_merge = vec![usize::MAX; words.len()];
        while bpe.pieces.len() < vocab_size {
            let best = pair_count.iter().filter(|(_, &c)| c > 0).max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)));
            let Some((&pair, &count)) = best else { break };
            if count < 2 {
                break;
            }
            let m = bpe.merges.len();
            let new_id = bpe.pieces.len() as u32;
            bpe.add_merge(pair)?;
            for wi in pair_words.remove(&pair).unwrap_or_default() {
                let wi = wi as usize;
                if last_merge[wi] == m {
                    continue; // already rewritten for this merge
                }
                last_merge[wi] = m;
                let w = &mut words[wi];
                if !w.windows(2).any(|p| (p[0], p[1]) == pair) {
                    continue; // stale index entry
                }
                for p in w.windows(2) {
                    *pair_count.get_mut(&(p[0], p[1])).expect("counted pair") -= counts[wi];
                }
                let mut merged = Vec::with_capacity(w.len());
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && (w[i], w[i + 1]) == pair {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(w[i]);
                        i += 1;
                    }
                }
                *w = merged;
                for p in w.windows(2) {
                    *pair_count.entry((p[0], p[1])).or_default() += counts[wi];
                    pair_words.entry((p[0], p[1])).or_default().push(wi as u32);
                }
            }
            pair_count.remove(&pair);
        }
        Ok(bpe)
    }

    fn add_merge(&mut self, (a, b): (u32, u32)) -> Result<()> {
        let n = self.pieces.len() as u32;
        if a >= n || b >= n || (a as usize) < self.specials.len() || (b as usize) < self.specials.len() {
            bail!("invalid merge ({a}, {b}) with {n} tokens")
        }
        let mut piece = self.pieces[a as usize].clone();
        piece.extend_from_slice(&self.pieces[b as usize]);
        self.pieces.push(piece);
        self.ranks.insert((a, b), n);
        self.merges.push((a, b));
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn num_specials(&self) -> usize {
        self.specials.len()
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Id of a special token by name.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials
            .iter()
            .position(|s| s == name)
            .map(|i| i as u32)
            .or_else(|| self.extras.iter().position(|s| s == name).map(|i| (self.first_extra() + i) as u32))
    }

    pub fn is_special(&self, id: u32) -> bool {
        let id = id as usize;
        id < self.specials.len() || (id >= self.first_extra() && id < self.pieces.len())
    }

    /// Id of the first appended special token (= the vocabulary size without them).
    pub fn first_extra(&self) -> usize {
        self.pieces.len() - self.extras.len()
    }

    /// Name of a special token.
    pub fn special_name(&self, id: u32) -> Option<&str> {
        let id = id as usize;
        if id < self.specials.len() {
            Some(&self.specials[id])
        } else if id >= self.first_extra() && id < self.pieces.len() {
            Some(&self.extras[id - self.first_extra()])
        } else {
            None
        }
    }

    /// Appends special tokens after the merges (see `extras`).
    pub fn with_extras(mut self, names: &[&str]) -> Self {
        for n in names {
            self.extras.push(n.to_string());
            self.pieces.push(Vec::new());
        }
        self
    }

    /// Bytes of a token (empty for specials and out-of-range ids).
    pub fn piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map_or(&[], |p| p.as_slice())
    }

    /// Encodes text (never produces special tokens).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 2);
        let byte0 = self.specials.len() as u32;
        for p in pretokenize(text) {
            let mut ids: Vec<u32> = p.bytes().map(|b| byte0 + b as u32).collect();
            // Repeatedly apply the earliest-learned merge present in the pre-token.
            while let Some((m, i)) =
                ids.windows(2).enumerate().filter_map(|(i, w)| self.ranks.get(&(w[0], w[1])).map(|&m| (m, i))).min()
            {
                ids[i] = m;
                ids.remove(i + 1);
            }
            out.extend(ids);
        }
        out
    }

    /// Decodes tokens to text; special tokens are skipped, invalid UTF-8 is replaced.
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.piece(i).iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Human-readable token sequence: specials by name, text pieces joined by `·`.
    pub fn describe(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut text: Vec<u8> = Vec::new();
        let flush = |text: &mut Vec<u8>, out: &mut String| {
            if !text.is_empty() {
                out.push_str(&String::from_utf8_lossy(text));
                text.clear();
            }
        };
        let mut prev_text = false;
        for &i in ids {
            if self.is_special(i) {
                flush(&mut text, &mut out);
                if i == 0 {
                    prev_text = false;
                    continue; // padding
                }
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(self.special_name(i).unwrap_or("<?>"));
                prev_text = false;
            } else {
                if prev_text && std::str::from_utf8(&text).is_ok() {
                    flush(&mut text, &mut out);
                    out.push('·');
                } else if !prev_text && !out.is_empty() {
                    out.push(' ');
                }
                text.extend_from_slice(self.piece(i));
                prev_text = true;
            }
        }
        flush(&mut text, &mut out);
        out
    }

    /// Serialises to the text format read by [`Bpe::from_text`].
    pub fn to_text(&self) -> String {
        let mut s = format!("{MAGIC}\nspecials {}\n", self.specials.len());
        for sp in &self.specials {
            s.push_str(sp);
            s.push('\n');
        }
        s.push_str(&format!("merges {}\n", self.merges.len()));
        for (a, b) in &self.merges {
            s.push_str(&format!("{a} {b}\n"));
        }
        if !self.extras.is_empty() {
            s.push_str(&format!("extras {}\n", self.extras.len()));
            for e in &self.extras {
                s.push_str(e);
                s.push('\n');
            }
        }
        s
    }

    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some(MAGIC) {
            bail!("not a tokenizer file (expected '{MAGIC}')")
        }
        let count = |line: Option<&str>, key: &str| -> Result<usize> {
            match line.and_then(|l| l.strip_prefix(key)).map(|n| n.trim().parse()) {
                Some(Ok(n)) => Ok(n),
                _ => bail!("tokenizer file: expected '{key}<n>'"),
            }
        };
        let ns = count(lines.next(), "specials ")?;
        let specials: Vec<&str> = lines.by_ref().take(ns).collect();
        let mut bpe = Self::bytes_only(&specials);
        let nm = count(lines.next(), "merges ")?;
        for _ in 0..nm {
            let line = lines.next().unwrap_or_default();
            let pair: Vec<u32> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            let &[a, b] = pair.as_slice() else { bail!("tokenizer file: bad merge line '{line}'") };
            bpe.add_merge((a, b))?;
        }
        match lines.next() {
            None | Some("") => Ok(bpe),
            line => {
                let ne = count(line, "extras ")?;
                let extras: Vec<&str> = lines.take(ne).collect();
                Ok(bpe.with_extras(&extras))
            }
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        std::fs::write(path, self.to_text()).map_err(Error::wrap)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_text(&std::fs::read_to_string(path).map_err(Error::wrap)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIALS: [&str; 3] = ["<pad>", "<a>", "<b>"];

    #[test]
    fn pretokenizer_keeps_text_and_splits_digits() {
        let text = "Лампа стоит 17 ₽,  а стул — 5?\nДа.";
        let p = pretokenize(text);
        assert_eq!(p.concat(), text);
        assert_eq!(&p[..6], &["Лампа", " стоит", " 1", "7", " ₽,", " "]);
        assert!(p.contains(&" стул") && p.contains(&" 5") && p.contains(&"?"));
        assert!(pretokenize("").is_empty());
    }

    #[test]
    fn train_encode_decode_round_trip() {
        let corpus = ["лампа стоит дёшево", "лампа и стул", "стул стоит 12 рублей", "лампа, лампа, лампа"];
        let bpe = Bpe::train(corpus.iter().copied(), &SPECIALS, 3 + 256 + 20).unwrap();
        assert_eq!(bpe.vocab_size(), 3 + 256 + 20);
        for text in ["лампа стоит 12", "новое слово: жираф 🦒", "", "  пробелы  "] {
            let ids = bpe.encode(text);
            assert!(ids.iter().all(|&i| i >= 3 && (i as usize) < bpe.vocab_size()));
            assert_eq!(bpe.decode(&ids), text);
        }
        // frequent words compress, digits stay single tokens
        assert!(bpe.encode("лампа").len() < "лампа".len());
        assert_eq!(bpe.encode("12").len(), 2);
        // determinism and file round trip
        let again = Bpe::train(corpus.iter().copied(), &SPECIALS, 3 + 256 + 20).unwrap();
        assert_eq!(again.to_text(), bpe.to_text());
        let loaded = Bpe::from_text(&bpe.to_text()).unwrap();
        assert_eq!(loaded.encode("лампа и стул"), bpe.encode("лампа и стул"));
        assert_eq!(loaded.special("<b>"), Some(2));
        assert!(Bpe::from_text("nope").is_err());
        // describe shows specials and piece boundaries
        let mut ids = vec![1];
        ids.extend(bpe.encode("лампа"));
        assert!(bpe.describe(&ids).starts_with("<a> "), "{}", bpe.describe(&ids));
        // specials appended after the merges keep every other id
        let ext = bpe.clone().with_extras(&["<x>", "<y>"]);
        assert_eq!(ext.vocab_size(), bpe.vocab_size() + 2);
        assert_eq!(ext.encode("лампа и стул"), bpe.encode("лампа и стул"));
        let y = ext.special("<y>").unwrap();
        assert_eq!(y as usize, bpe.vocab_size() + 1);
        assert!(ext.is_special(y) && !ext.is_special(y - 2) && ext.is_special(1));
        let back = Bpe::from_text(&ext.to_text()).unwrap();
        assert_eq!((back.vocab_size(), back.special("<x>")), (ext.vocab_size(), ext.special("<x>")));
        assert!(ext.describe(&[y]).contains("<y>"));
    }
}
