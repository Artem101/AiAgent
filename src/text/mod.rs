//! Open-vocabulary text: a byte-level BPE tokenizer ([`bpe`]) trained on Russian text, the
//! shared special tokens, and a text corpus for next-chunk prediction (`--task text`).
//!
//! The shipped tokenizer (`models/tokenizer_ru.bpe`, [`ru`]) has 20 special tokens, 256 byte
//! tokens, 7916 merges and two appended specials (`THINK`, `LOOKUP`, ids 8192 and 8193) learned on the «ru20k» dataset (dialogues, example sentences for the
//! 20 000 most frequent words, grammar questions — see `scripts/build_ru20k.py`), the UD Russian
//! corpus and the browsing task's texts — 8194 tokens in total. Thanks to byte fallback any
//! text, in any language, can be encoded. (The first agent used a 1024-token tokenizer trained
//! on UD Russian alone; it and that agent's checkpoint are in the repository history.)
//!
//! Every text fragment — a sentence, an instruction, a page element, the text of an action — is
//! encoded with a leading space ([`fragment`]). A word or a number is then the same tokens
//! wherever it occurs (`лампа` in a question, on a link and in `TYPE «лампа»`), which is what
//! lets the decoder copy it from the context.

pub mod bpe;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use candle_core::{bail, Error, Result};

pub use bpe::Bpe;

use crate::kernels::rng::Rng;

/// Special tokens (ids `0..SPECIALS.len()`), shared by every task that uses [`ru`].
pub const SPECIALS: [&str; 20] = [
    "<pad>", "<goal>", "<end>", "<none>", "<empty>", "<text>", "[head]", "[link]", "[input]", "[button]", "[label]",
    "[value]", "[text]", "CLICK", "TYPE", "BACK", "ANSWER", "CALC", "<user>", "<bot>",
];

pub const PAD: u32 = 0;
/// Separates the page from the instruction in a browser observation.
pub const GOAL: u32 = 1;
/// End of an answer / action.
pub const END: u32 = 2;
pub const NONE: u32 = 3;
/// Value of an empty text field.
pub const EMPTY: u32 = 4;
/// Starts the context of a text-continuation prompt.
pub const TEXT: u32 = 5;
pub const CLICK: u32 = 13;
pub const TYPE: u32 = 14;
pub const BACK: u32 = 15;
pub const ANSWER: u32 = 16;
/// Calculator call (an action) and its result (in the next observation).
pub const CALC: u32 = 17;
/// Earlier turns of a conversation in an observation: what the user said…
pub const USER: u32 = 18;
/// …and what the agent answered. Also the first input of the speech decoder.
pub const BOT: u32 = 19;
/// Specials appended after the merges (ids 8192, 8193): a step of reasoning written into the
/// agent's scratchpad, and a dictionary look-up (see docs/scaling.md). Models trained before
/// them have 8192 tokens.
pub const EXTRAS: [&str; 2] = ["THINK", "LOOKUP"];
pub const THINK: u32 = 8192;
pub const LOOKUP: u32 = 8193;
/// Vocabulary size without [`EXTRAS`].
pub const BASE_VOCAB: usize = 8192;

static RU: OnceLock<Bpe> = OnceLock::new();

/// The shipped Russian tokenizer (`models/tokenizer_ru.bpe`, compiled into the binary).
pub fn ru() -> &'static Bpe {
    RU.get_or_init(|| {
        let bpe = Bpe::from_text(include_str!("../../models/tokenizer_ru.bpe")).expect("valid shipped tokenizer");
        assert_eq!(bpe.num_specials(), SPECIALS.len(), "tokenizer specials out of sync with text::SPECIALS");
        assert_eq!((bpe.special("THINK"), bpe.special("LOOKUP")), (Some(THINK), Some(LOOKUP)), "appended specials");
        bpe
    })
}

/// Tokens of a text fragment: trimmed and encoded with one leading space (empty text → no
/// tokens).
pub fn fragment(bpe: &Bpe, text: &str) -> Vec<u32> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    bpe.encode(&format!(" {text}"))
}

/// Sentences of a text file (one per line), tokenized.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub sentences: Arc<Vec<Vec<u32>>>,
}

impl Corpus {
    pub fn load<P: AsRef<Path>>(path: P, bpe: &Bpe) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Msg(format!("{}: {e} (download it with scripts/fetch_ru_corpus.sh)", path.display()))
        })?;
        let sentences: Vec<Vec<u32>> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| fragment(bpe, l))
            .filter(|s| s.len() >= 2)
            .collect();
        if sentences.is_empty() {
            bail!("{}: no sentences", path.display())
        }
        Ok(Self { sentences: Arc::new(sentences) })
    }

    pub fn tokens(&self) -> usize {
        self.sentences.iter().map(Vec::len).sum()
    }

    /// A next-chunk example: the prompt ends with `<text>` + up to `n − 1` tokens of a sentence
    /// (left-padded to `n`), the answer is the next `l` tokens, then `<end>` and padding.
    pub fn example(&self, rng: &mut Rng, n: usize, l: usize) -> (Vec<u32>, Vec<u32>) {
        let s = &self.sentences[rng.below(self.sentences.len())];
        let cut = 1 + rng.below(s.len() - 1); // ≥ 1 token of context, ≥ 1 to predict
        let ctx = &s[cut.saturating_sub(n - 1)..cut];
        let mut prompt = vec![PAD; n - 1 - ctx.len()];
        prompt.push(TEXT);
        prompt.extend_from_slice(ctx);
        let mut answer: Vec<u32> = s[cut..].iter().copied().take(l).collect();
        if answer.len() < l {
            answer.push(END);
        }
        answer.resize(l, PAD);
        (prompt, answer)
    }
}

/// Next-chunk accuracy on held-out text, next to frequency baselines.
#[derive(Debug, Clone, Copy, Default)]
pub struct LmReport {
    pub samples: usize,
    /// The first predicted token is right.
    pub first: f32,
    /// Right tokens over all positions up to and including `<end>`.
    pub tokens: f32,
    /// Baseline: always the most frequent token of the training corpus (first position).
    pub unigram_first: f32,
    /// Baseline: the most frequent successor of the last context token (first position).
    pub bigram_first: f32,
}

impl std::fmt::Display for LmReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "text n={} | next token {:.1}% (baselines: unigram {:.1}%, bigram {:.1}%) | all {} tokens {:.1}%",
            self.samples,
            100.0 * self.first,
            100.0 * self.unigram_first,
            100.0 * self.bigram_first,
            crate::browser::ACTION_LEN,
            100.0 * self.tokens
        )
    }
}

/// Evaluates `engine` on `n` continuation examples of `valid`; baselines come from `train`.
pub fn evaluate_lm(
    engine: &mut crate::pipeline::CognitiveEngine,
    train: &Corpus,
    valid: &Corpus,
    n: usize,
    seed: u64,
) -> Result<LmReport> {
    use std::collections::HashMap;
    let vocab = engine.config().vocab_size;
    let (np, l) = (engine.config().max_prompt_len, engine.config().answer_len());
    let mut uni = vec![0u64; vocab];
    let mut bi: HashMap<u32, HashMap<u32, u64>> = HashMap::new();
    for s in train.sentences.iter() {
        for w in s.windows(2) {
            uni[w[1] as usize] += 1;
            *bi.entry(w[0]).or_default().entry(w[1]).or_default() += 1;
        }
    }
    let top = |m: &HashMap<u32, u64>| m.iter().max_by_key(|(t, c)| (**c, std::cmp::Reverse(**t))).map(|(t, _)| *t);
    let unigram = (0..vocab).max_by_key(|&t| uni[t]).unwrap_or(0) as u32;
    let mut rng = Rng::stream(seed, 0x7E47, 0);
    let mut r = LmReport { samples: n, ..Default::default() };
    let (mut right, mut total) = (0usize, 0usize);
    for i in 0..n {
        let (prompt, answer) = valid.example(&mut rng, np, l);
        let (out, _) = engine.generate(&prompt, seed.wrapping_add(i as u64))?;
        r.first += (out[0] == answer[0]) as u8 as f32;
        r.unigram_first += (unigram == answer[0]) as u8 as f32;
        let last = *prompt.last().expect("non-empty prompt");
        r.bigram_first += (bi.get(&last).and_then(top) == Some(answer[0])) as u8 as f32;
        for (o, a) in out.iter().zip(&answer) {
            if *a == PAD {
                break;
            }
            right += (o == a) as usize;
            total += 1;
        }
    }
    let k = n.max(1) as f32;
    r.first /= k;
    r.unigram_first /= k;
    r.bigram_first /= k;
    r.tokens = right as f32 / total.max(1) as f32;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_tokenizer_round_trips_russian() {
        let bpe = ru();
        assert_eq!(bpe.vocab_size(), BASE_VOCAB + EXTRAS.len());
        for (i, s) in SPECIALS.iter().enumerate() {
            assert_eq!(bpe.special(s), Some(i as u32));
        }
        for text in ["Сколько стоит лампа?", "Что дешевле: клавиатура или мышь?", "Ёжик 🦔 и English 42"]
        {
            assert_eq!(bpe.decode(&bpe.encode(text)), text);
        }
        // a word is the same token in a question and on a page
        let q = fragment(bpe, "Что дешевле: лампа или стул?");
        assert!(q.contains(&fragment(bpe, "лампа")[0]), "{}", bpe.describe(&q));
        // Russian compresses well below one token per character
        let t = "Глухой стук в окно заставил Петровича испуганно обернуться.";
        let ids = fragment(bpe, t);
        assert!(ids.len() * 3 < t.chars().count() * 2, "{}", bpe.describe(&ids));
    }

    #[test]
    fn text_examples_have_fixed_shapes() {
        let bpe = ru();
        let path = std::env::temp_dir().join(format!("cog_corpus_{}.txt", std::process::id()));
        std::fs::write(&path, "Москва — столица России.\nКороткое\n\nЕщё одно предложение подлиннее, чем первое.\n")
            .unwrap();
        let corpus = Corpus::load(&path, bpe).unwrap();
        std::fs::remove_file(&path).ok();
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let (p, a) = corpus.example(&mut rng, 16, 8);
            assert_eq!((p.len(), a.len()), (16, 8));
            let t = p.iter().position(|&x| x == TEXT).unwrap();
            assert!(p[..t].iter().all(|&x| x == PAD) && t < 15);
            assert!(!bpe.is_special(a[0]), "at least one real token to predict");
        }
    }
}
