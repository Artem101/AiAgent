//! Tasks used to train and evaluate the engine: synthetic sequence transduction, Russian text
//! continuation ([`Task::Text`]) and web browsing ([`Task::Browser`], see [`crate::browser`]).

use candle_core::{bail, Device, Result, Tensor};

use crate::browser;
use crate::kernels::rng::Rng;
use crate::text::{self, Corpus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Output the prompt sorted ascending (a bag-of-tokens → ordered-sequence problem).
    Sort,
    /// Output the prompt reversed (requires positional memory).
    Reverse,
    /// Output the prompt unchanged.
    Copy,
    /// Web browsing: the prompt is a page plus a Russian instruction, the answer is the next
    /// browser action (see [`crate::browser::obs`] and [`crate::browser::expert`]).
    Browser,
    /// Russian text continuation: the prompt ends with a piece of a sentence, the answer is its
    /// next `L` BPE tokens (needs a corpus, see [`crate::text::Corpus`]).
    Text,
}

impl Task {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "sort" => Ok(Self::Sort),
            "reverse" => Ok(Self::Reverse),
            "copy" => Ok(Self::Copy),
            "browser" => Ok(Self::Browser),
            "text" => Ok(Self::Text),
            other => bail!("unknown task '{other}' (expected sort | reverse | copy | browser | text)"),
        }
    }

    /// Name used on the command line.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sort => "sort",
            Self::Reverse => "reverse",
            Self::Copy => "copy",
            Self::Browser => "browser",
            Self::Text => "text",
        }
    }

    /// Whether the task uses the Russian BPE tokenizer ([`crate::text::ru`]).
    pub fn uses_text(&self) -> bool {
        matches!(self, Self::Browser | Self::Text)
    }

    /// `(vocab, prompt length N, answer length L)` for the requested `vocab` / `len`. The text
    /// tasks share the BPE vocabulary and the browser's observation / action lengths, so one
    /// model can learn both.
    pub fn dims(&self, vocab: usize, len: usize) -> (usize, usize, usize) {
        match self {
            Self::Browser | Self::Text => (browser::vocab_size(), browser::OBS_LEN, browser::ACTION_LEN),
            _ => (vocab, len, len),
        }
    }

    /// The correct answer for `prompt`, when it is a function of the prompt tokens alone
    /// (`None` for browsing and text continuation).
    pub fn apply(&self, prompt: &[u32]) -> Option<Vec<u32>> {
        let mut out = prompt.to_vec();
        match self {
            Self::Sort => out.sort_unstable(),
            Self::Reverse => out.reverse(),
            Self::Copy => {}
            Self::Browser | Self::Text => return None,
        }
        Some(out)
    }
}

/// A training batch: token tensors for the graph path plus the raw ids.
pub struct Batch {
    /// `[B, N]` u32.
    pub prompt: Tensor,
    /// `[B, L]` u32.
    pub answer: Tensor,
    pub prompts: Vec<Vec<u32>>,
    pub answers: Vec<Vec<u32>>,
}

/// Draws random prompts and their answers for a [`Task`].
#[derive(Debug, Clone)]
pub struct TaskSampler {
    pub task: Task,
    pub vocab: usize,
    pub prompt_len: usize,
    pub answer_len: usize,
    /// Text corpus for [`Task::Text`] and for mixing into [`Task::Browser`].
    pub corpus: Option<Corpus>,
    /// Share of text-continuation examples in a browsing batch.
    pub text_mix: f64,
}

impl TaskSampler {
    pub fn new(task: Task, vocab: usize, prompt_len: usize, answer_len: usize) -> Result<Self> {
        if task.uses_text() {
            let want = task.dims(vocab, prompt_len);
            if (vocab, prompt_len, answer_len) != want {
                bail!(
                    "task '{}' needs (vocab, N, L) = {want:?}, got ({vocab}, {prompt_len}, {answer_len})",
                    task.name()
                )
            }
        } else if prompt_len != answer_len {
            bail!("task '{}' maps N tokens to N tokens (got N={prompt_len}, L={answer_len})", task.name())
        }
        Ok(Self { task, vocab, prompt_len, answer_len, corpus: None, text_mix: 0.0 })
    }

    /// Adds a text corpus: all examples of [`Task::Text`], a `text_mix` share of the examples of
    /// [`Task::Browser`].
    pub fn with_corpus(mut self, corpus: Corpus, text_mix: f64) -> Self {
        self.text_mix = if self.task == Task::Text { 1.0 } else { text_mix };
        self.corpus = Some(corpus);
        self
    }

    /// Loads the corpus at `path` with the Russian tokenizer (see [`TaskSampler::with_corpus`]).
    pub fn with_corpus_file(self, path: &std::path::Path, text_mix: f64) -> Result<Self> {
        Ok(self.with_corpus(Corpus::load(path, text::ru())?, text_mix))
    }

    /// One `(prompt, answer)` pair.
    pub fn example(&self, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        match (self.task, &self.corpus) {
            (Task::Text | Task::Browser, Some(c)) if rng.uniform() < self.text_mix => {
                c.example(rng, self.prompt_len, self.answer_len)
            }
            (Task::Browser, _) => {
                let (observation, action) = browser::data::example(rng);
                (observation.to_vec(), action.to_vec())
            }
            (Task::Text, _) => panic!("task 'text' needs a corpus (TaskSampler::with_corpus)"),
            _ => {
                let prompt: Vec<u32> = (0..self.prompt_len).map(|_| rng.below(self.vocab) as u32).collect();
                let answer = self.task.apply(&prompt).expect("synthetic task");
                (prompt, answer)
            }
        }
    }

    /// A batch of `size` examples as tensors on `device`.
    pub fn batch(&self, rng: &mut Rng, size: usize, device: &Device) -> Result<Batch> {
        let (prompts, answers): (Vec<_>, Vec<_>) = (0..size).map(|_| self.example(rng)).unzip();
        Ok(Batch {
            prompt: Tensor::from_vec(prompts.concat(), (size, self.prompt_len), device)?,
            answer: Tensor::from_vec(answers.concat(), (size, self.answer_len), device)?,
            prompts,
            answers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks() {
        assert_eq!(Task::Sort.apply(&[3, 1, 2]), Some(vec![1, 2, 3]));
        assert_eq!(Task::Reverse.apply(&[3, 1, 2]), Some(vec![2, 1, 3]));
        assert_eq!(Task::Browser.apply(&[3, 1, 2]), None);
        let s = TaskSampler::new(Task::Sort, 10, 5, 5).unwrap();
        let b = s.batch(&mut Rng::new(0), 4, &Device::Cpu).unwrap();
        assert_eq!(b.prompt.dims(), &[4, 5]);
        assert!(b.answers.iter().all(|a| a.windows(2).all(|w| w[0] <= w[1])));
    }
}
