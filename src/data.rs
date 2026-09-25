//! Tasks used to train and evaluate the engine: synthetic sequence transduction and
//! web browsing ([`Task::Browser`], see [`crate::browser`]).

use candle_core::{bail, Device, Result, Tensor};

use crate::browser;
use crate::kernels::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Output the prompt sorted ascending (a bag-of-tokens → ordered-sequence problem).
    Sort,
    /// Output the prompt reversed (requires positional memory).
    Reverse,
    /// Output the prompt unchanged.
    Copy,
    /// Web browsing: the prompt is an observation of a page plus a goal, the answer is the
    /// next browser action (see [`crate::browser::obs`] and [`crate::browser::expert`]).
    Browser,
}

impl Task {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "sort" => Ok(Self::Sort),
            "reverse" => Ok(Self::Reverse),
            "copy" => Ok(Self::Copy),
            "browser" => Ok(Self::Browser),
            other => bail!("unknown task '{other}' (expected sort | reverse | copy | browser)"),
        }
    }

    /// Name used on the command line.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sort => "sort",
            Self::Reverse => "reverse",
            Self::Copy => "copy",
            Self::Browser => "browser",
        }
    }

    /// `(vocab, prompt length N, answer length L)` for the requested `vocab` / `len`. The
    /// browsing task has a fixed vocabulary and fixed observation / action lengths.
    pub fn dims(&self, vocab: usize, len: usize) -> (usize, usize, usize) {
        match self {
            Self::Browser => (browser::VOCAB_SIZE, browser::OBS_LEN, browser::ACTION_LEN),
            _ => (vocab, len, len),
        }
    }

    /// The correct answer for `prompt` (for [`Task::Browser`]: the expert's next action).
    pub fn apply(&self, prompt: &[u32]) -> Vec<u32> {
        let mut out = prompt.to_vec();
        match self {
            Self::Sort => out.sort_unstable(),
            Self::Reverse => out.reverse(),
            Self::Copy => {}
            Self::Browser => return browser::expert::act_tokens(prompt).to_vec(),
        }
        out
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
}

impl TaskSampler {
    pub fn new(task: Task, vocab: usize, prompt_len: usize, answer_len: usize) -> Result<Self> {
        if task == Task::Browser {
            let want = task.dims(vocab, prompt_len);
            if (vocab, prompt_len, answer_len) != want {
                bail!("task 'browser' needs (vocab, N, L) = {want:?}, got ({vocab}, {prompt_len}, {answer_len})")
            }
        } else if prompt_len != answer_len {
            bail!("task '{}' maps N tokens to N tokens (got N={prompt_len}, L={answer_len})", task.name())
        }
        Ok(Self { task, vocab, prompt_len, answer_len })
    }

    /// One `(prompt, answer)` pair.
    pub fn example(&self, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        if self.task == Task::Browser {
            let (observation, action) = browser::data::example(rng);
            return (observation.to_vec(), action.to_vec());
        }
        let prompt: Vec<u32> = (0..self.prompt_len).map(|_| rng.below(self.vocab) as u32).collect();
        let answer = self.task.apply(&prompt);
        (prompt, answer)
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
        assert_eq!(Task::Sort.apply(&[3, 1, 2]), vec![1, 2, 3]);
        assert_eq!(Task::Reverse.apply(&[3, 1, 2]), vec![2, 1, 3]);
        let s = TaskSampler::new(Task::Sort, 10, 5, 5).unwrap();
        let b = s.batch(&mut Rng::new(0), 4, &Device::Cpu).unwrap();
        assert_eq!(b.prompt.dims(), &[4, 5]);
        assert!(b.answers.iter().all(|a| a.windows(2).all(|w| w[0] <= w[1])));
    }
}
