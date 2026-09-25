//! Synthetic sequence-transduction tasks used to train and evaluate the engine.

use candle_core::{bail, Device, Result, Tensor};

use crate::kernels::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Output the prompt sorted ascending (a bag-of-tokens → ordered-sequence problem).
    Sort,
    /// Output the prompt reversed (requires positional memory).
    Reverse,
    /// Output the prompt unchanged.
    Copy,
}

impl Task {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "sort" => Ok(Self::Sort),
            "reverse" => Ok(Self::Reverse),
            "copy" => Ok(Self::Copy),
            other => bail!("unknown task '{other}' (expected sort | reverse | copy)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Sort => "sort",
            Self::Reverse => "reverse",
            Self::Copy => "copy",
        }
    }

    pub fn apply(&self, prompt: &[u32]) -> Vec<u32> {
        let mut out = prompt.to_vec();
        match self {
            Self::Sort => out.sort_unstable(),
            Self::Reverse => out.reverse(),
            Self::Copy => {}
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

#[derive(Debug, Clone)]
pub struct TaskSampler {
    pub task: Task,
    pub vocab: usize,
    pub prompt_len: usize,
    pub answer_len: usize,
}

impl TaskSampler {
    pub fn new(task: Task, vocab: usize, prompt_len: usize, answer_len: usize) -> Result<Self> {
        if prompt_len != answer_len {
            bail!("task '{}' maps N tokens to N tokens (got N={prompt_len}, L={answer_len})", task.name())
        }
        Ok(Self { task, vocab, prompt_len, answer_len })
    }

    pub fn example(&self, rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
        let prompt: Vec<u32> = (0..self.prompt_len).map(|_| rng.below(self.vocab) as u32).collect();
        let answer = self.task.apply(&prompt);
        (prompt, answer)
    }

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
