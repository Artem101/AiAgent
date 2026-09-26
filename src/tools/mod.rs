//! Tools the agent can call.
//!
//! * [`calc`] — exact arithmetic (`+ - * /`, parentheses, decimals) with Python semantics;
//! * [`python`] — the same calculator executed by a sandboxed `python3` worker;
//! * [`dictionary`] — the morphological dictionary of `LOOKUP «слово»`.
//!
//! The browsing agent calls the calculator with the action `CALC «12+30»`; the answer comes
//! back in its next observation (`CALC 1·2·+·3·0 = 4·2`, see [`crate::browser::obs`]).

pub mod calc;
pub mod dictionary;
pub mod python;

pub use calc::CalcError;
pub use python::PythonCalc;

/// Something that evaluates arithmetic expressions.
pub trait Calculator {
    /// Short name for logs (`python`, `rust`).
    fn name(&self) -> &str;
    /// The printed value of `expr` (`"42"`, `"2.5"`) or why it has none.
    fn eval(&mut self, expr: &str) -> Result<String, CalcError>;
}

/// The in-process mirror of the Python calculator ([`calc::run`]): deterministic and free,
/// used to label training data, in the simulator and when Python is not installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct RustCalc;

impl Calculator for RustCalc {
    fn name(&self) -> &str {
        "rust"
    }

    fn eval(&mut self, expr: &str) -> Result<String, CalcError> {
        calc::run(expr)
    }
}

/// Python when available, the Rust mirror otherwise.
pub fn default_calculator() -> Box<dyn Calculator> {
    match PythonCalc::start() {
        Ok(p) => Box::new(p),
        Err(_) => Box::new(RustCalc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::rng::Rng;

    /// Random expressions, well-formed and not: digits, operators, parentheses, dots, spaces.
    fn random_expr(rng: &mut Rng) -> String {
        const PIECES: [&str; 17] =
            ["1", "2", "5", "7", "0", "12", "30", "99", "+", "-", "*", "/", "(", ")", ".", " ", "00"];
        if rng.uniform() < 0.6 {
            // mostly well-formed: a op b [op c], sometimes parenthesised or decimal
            let num = |rng: &mut Rng| match rng.below(10) {
                0 => format!("{}.{}", rng.below(100), rng.below(100)),
                1 => format!("-{}", rng.below(1000)),
                2 => "0".to_string(),
                _ => rng.below(1000).to_string(),
            };
            let op = |rng: &mut Rng| ["+", "-", "*", "/"][rng.below(4)];
            let mut e = format!("{}{}{}", num(rng), op(rng), num(rng));
            if rng.uniform() < 0.5 {
                e = if rng.uniform() < 0.5 {
                    format!("({e}){}{}", op(rng), num(rng))
                } else {
                    format!("{e}{}{}", op(rng), num(rng))
                };
            }
            e
        } else {
            (0..1 + rng.below(9)).map(|_| PIECES[rng.below(PIECES.len())]).collect()
        }
    }

    #[test]
    fn python_worker_matches_the_rust_mirror() {
        let Ok(mut py) = PythonCalc::start() else {
            eprintln!("python3 not available: skipping the parity test");
            return;
        };
        let mut rng = Rng::new(11);
        let mut exprs: Vec<String> = (0..3000).map(|_| random_expr(&mut rng)).collect();
        exprs.extend(
            ["5+5", "10/3", "2**10", "__import__('os').system('true')", "1/0+", "05", "999999999*999999999", "7/0"]
                .map(String::from),
        );
        for e in &exprs {
            assert_eq!(py.eval(e), RustCalc.eval(e), "«{e}»");
        }
        assert_eq!(py.eval("2+2\nimport os"), Err(CalcError::Invalid));
        assert_eq!(py.eval("6*7").as_deref(), Ok("42")); // still alive after rejections
        assert!(py.calls >= exprs.len());
    }
}
