//! Exact arithmetic for the calculator tool.
//!
//! The grammar is Python's arithmetic subset — `+ - * /`, unary `±`, parentheses, integer and
//! decimal literals — evaluated over exact rationals, so `0.1 + 0.2` is `0.3` and `10/4` is
//! `2.5`. [`super::python::PythonCalc`] runs the same program in a real `python3`
//! (`ast` + `fractions.Fraction`); this module is its in-process mirror, used to label training
//! data and in the simulator, and `tests` checks that both agree on every input, errors
//! included.
//!
//! ```text
//! expr  := term (('+' | '-') term)*
//! term  := unary (('*' | '/') unary)*
//! unary := ('+' | '-') unary | atom
//! atom  := number | '(' expr ')'
//! ```
//!
//! The whole expression is parsed before anything is evaluated (as Python does), then it is
//! evaluated left to right; literals have at most [`MAX_DIGITS`] digits, and every literal and
//! intermediate result must stay within [`LIMIT`]. Results are printed as an integer or a
//! decimal rounded to [`DECIMALS`] places.

use std::fmt;

/// Longest accepted expression, in characters.
pub const MAX_EXPR: usize = 64;
/// Bound on the numerator and denominator of every value (literals and intermediate results).
pub const LIMIT: i128 = 1_000_000_000_000_000;
/// Most digits a literal may have.
pub const MAX_DIGITS: usize = 18;
/// Decimal places of a non-integer result.
pub const DECIMALS: u32 = 4;
/// Characters an expression may contain.
pub const ALPHABET: &str = "0123456789+-*/(). ";

/// Why an expression has no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalcError {
    /// Syntax error, forbidden operation or character, too long.
    Invalid,
    DivisionByZero,
    /// A value outside [`LIMIT`].
    TooLarge,
    /// The Python worker did not answer in time.
    Timeout,
    /// The Python worker is not available or failed.
    Unavailable(String),
}

impl CalcError {
    /// The code both implementations print (`err <code>`).
    pub fn code(&self) -> &str {
        match self {
            Self::Invalid => "invalid",
            Self::DivisionByZero => "zero",
            Self::TooLarge => "large",
            Self::Timeout => "timeout",
            Self::Unavailable(_) => "unavailable",
        }
    }

    pub fn from_code(code: &str) -> Self {
        match code {
            "invalid" => Self::Invalid,
            "zero" => Self::DivisionByZero,
            "large" => Self::TooLarge,
            "timeout" => Self::Timeout,
            other => Self::Unavailable(other.to_string()),
        }
    }
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => write!(f, "неверное выражение"),
            Self::DivisionByZero => write!(f, "деление на ноль"),
            Self::TooLarge => write!(f, "слишком большое число"),
            Self::Timeout => write!(f, "превышено время"),
            Self::Unavailable(why) => write!(f, "калькулятор недоступен: {why}"),
        }
    }
}

impl std::error::Error for CalcError {}

/// An exact rational `num / den` in lowest terms, `den > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ratio {
    pub num: i128,
    pub den: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    (a, b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Ratio {
    /// Reduced `num / den`, checked against [`LIMIT`].
    fn new(num: i128, den: i128) -> Result<Self, CalcError> {
        if den == 0 {
            return Err(CalcError::DivisionByZero);
        }
        let g = gcd(num, den).max(1) * den.signum();
        let r = Self { num: num / g, den: den / g };
        if r.num.abs() > LIMIT || r.den > LIMIT {
            return Err(CalcError::TooLarge);
        }
        Ok(r)
    }

    fn neg(self) -> Self {
        Self { num: -self.num, den: self.den }
    }

    fn apply(self, op: u8, o: Self) -> Result<Self, CalcError> {
        // |num|, den ≤ 10^15, so every product fits into i128 with room to spare.
        match op {
            b'+' => Self::new(self.num * o.den + o.num * self.den, self.den * o.den),
            b'-' => Self::new(self.num * o.den - o.num * self.den, self.den * o.den),
            b'*' => Self::new(self.num * o.num, self.den * o.den),
            _ => Self::new(self.num * o.den, self.den * o.num),
        }
    }
}

impl fmt::Display for Ratio {
    /// Integer, or a decimal rounded half away from zero to [`DECIMALS`] places with trailing
    /// zeros removed (`5`, `2.5`, `3.3333`, `-0.5`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            return write!(f, "{}", self.num);
        }
        let scale = 10i128.pow(DECIMALS);
        let scaled = (2 * self.num.abs() * scale + self.den) / (2 * self.den);
        let (int, frac) = (scaled / scale, scaled % scale);
        let sign = if self.num < 0 && scaled != 0 { "-" } else { "" };
        if frac == 0 {
            return write!(f, "{sign}{int}");
        }
        let frac = format!("{frac:0width$}", width = DECIMALS as usize);
        write!(f, "{sign}{int}.{}", frac.trim_end_matches('0'))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok<'a> {
    Num(&'a str),
    Op(u8),
    Open,
    Close,
}

fn lex(s: &str) -> Result<Vec<Tok<'_>>, CalcError> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b' ' => i += 1,
            b'(' => (out.push(Tok::Open), i += 1).1,
            b')' => (out.push(Tok::Close), i += 1).1,
            c @ (b'+' | b'-' | b'*' | b'/') => (out.push(Tok::Op(c)), i += 1).1,
            b'0'..=b'9' | b'.' => {
                let start = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                let int = &s[start..i];
                if i < b.len() && b[i] == b'.' {
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else if int.len() > 1 && int.starts_with('0') && int.bytes().any(|c| c != b'0') {
                    return Err(CalcError::Invalid); // Python: no leading zeros in integers
                }
                let lit = &s[start..i];
                if !lit.bytes().any(|c| c.is_ascii_digit()) {
                    return Err(CalcError::Invalid);
                }
                out.push(Tok::Num(lit));
            }
            _ => return Err(CalcError::Invalid),
        }
    }
    Ok(out)
}

#[derive(Debug)]
enum Node<'a> {
    Num(&'a str),
    Neg(Box<Node<'a>>),
    Bin(u8, Box<Node<'a>>, Box<Node<'a>>),
}

struct Parser<'a> {
    toks: Vec<Tok<'a>>,
    at: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Tok<'a>> {
        self.toks.get(self.at).copied()
    }

    fn expr(&mut self) -> Result<Node<'a>, CalcError> {
        let mut left = self.term()?;
        while let Some(Tok::Op(op @ (b'+' | b'-'))) = self.peek() {
            self.at += 1;
            left = Node::Bin(op, Box::new(left), Box::new(self.term()?));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Node<'a>, CalcError> {
        let mut left = self.unary()?;
        while let Some(Tok::Op(op @ (b'*' | b'/'))) = self.peek() {
            self.at += 1;
            left = Node::Bin(op, Box::new(left), Box::new(self.unary()?));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Node<'a>, CalcError> {
        match self.peek() {
            Some(Tok::Op(b'+')) => {
                self.at += 1;
                self.unary()
            }
            Some(Tok::Op(b'-')) => {
                self.at += 1;
                Ok(Node::Neg(Box::new(self.unary()?)))
            }
            Some(Tok::Num(lit)) => {
                self.at += 1;
                Ok(Node::Num(lit))
            }
            Some(Tok::Open) => {
                self.at += 1;
                let inner = self.expr()?;
                if self.peek() != Some(Tok::Close) {
                    return Err(CalcError::Invalid);
                }
                self.at += 1;
                Ok(inner)
            }
            _ => Err(CalcError::Invalid),
        }
    }
}

/// Exact value of a decimal literal (`"12"`, `"2.50"`, `".5"`, `"5."`).
fn literal(lit: &str) -> Result<Ratio, CalcError> {
    if lit.bytes().filter(u8::is_ascii_digit).count() > MAX_DIGITS {
        return Err(CalcError::TooLarge);
    }
    let (int, frac) = lit.split_once('.').unwrap_or((lit, ""));
    let num: i128 = format!("{int}{frac}").parse().map_err(|_| CalcError::Invalid)?;
    Ratio::new(num, 10i128.pow(frac.len() as u32))
}

fn eval(node: &Node) -> Result<Ratio, CalcError> {
    match node {
        Node::Num(lit) => literal(lit),
        Node::Neg(x) => Ok(eval(x)?.neg()),
        Node::Bin(op, a, b) => {
            let a = eval(a)?;
            let b = eval(b)?;
            a.apply(*op, b)
        }
    }
}

/// Exact value of `expr`.
pub fn evaluate(expr: &str) -> Result<Ratio, CalcError> {
    let expr = expr.trim();
    if expr.chars().count() > MAX_EXPR || !expr.chars().all(|c| ALPHABET.contains(c)) {
        return Err(CalcError::Invalid);
    }
    let mut p = Parser { toks: lex(expr)?, at: 0 };
    let tree = p.expr()?;
    if p.at != p.toks.len() {
        return Err(CalcError::Invalid);
    }
    eval(&tree)
}

/// `evaluate(expr)` printed as the tool prints it.
pub fn run(expr: &str) -> Result<String, CalcError> {
    evaluate(expr).map(|r| r.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_is_exact_and_python_like() {
        let ok = |e: &str, want: &str| assert_eq!(run(e).as_deref(), Ok(want), "{e}");
        ok("5+5", "10");
        ok("2+3*4", "14");
        ok("(2+3)*4", "20");
        ok("10/4", "2.5");
        ok("10/3", "3.3333");
        ok("2/3", "0.6667");
        ok("-2/3", "-0.6667");
        ok("0.1+0.2", "0.3");
        ok("7-10", "-3");
        ok("2*-3", "-6");
        ok("2--3", "5");
        ok(" 12 * 3 ", "36");
        ok(".5+5.", "5.5");
        ok("00", "0");
        ok("1/100000", "0");
        ok("-1/100000", "0");
        ok("999*999", "998001");
        let err = |e: &str, want: CalcError| assert_eq!(run(e), Err(want), "{e}");
        err("5/0", CalcError::DivisionByZero);
        err("5/(3-3)", CalcError::DivisionByZero);
        err("05", CalcError::Invalid);
        err("2**3", CalcError::Invalid);
        err("2//3", CalcError::Invalid);
        err("5 (2)", CalcError::Invalid);
        err("1/0+", CalcError::Invalid); // parsed before evaluated, like Python
        err("", CalcError::Invalid);
        err(".", CalcError::Invalid);
        err("1.2.3", CalcError::Invalid);
        err("__import__('os')", CalcError::Invalid);
        err("2^3", CalcError::Invalid);
        err("1000000*1000000*10000", CalcError::TooLarge);
        ok("1000000*1000000*1000", "1000000000000000");
        err("12345678901234567890", CalcError::TooLarge);
    }
}
