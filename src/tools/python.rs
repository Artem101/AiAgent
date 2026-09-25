//! The calculator executed by a real Python interpreter.
//!
//! One `python3 -I -S` worker process lives as long as [`PythonCalc`]; expressions go to its
//! stdin one per line, answers come back one per line (`ok <value>` / `err <code>`). The
//! worker never calls `eval`: it parses the expression with `ast`, rejects every node except
//! numeric literals, unary `±` and binary `+ - * /`, and evaluates the tree with
//! `fractions.Fraction` — the exact program [`super::calc`] mirrors in Rust. Isolated mode (no
//! environment variables, no user site), CPU and memory limits, and a per-call timeout (the
//! worker is killed and restarted) keep it a calculator and nothing else.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use super::calc::{CalcError, ALPHABET, LIMIT, MAX_DIGITS, MAX_EXPR};
use super::Calculator;

/// The worker program (`{…}` constants are filled from [`super::calc`]).
const WORKER: &str = r#"
import ast, sys
from fractions import Fraction
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
except Exception:
    pass

LIMIT, MAX_EXPR, MAX_DIGITS, DECIMALS = {LIMIT}, {MAX_EXPR}, {MAX_DIGITS}, {DECIMALS}
ALPHABET = set("{ALPHABET}")
ALLOWED = (ast.Expression, ast.Constant, ast.UnaryOp, ast.UAdd, ast.USub,
           ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div)

class CalcError(Exception):
    pass

def check(x):
    if abs(x.numerator) > LIMIT or x.denominator > LIMIT:
        raise CalcError("large")
    return x

def value(node, src):
    if isinstance(node, ast.Expression):
        return value(node.body, src)
    if isinstance(node, ast.Constant):
        lit = ast.get_source_segment(src, node)
        if sum(c.isdigit() for c in lit) > MAX_DIGITS:
            raise CalcError("large")
        return check(Fraction(lit))
    if isinstance(node, ast.UnaryOp):
        v = value(node.operand, src)
        return -v if isinstance(node.op, ast.USub) else v
    a, b = value(node.left, src), value(node.right, src)
    if isinstance(node.op, ast.Add):
        return check(a + b)
    if isinstance(node.op, ast.Sub):
        return check(a - b)
    if isinstance(node.op, ast.Mult):
        return check(a * b)
    if b == 0:
        raise CalcError("zero")
    return check(a / b)

def calc(src):
    src = src.strip()
    if len(src) > MAX_EXPR or not set(src) <= ALPHABET:
        raise CalcError("invalid")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise CalcError("invalid")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise CalcError("invalid")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise CalcError("invalid")
    return value(tree, src)

def show(x):
    if x.denominator == 1:
        return str(x.numerator)
    scale = 10 ** DECIMALS
    scaled = (2 * abs(x.numerator) * scale + x.denominator) // (2 * x.denominator)
    whole, frac = divmod(scaled, scale)
    sign = "-" if x < 0 and scaled != 0 else ""
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}." + f"{frac:0{DECIMALS}d}".rstrip("0")

for line in sys.stdin:
    try:
        out = "ok " + show(calc(line.rstrip("\n")))
    except CalcError as e:
        out = "err " + str(e)
    except Exception:
        out = "err invalid"
    sys.stdout.write(out + "\n")
    sys.stdout.flush()
"#;

/// A running Python worker.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Calculator backed by `python3` (see the module docs).
pub struct PythonCalc {
    python: PathBuf,
    timeout: Duration,
    worker: Option<Worker>,
    /// Calls answered by Python (for reporting).
    pub calls: usize,
}

impl PythonCalc {
    /// `$COG_PYTHON`, else `python3` on `PATH`.
    pub fn find_python() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("COG_PYTHON").map(PathBuf::from) {
            return Some(p);
        }
        std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join("python3")).find(|p| p.is_file())
    }

    /// Starts a worker (fails when Python is missing or does not start).
    pub fn start() -> Result<Self, CalcError> {
        let python = Self::find_python().ok_or_else(|| CalcError::Unavailable("python3 not found".into()))?;
        let mut calc = Self { python, timeout: Duration::from_secs(2), worker: None, calls: 0 };
        calc.eval("1+1")?; // spawns the worker and checks that it answers
        calc.calls = 0;
        Ok(calc)
    }

    /// The interpreter in use.
    pub fn python(&self) -> &std::path::Path {
        &self.python
    }

    fn spawn(&self) -> Result<Worker, CalcError> {
        let program = WORKER
            .replace("{LIMIT}", &LIMIT.to_string())
            .replace("{MAX_EXPR}", &MAX_EXPR.to_string())
            .replace("{MAX_DIGITS}", &MAX_DIGITS.to_string())
            .replace("{DECIMALS}", &super::calc::DECIMALS.to_string())
            .replace("{ALPHABET}", ALPHABET);
        let mut child = Command::new(&self.python)
            .args(["-I", "-S", "-u", "-c", &program])
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| CalcError::Unavailable(format!("{}: {e}", self.python.display())))?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = child.stdout.take().expect("stdout is piped");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(std::io::Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Worker { child, stdin, lines })
    }
}

impl Calculator for PythonCalc {
    fn name(&self) -> &str {
        "python"
    }

    fn eval(&mut self, expr: &str) -> Result<String, CalcError> {
        // One expression per line: anything that could break the protocol is not arithmetic.
        if expr.contains(['\n', '\r']) || expr.chars().count() > MAX_EXPR {
            return Err(CalcError::Invalid);
        }
        if self.worker.is_none() {
            self.worker = Some(self.spawn()?);
        }
        let w = self.worker.as_mut().expect("worker started");
        let sent = writeln!(w.stdin, "{expr}").and_then(|_| w.stdin.flush());
        let reply = match sent {
            Ok(()) => w.lines.recv_timeout(self.timeout).ok(),
            Err(_) => None,
        };
        let Some(reply) = reply else {
            // Hung or dead: kill it; the next call starts a fresh worker.
            self.worker = None;
            return Err(CalcError::Timeout);
        };
        self.calls += 1;
        match reply.split_once(' ') {
            Some(("ok", v)) => Ok(v.to_string()),
            Some(("err", code)) => Err(CalcError::from_code(code)),
            _ => Err(CalcError::Unavailable(format!("unexpected reply «{reply}»"))),
        }
    }
}
