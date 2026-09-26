//! What the agent is asked: task specs and their wordings in Russian.
//!
//! Six task families:
//!
//! | family | example | answer |
//! |---|---|---|
//! | lookup — find an attribute | «Сколько стоит лампа?», «Кто производитель дрона?» | «Сейчас лампа стоит 12 ₽.», «Бренд — Vega.» |
//! | compare — read and compare two numbers | «Что дешевле: лампа или стул?» | «Дешевле стул.» |
//! | filter — scan the catalogue for a condition | «Найди товар дешевле 5 ₽.», «Нужен красный товар.» | «Например, зонт.» (any matching product) |
//! | calc — arithmetic with the calculator tool | «Сколько будет 5+5?», «Умножь 12 на 3.» | «5+5 = 10», «12 * 3 = 36» |
//! | total — look prices up, then calculate | «Сколько стоят вместе лампа и стул?», «На сколько лампа дороже стула?» | «Вместе 19 ₽.», «Дороже на 5 ₽.» |
//! | chat — a few conversational phrases | «Привет!», «Спасибо», «Что ты умеешь?» | «Привет! Чем помочь?» |
//!
//! Every family has several templates with case forms (именительный, родительный,
//! винительный). Templates marked *held out* are never used for training: evaluating on them
//! measures whether the model understands new phrasings rather than memorised ones. The model
//! only ever sees [`Goal::text`]; the [`Spec`] is used by the teacher and to check answers.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::world::{Attr, World, BRANDS, COLORS, ITEMS, MAX_PRICE};
use crate::kernels::rng::Rng;
use crate::tools::calc;

/// A condition for the catalogue filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cond {
    PriceBelow(u32),
    PriceAbove(u32),
    /// Index into [`COLORS`].
    Color(usize),
}

impl Cond {
    pub fn holds(&self, world: &World, item: usize) -> bool {
        match *self {
            Self::PriceBelow(x) => world.price[item] < x,
            Self::PriceAbove(x) => world.price[item] > x,
            Self::Color(c) => world.color[item] == c,
        }
    }
}

/// An arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    pub const ALL: [Op; 4] = [Self::Add, Self::Sub, Self::Mul, Self::Div];

    pub fn symbol(self) -> char {
        match self {
            Self::Add => '+',
            Self::Sub => '-',
            Self::Mul => '*',
            Self::Div => '/',
        }
    }

    /// Spoken form: «5 плюс 3», «5 умножить на 3».
    pub fn word(self) -> &'static str {
        match self {
            Self::Add => "плюс",
            Self::Sub => "минус",
            Self::Mul => "умножить на",
            Self::Div => "разделить на",
        }
    }

    fn from_symbol(c: char) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.symbol() == c)
    }
}

/// An arithmetic expression over non-negative integers: `a op b`, `a op b op c` or
/// `(a op b) op c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Expr {
    pub nums: [u32; 3],
    pub ops: [Op; 2],
    /// Three operands instead of two.
    pub three: bool,
    /// `(a op b) op c`.
    pub parens: bool,
}

impl Expr {
    pub fn binary(a: u32, op: Op, b: u32) -> Self {
        Self { nums: [a, b, 0], ops: [op, Op::Add], three: false, parens: false }
    }

    /// `12+30` or, `spaced`, `12 + 30`.
    pub fn text(&self, spaced: bool) -> String {
        let [a, b, c] = self.nums;
        let op = |o: Op| if spaced { format!(" {} ", o.symbol()) } else { o.symbol().to_string() };
        let ab = format!("{a}{}{b}", op(self.ops[0]));
        match (self.three, self.parens) {
            (false, _) => ab,
            (true, false) => format!("{ab}{}{c}", op(self.ops[1])),
            (true, true) => format!("({ab}){}{c}", op(self.ops[1])),
        }
    }

    /// Printed value, as the calculator prints it.
    pub fn value(&self) -> Result<String, calc::CalcError> {
        calc::run(&self.text(false))
    }

    /// Parses `a op b`, `a op b op c` or `(a op b) op c` (integers, spaces allowed).
    pub fn parse(text: &str) -> Option<Self> {
        let t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let (parens, t) = match t.strip_prefix('(') {
            Some(rest) => (true, rest.replacen(')', "", 1)),
            None => (false, t),
        };
        let mut nums = Vec::new();
        let mut ops = Vec::new();
        let mut cur = String::new();
        for c in t.chars() {
            if c.is_ascii_digit() {
                cur.push(c);
            } else {
                ops.push(Op::from_symbol(c)?);
                nums.push(std::mem::take(&mut cur).parse().ok()?);
            }
        }
        nums.push(cur.parse().ok()?);
        match (nums.len(), parens) {
            (2, false) => Some(Self::binary(nums[0], ops[0], nums[1])),
            (3, _) => Some(Self { nums: [nums[0], nums[1], nums[2]], ops: [ops[0], ops[1]], three: true, parens }),
            _ => None,
        }
    }
}

/// How an arithmetic question is written: `12+30`, `12 + 30`, or in words (`12 плюс 30`,
/// «Сложи 12 и 30»).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalcForm {
    Compact,
    Spaced,
    Words,
}

/// What to calculate from two prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TotalKind {
    /// `price a + price b`.
    Sum,
    /// How much more `a` costs than `b` (`price a − price b`, asked when positive).
    Pricier,
    /// How much less `a` costs than `b` (`price b − price a`, asked when positive).
    Cheaper,
}

/// A conversational phrase. The teacher answers with one of several replies; any reply with
/// one of the phrase's keywords counts (the model also learns free conversation from dialogues,
/// see [`crate::dialog`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Chat {
    Hello,
    HowAreYou,
    Thanks,
    Bye,
    Who,
    Skills,
}

impl Chat {
    pub const ALL: [Chat; 6] = [Self::Hello, Self::HowAreYou, Self::Thanks, Self::Bye, Self::Who, Self::Skills];

    /// The teacher's replies (the first one is canonical).
    pub fn replies(self) -> &'static [&'static str] {
        match self {
            Self::Hello => {
                &["Привет! Чем помочь?", "Здравствуйте! Чем могу помочь?", "Привет! Что найти или посчитать?"]
            }
            Self::HowAreYou => &["Хорошо, спасибо! А у вас?", "Отлично! Чем помочь?", "Всё хорошо, спасибо."],
            Self::Thanks => &["Пожалуйста!", "Не за что!", "Рад помочь!"],
            Self::Bye => &["До встречи!", "Пока!", "До свидания!"],
            Self::Who => &["Я ваш помощник.", "Я помощник: ищу в интернете, считаю и отвечаю на вопросы."],
            Self::Skills => &[
                "Ищу товары в магазине, считаю и разговариваю.",
                "Умею искать в браузере, считать на калькуляторе и отвечать на вопросы.",
            ],
        }
    }

    /// The canonical reply.
    pub fn reply(self) -> &'static str {
        self.replies()[0]
    }

    /// Word stems one of which a fitting reply contains.
    fn keywords(self) -> &'static [&'static str] {
        match self {
            Self::Hello => &["привет", "здравств", "добр"],
            Self::HowAreYou => &["хорош", "отличн", "нормальн", "неплох", "прекрасн"],
            Self::Thanks => &["пожалуйста", "не за что", "рад"],
            Self::Bye => &["пока", "встреч", "свидан", "всего"],
            Self::Who => &["помощник", "ассистент", "агент"],
            Self::Skills => &["ищу", "искать", "найти", "считаю", "считать", "посчитать"],
        }
    }

    /// Whether `reply` fits the phrase.
    pub fn accepts(self, reply: &str) -> bool {
        let r = reply.to_lowercase().replace('ё', "е");
        self.keywords().iter().any(|k| r.contains(k))
    }
}

/// A task in structured form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spec {
    Lookup {
        item: usize,
        attr: Attr,
    },
    /// Which of `a`, `b` has the larger (`most`) or smaller price / rating.
    Compare {
        a: usize,
        b: usize,
        attr: Attr,
        most: bool,
    },
    Filter(Cond),
    /// Evaluate `expr` with the calculator.
    Calc {
        expr: Expr,
        form: CalcForm,
    },
    /// Look up the prices of `a` and `b`, then calculate.
    Total {
        a: usize,
        b: usize,
        kind: TotalKind,
    },
    Chat(Chat),
}

/// Task family, for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Lookup,
    Compare,
    Filter,
    Calc,
    Total,
    Chat,
}

impl Family {
    pub const ALL: [Family; 6] = [Self::Lookup, Self::Compare, Self::Filter, Self::Calc, Self::Total, Self::Chat];

    pub fn name(self) -> &'static str {
        match self {
            Self::Lookup => "lookup",
            Self::Compare => "compare",
            Self::Filter => "filter",
            Self::Calc => "calc",
            Self::Total => "total",
            Self::Chat => "chat",
        }
    }
}

impl Spec {
    pub fn family(&self) -> Family {
        match self {
            Self::Lookup { .. } => Family::Lookup,
            Self::Compare { .. } => Family::Compare,
            Self::Filter(_) => Family::Filter,
            Self::Calc { .. } => Family::Calc,
            Self::Total { .. } => Family::Total,
            Self::Chat(_) => Family::Chat,
        }
    }

    /// The expression of a total for prices `pa`, `pb`, as the teacher sends it to the
    /// calculator (spaced, so every number reads as on the page).
    pub fn total_expr(kind: TotalKind, pa: u32, pb: u32) -> String {
        match kind {
            TotalKind::Sum => format!("{pa} + {pb}"),
            TotalKind::Pricier => format!("{pa} - {pb}"),
            TotalKind::Cheaper => format!("{pb} - {pa}"),
        }
    }

    /// The expression the teacher sends to the calculator (`None` for tasks without one).
    pub fn calc_expr(&self, world: &World) -> Option<String> {
        match *self {
            Self::Calc { expr, form } => Some(expr.text(form != CalcForm::Compact)),
            Self::Total { a, b, kind } => Some(Self::total_expr(kind, world.price[a], world.price[b])),
            _ => None,
        }
    }

    /// Winner of a comparison.
    fn winner(world: &World, a: usize, b: usize, attr: Attr, most: bool) -> usize {
        let v = |i: usize| if attr == Attr::Price { world.price[i] } else { world.rating[i] };
        if (v(a) > v(b)) == most {
            a
        } else {
            b
        }
    }

    /// Whether `answer` is correct in `world`. The answer may be a sentence («Сейчас лампа стоит 12 ₽.»):
    /// the checker extracts what it states — its numbers, the products and values it names — and
    /// accepts it when that is exactly the right value (any matching product for a filter).
    pub fn accepts(&self, world: &World, answer: &str) -> bool {
        let answer = answer.trim();
        let words = words_of(answer);
        match *self {
            Self::Lookup { item, attr } => match attr {
                Attr::Price | Attr::Rating => numbers(answer) == [world.answer(item, attr)],
                Attr::Color => {
                    let named: Vec<usize> = (0..COLORS.len()).filter(|&c| mentions(&words, &[COLORS[c].0])).collect();
                    named == [world.color[item]]
                }
                Attr::Brand => {
                    let named: Vec<usize> = (0..BRANDS.len()).filter(|&b| mentions(&words, &[BRANDS[b]])).collect();
                    named == [world.brand[item]]
                }
            },
            // the product named first is the one the answer is about («Дешевле лампа.», «Лампа дешевле стула.»)
            Self::Compare { a, b, attr, most } => first_item(&words) == Some(Self::winner(world, a, b, attr, most)),
            Self::Filter(cond) => first_item(&words).is_some_and(|i| cond.holds(world, i)),
            Self::Calc { .. } | Self::Total { .. } => {
                // the last number stated; equal as printed: `10` = `10.0`, a rounded `3.3333` counts for 10/3
                let want = calc::run(&self.calc_expr(world).expect("has an expression"));
                numbers(answer).last().is_some_and(|n| want.is_ok() && calc::run(n) == want)
            }
            Self::Chat(c) => c.accepts(answer),
        }
    }

    /// The teacher's final answer in words (for a filter: the first match in catalogue order;
    /// for a phrase: its canonical reply).
    pub fn answer_sentence(&self, world: &World) -> String {
        match *self {
            Self::Lookup { item, attr } => say::lookup(item, attr, &world.display(item, attr)),
            Self::Compare { a, b, attr, most } => say::compare(Self::winner(world, a, b, attr, most), attr, most),
            Self::Filter(cond) => world
                .catalog
                .iter()
                .find(|&&i| cond.holds(world, i))
                .map_or_else(|| "Не нашёл такого товара.".to_string(), |&i| say::filter(i)),
            Self::Calc { .. } => say::calc(&self.calc_expr(world).expect("has an expression"), &self.expected(world)),
            Self::Total { kind, .. } => say::total(kind, &self.expected(world)),
            Self::Chat(c) => c.reply().to_string(),
        }
    }

    /// The canonical answer (for a filter: the first match in catalogue order).
    pub fn expected(&self, world: &World) -> String {
        match *self {
            Self::Lookup { item, attr } => world.answer(item, attr),
            Self::Compare { a, b, attr, most } => ITEMS[Self::winner(world, a, b, attr, most)].nom.to_string(),
            Self::Filter(cond) => world
                .catalog
                .iter()
                .find(|&&i| cond.holds(world, i))
                .map_or(String::new(), |&i| ITEMS[i].nom.to_string()),
            Self::Calc { .. } | Self::Total { .. } => {
                calc::run(&self.calc_expr(world).expect("has an expression")).unwrap_or_else(|e| e.to_string())
            }
            Self::Chat(c) => c.reply().to_string(),
        }
    }
}

/// Lower-case words of a text (`ё` → `е`).
fn words_of(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace('ё', "е")
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// Whether one of `forms` occurs among `words`.
fn mentions(words: &[String], forms: &[&str]) -> bool {
    forms.iter().any(|f| {
        let f = f.to_lowercase().replace('ё', "е");
        words.contains(&f)
    })
}

/// The product named first, in any of its case forms.
fn first_item(words: &[String]) -> Option<usize> {
    words
        .iter()
        .find_map(|w| ITEMS.iter().position(|it| [it.nom, it.gen, it.acc].iter().any(|f| f.replace('ё', "е") == *w)))
}

/// The numbers a text states, in order (`-` directly before a digit is a sign, `,` a decimal
/// comma): «5 + 5 = 10» → `5 5 10`.
pub fn numbers(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_digit() {
            let neg = i > 0 && c[i - 1] == '-' && (i < 2 || !c[i - 2].is_ascii_digit());
            let mut n = String::from(if neg { "-" } else { "" });
            while i < c.len() && c[i].is_ascii_digit() {
                n.push(c[i]);
                i += 1;
            }
            if i + 1 < c.len() && (c[i] == '.' || c[i] == ',') && c[i + 1].is_ascii_digit() {
                n.push('.');
                i += 1;
                while i < c.len() && c[i].is_ascii_digit() {
                    n.push(c[i]);
                    i += 1;
                }
            }
            out.push(n);
        } else {
            i += 1;
        }
    }
    out
}

/// Sentences the teacher answers with. Every value and every product name is written as the
/// page, the question or the calculator shows it — lower-case, mid-sentence — so the model can
/// copy it (a capitalised «Лампа» at the start of a sentence would be a different token).
pub mod say {
    use super::super::world::{Attr, ITEMS};
    use super::TotalKind;

    /// `стоит` / `стоят` (plural-only nouns like «часы»).
    fn costs(item: usize) -> &'static str {
        if ITEMS[item].nom == "часы" {
            "стоят"
        } else {
            "стоит"
        }
    }

    /// «Сейчас лампа стоит 12 ₽.», «Цвет — красный.», «Бренд — Nova.», «Рейтинг — 4 ★.»
    pub fn lookup(item: usize, attr: Attr, shown: &str) -> String {
        match attr {
            Attr::Price => format!("Сейчас {} {} {shown}.", ITEMS[item].nom, costs(item)),
            Attr::Color => format!("Цвет — {shown}."),
            Attr::Brand => format!("Бренд — {shown}."),
            Attr::Rating => format!("Рейтинг — {shown}."),
        }
    }

    /// «Дешевле лампа.», «По рейтингу лучше лампа.»
    pub fn compare(winner: usize, attr: Attr, most: bool) -> String {
        let it = ITEMS[winner].nom;
        match (attr, most) {
            (Attr::Rating, true) => format!("По рейтингу лучше {it}."),
            (Attr::Rating, false) => format!("По рейтингу хуже {it}."),
            (_, true) => format!("Дороже {it}."),
            (_, false) => format!("Дешевле {it}."),
        }
    }

    /// «Например, лампа.»
    pub fn filter(item: usize) -> String {
        format!("Например, {}.", ITEMS[item].nom)
    }

    /// «12 * 3 = 36» — the calculator's line.
    pub fn calc(expr: &str, result: &str) -> String {
        format!("{} = {result}", expr.trim())
    }

    /// «Вместе 19 ₽.», «Дороже на 5 ₽.», «Дешевле на 5 ₽.»
    pub fn total(kind: TotalKind, result: &str) -> String {
        match kind {
            TotalKind::Sum => format!("Вместе {result} ₽."),
            TotalKind::Pricier => format!("Дороже на {result} ₽."),
            TotalKind::Cheaper => format!("Дешевле на {result} ₽."),
        }
    }
}

/// An instruction: the text the model reads and, when known, its structured meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Goal {
    pub text: String,
    pub spec: Option<Spec>,
}

/// Which templates to draw from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Templates used for training.
    Train,
    /// Held-out templates only.
    HeldOut,
}

/// `(template, held out)`.
type Templates = &'static [(&'static str, bool)];

// Lookup: {nom} {gen} {acc}.
const LOOKUP_PRICE: Templates = &[
    ("Сколько стоит {nom}?", false),
    ("Какая цена у {gen}?", false),
    ("Найди цену {gen}.", false),
    ("Узнай стоимость {gen}.", false),
    ("Посмотри, сколько стоит {nom}.", false),
    ("Мне нужна цена {gen}.", false),
    ("Цена {gen}?", false),
    ("Во сколько обойдётся {nom}?", true),
    ("Подскажи, сколько рублей стоит {nom}.", true),
];
const LOOKUP_COLOR: Templates = &[
    ("Какого цвета {nom}?", false),
    ("Найди цвет {gen}.", false),
    ("Какой цвет у {gen}?", false),
    ("Узнай, какого цвета {nom}.", false),
    ("Цвет {gen}?", false),
    ("В каком цвете продаётся {nom}?", true),
    ("Посмотри, какой окраски {nom}.", true),
];
const LOOKUP_BRAND: Templates = &[
    ("Какой бренд у {gen}?", false),
    ("Кто производитель {gen}?", false),
    ("Найди бренд {gen}.", false),
    ("Какой фирмы {nom}?", false),
    ("Кто выпускает {acc}?", false),
    ("Какая марка у {gen}?", true),
    ("Узнай, кто сделал {acc}.", true),
];
const LOOKUP_RATING: Templates = &[
    ("Какой рейтинг у {gen}?", false),
    ("Сколько звёзд у {gen}?", false),
    ("Найди рейтинг {gen}.", false),
    ("Какая оценка у {gen}?", false),
    ("Узнай рейтинг {gen}.", false),
    ("Как покупатели оценивают {acc}?", true),
    ("На сколько звёзд оценили {acc}?", true),
];
// Compare: {a} {b} (nom), {a_gen} {b_gen}, {a_acc} {b_acc}.
const CHEAPER: Templates = &[
    ("Что дешевле: {a} или {b}?", false),
    ("Какой товар дешевле — {a} или {b}?", false),
    ("Сравни цены: {a} и {b}. Что дешевле?", false),
    ("Что стоит меньше: {a} или {b}?", false),
    ("Что выгоднее купить: {a} или {b}?", true),
    ("Выбери более дешёвый товар: {a} или {b}.", true),
];
const PRICIER: Templates = &[
    ("Что дороже: {a} или {b}?", false),
    ("Какой товар дороже — {a} или {b}?", false),
    ("Что стоит больше: {a} или {b}?", false),
    ("Сравни {a_acc} и {b_acc}: что из них дороже?", true),
    ("Какая покупка обойдётся дороже: {a} или {b}?", true),
];
const RATED_HIGHER: Templates = &[
    ("У чего рейтинг выше: у {a_gen} или у {b_gen}?", false),
    ("Что лучше оценено: {a} или {b}?", false),
    ("Какой товар с более высоким рейтингом — {a} или {b}?", false),
    ("У какого товара больше звёзд: у {a_gen} или у {b_gen}?", true),
];
const RATED_LOWER: Templates = &[
    ("У чего рейтинг ниже: у {a_gen} или у {b_gen}?", false),
    ("Что хуже оценено: {a} или {b}?", false),
    ("Какой товар с более низким рейтингом — {a} или {b}?", false),
    ("У какого товара меньше звёзд: у {a_gen} или у {b_gen}?", true),
];
// Filter: {x}, {cnom} {cgen}.
const BELOW: Templates = &[
    ("Найди товар дешевле {x} ₽.", false),
    ("Найди что-нибудь дешевле {x} рублей.", false),
    ("Нужен товар стоимостью меньше {x} ₽.", false),
    ("Покажи товар, который стоит меньше {x} ₽.", false),
    ("Подбери любой товар по цене ниже {x} рублей.", true),
];
const ABOVE: Templates = &[
    ("Найди товар дороже {x} ₽.", false),
    ("Найди что-нибудь дороже {x} рублей.", false),
    ("Нужен товар стоимостью больше {x} ₽.", false),
    ("Покажи товар, который стоит больше {x} ₽.", false),
    ("Подбери любой товар по цене выше {x} рублей.", true),
];
const COLORED: Templates = &[
    ("Найди товар {cgen} цвета.", false),
    ("Найди что-нибудь {cgen} цвета.", false),
    ("Нужен {cnom} товар.", false),
    ("Покажи товар, у которого цвет {cnom}.", false),
    ("Подбери любую вещь {cgen} цвета.", true),
];

// Calc, symbolic: {e} is the expression as written (`12+30` or `12 + 30`).
const CALC_SYMBOLS: Templates = &[
    ("Сколько будет {e}?", false),
    ("Посчитай {e}.", false),
    ("Вычисли {e}.", false),
    ("Чему равно {e}?", false),
    ("Реши пример: {e}.", false),
    ("{e} = ?", false),
    ("Сколько получится: {e}?", false),
    ("Помоги посчитать {e}.", true),
    ("Какой ответ у примера {e}?", true),
];
// Calc in words: {a} {op} {b} with the spoken operation («12 умножить на 3»).
const CALC_WORDS: Templates = &[
    ("Сколько будет {a} {op} {b}?", false),
    ("Чему равно {a} {op} {b}?", false),
    ("Посчитай, сколько будет {a} {op} {b}.", false),
    ("Вычисли {a} {op} {b}.", false),
    ("Сколько получится, если {a} {op} {b}?", true),
];
const CALC_ADD: Templates = &[("Сложи {a} и {b}.", false), ("Найди сумму {a} и {b}.", true)];
const CALC_SUB: Templates =
    &[("Вычти {b} из {a}.", false), ("Отними от {a} {b}.", false), ("Найди разность {a} и {b}.", true)];
const CALC_MUL: Templates = &[("Умножь {a} на {b}.", false), ("Найди произведение {a} и {b}.", true)];
const CALC_DIV: Templates = &[("Раздели {a} на {b}.", false), ("Найди частное {a} и {b}.", true)];
// Total: {a} {b} (nom), {a_gen} {b_gen}, {a_acc} {b_acc}.
const TOTAL_SUM: Templates = &[
    ("Сколько стоят вместе {a} и {b}?", false),
    ("Сколько будут стоить {a} и {b} вместе?", false),
    ("Какова общая стоимость {a_gen} и {b_gen}?", false),
    ("Сколько нужно заплатить за {a_acc} и {b_acc}?", false),
    ("Посчитай, сколько стоят {a} и {b} вместе.", false),
    ("Во сколько обойдутся {a} и {b} вместе?", true),
    ("Сколько рублей стоят {a} и {b} в сумме?", true),
];
const TOTAL_PRICIER: Templates = &[
    ("На сколько {a} дороже, чем {b}?", false),
    ("На сколько {a} дороже {b_gen}?", false),
    ("Насколько {a} дороже {b_gen}?", false),
    ("На сколько рублей {a} дороже, чем {b}?", true),
];
const TOTAL_CHEAPER: Templates = &[
    ("На сколько {a} дешевле, чем {b}?", false),
    ("На сколько {a} дешевле {b_gen}?", false),
    ("Насколько {a} дешевле {b_gen}?", false),
    ("На сколько рублей {a} дешевле, чем {b}?", true),
];
// Chat.
const CHAT_HELLO: Templates = &[
    ("Привет!", false),
    ("Привет", false),
    ("Здравствуй!", false),
    ("Добрый день!", false),
    ("Здравствуйте", false),
    ("Приветствую!", true),
    ("Доброе утро!", true),
];
const CHAT_HOW: Templates =
    &[("Как дела?", false), ("Как ты?", false), ("Как у тебя дела?", false), ("Как поживаешь?", true)];
const CHAT_THANKS: Templates =
    &[("Спасибо!", false), ("Спасибо", false), ("Благодарю!", false), ("Спасибо большое!", true)];
const CHAT_BYE: Templates = &[("Пока!", false), ("До свидания!", false), ("Пока", false), ("Всего доброго!", true)];
const CHAT_WHO: Templates =
    &[("Кто ты?", false), ("Ты кто?", false), ("Представься.", false), ("Расскажи о себе.", true)];
const CHAT_SKILLS: Templates = &[
    ("Что ты умеешь?", false),
    ("Что ты можешь?", false),
    ("Чем ты можешь помочь?", false),
    ("Какие у тебя возможности?", true),
];

fn lookup_templates(attr: Attr) -> Templates {
    match attr {
        Attr::Price => LOOKUP_PRICE,
        Attr::Color => LOOKUP_COLOR,
        Attr::Brand => LOOKUP_BRAND,
        Attr::Rating => LOOKUP_RATING,
    }
}

fn compare_templates(attr: Attr, most: bool) -> Templates {
    match (attr, most) {
        (Attr::Price, false) => CHEAPER,
        (Attr::Price, true) => PRICIER,
        (_, true) => RATED_HIGHER,
        (_, false) => RATED_LOWER,
    }
}

fn filter_templates(cond: Cond) -> Templates {
    match cond {
        Cond::PriceBelow(_) => BELOW,
        Cond::PriceAbove(_) => ABOVE,
        Cond::Color(_) => COLORED,
    }
}

fn calc_templates(expr: &Expr, form: CalcForm) -> Vec<(&'static str, bool)> {
    match form {
        CalcForm::Compact | CalcForm::Spaced => CALC_SYMBOLS.to_vec(),
        CalcForm::Words => {
            let verbs = match expr.ops[0] {
                Op::Add => CALC_ADD,
                Op::Sub => CALC_SUB,
                Op::Mul => CALC_MUL,
                Op::Div => CALC_DIV,
            };
            [CALC_WORDS, verbs].concat()
        }
    }
}

fn templates(spec: &Spec) -> Vec<(&'static str, bool)> {
    match *spec {
        Spec::Lookup { attr, .. } => lookup_templates(attr).to_vec(),
        Spec::Compare { attr, most, .. } => compare_templates(attr, most).to_vec(),
        Spec::Filter(cond) => filter_templates(cond).to_vec(),
        Spec::Calc { expr, form } => calc_templates(&expr, form),
        Spec::Total { kind, .. } => match kind {
            TotalKind::Sum => TOTAL_SUM,
            TotalKind::Pricier => TOTAL_PRICIER,
            TotalKind::Cheaper => TOTAL_CHEAPER,
        }
        .to_vec(),
        Spec::Chat(c) => match c {
            Chat::Hello => CHAT_HELLO,
            Chat::HowAreYou => CHAT_HOW,
            Chat::Thanks => CHAT_THANKS,
            Chat::Bye => CHAT_BYE,
            Chat::Who => CHAT_WHO,
            Chat::Skills => CHAT_SKILLS,
        }
        .to_vec(),
    }
}

/// Fills a template for `spec`.
fn fill(template: &str, spec: &Spec) -> String {
    let mut s = template.to_string();
    let mut put = |key: &str, value: &str| s = s.replace(key, value);
    match *spec {
        Spec::Lookup { item, .. } => {
            let it = ITEMS[item];
            put("{nom}", it.nom);
            put("{gen}", it.gen);
            put("{acc}", it.acc);
        }
        Spec::Compare { a, b, .. } | Spec::Total { a, b, .. } => {
            let (a, b) = (ITEMS[a], ITEMS[b]);
            put("{a_gen}", a.gen);
            put("{b_gen}", b.gen);
            put("{a_acc}", a.acc);
            put("{b_acc}", b.acc);
            put("{a}", a.nom);
            put("{b}", b.nom);
        }
        Spec::Filter(cond) => match cond {
            Cond::PriceBelow(x) | Cond::PriceAbove(x) => put("{x}", &x.to_string()),
            Cond::Color(c) => {
                put("{cnom}", COLORS[c].0);
                put("{cgen}", COLORS[c].1);
            }
        },
        Spec::Calc { expr, form } => {
            put("{e}", &expr.text(form == CalcForm::Spaced));
            put("{a}", &expr.nums[0].to_string());
            put("{b}", &expr.nums[1].to_string());
            put("{op}", expr.ops[0].word());
        }
        Spec::Chat(_) => {}
    }
    s
}

fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_lowercase().chain(c).collect())
}

/// Random surface variation: lower-case start, dropped final punctuation, «пожалуйста», and
/// letter case — product names written with a capital («Сколько стоит Лампа?»), a short line in
/// capitals or a line in lower case — so the model learns that «Лампа», «ЛАМПА» and «лампа» (different
/// tokens) are one word.
fn perturb(rng: &mut Rng, text: String, chat: bool) -> String {
    let mut t = text;
    if !chat && rng.uniform() < 0.1 {
        t = format!("Пожалуйста, {}", lower_first(&t));
    }
    if rng.uniform() < 0.25 {
        t = lower_first(&t);
    }
    if rng.uniform() < 0.2 {
        t = t.trim_end_matches(['?', '.', '!']).to_string();
    }
    match rng.uniform() {
        x if x < 0.12 => t = capitalize_items(&t),
        // capitals take ~2 tokens a letter: only short lines are shouted
        x if x < 0.16 && t.chars().count() <= 24 => t = t.to_uppercase(),
        x if x < 0.20 => t = t.to_lowercase(),
        _ => {}
    }
    t
}

/// Every product name (any case form) with a capital first letter.
fn capitalize_items(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        let lower = word.to_lowercase();
        if ITEMS.iter().any(|it| [it.nom, it.gen, it.acc].contains(&lower.as_str())) {
            let mut c = word.chars();
            if let Some(f) = c.next() {
                out.extend(f.to_uppercase());
                out.push_str(c.as_str());
            }
        } else {
            out.push_str(word);
        }
        word.clear();
    };
    for ch in text.chars() {
        if ch.is_alphabetic() {
            word.push(ch);
        } else {
            flush(&mut word, &mut out);
            out.push(ch);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// A random operand: mostly small, sometimes up to three digits.
fn operand(rng: &mut Rng) -> u32 {
    match rng.uniform() {
        x if x < 0.5 => rng.below(21) as u32,
        x if x < 0.8 => rng.below(100) as u32,
        _ => rng.below(1000) as u32,
    }
}

/// A random arithmetic expression (division by zero excluded; most divisions are exact).
pub fn sample_expr(rng: &mut Rng) -> Expr {
    let op = Op::ALL[rng.below(4)];
    let (a, b) = match op {
        Op::Add | Op::Sub => (operand(rng), operand(rng)),
        Op::Mul => (operand(rng), if rng.uniform() < 0.7 { rng.below(21) as u32 } else { rng.below(100) as u32 }),
        Op::Div => {
            let b = 1 + rng.below(20) as u32;
            if rng.uniform() < 0.8 {
                (b * rng.below(51) as u32, b)
            } else {
                (rng.below(101) as u32, b)
            }
        }
    };
    let mut e = Expr::binary(a, op, b);
    if rng.uniform() < 0.2 {
        // three small operands, e.g. «2+3*4», «(12-2)*3»
        e.nums = [rng.below(51) as u32, rng.below(51) as u32, 1 + rng.below(20) as u32];
        e.ops = [Op::ALL[rng.below(4)], Op::ALL[rng.below(4)]];
        e.three = true;
        e.parens = rng.uniform() < 0.3;
        if e.value().is_err() {
            return Expr::binary(a, op, b);
        }
    }
    e
}

/// A random spec that has an answer in `world`.
pub fn sample_spec(rng: &mut Rng, world: &World, family: Family) -> Spec {
    let n = ITEMS.len();
    match family {
        Family::Calc => {
            let expr = sample_expr(rng);
            let form = match rng.uniform() {
                _ if expr.three => [CalcForm::Compact, CalcForm::Spaced][rng.below(2)],
                x if x < 0.4 => CalcForm::Compact,
                x if x < 0.65 => CalcForm::Spaced,
                _ => CalcForm::Words,
            };
            Spec::Calc { expr, form }
        }
        Family::Total => loop {
            let (a, b) = (rng.below(n), rng.below(n));
            let (pa, pb) = (world.price[a], world.price[b]);
            if a == b {
                continue;
            }
            let kind = if rng.uniform() < 0.5 {
                TotalKind::Sum
            } else if pa > pb {
                TotalKind::Pricier
            } else if pa < pb {
                TotalKind::Cheaper
            } else {
                continue;
            };
            return Spec::Total { a, b, kind };
        },
        Family::Chat => Spec::Chat(Chat::ALL[rng.below(Chat::ALL.len())]),
        Family::Lookup => Spec::Lookup { item: rng.below(n), attr: Attr::ALL[rng.below(4)] },
        Family::Compare => {
            let attr = if rng.uniform() < 0.5 { Attr::Price } else { Attr::Rating };
            let v = |i: usize| if attr == Attr::Price { world.price[i] } else { world.rating[i] };
            loop {
                let (a, b) = (rng.below(n), rng.below(n));
                if a != b && v(a) != v(b) {
                    return Spec::Compare { a, b, attr, most: rng.uniform() < 0.5 };
                }
            }
        }
        Family::Filter => loop {
            // derived from a random product, so at least one product matches
            let i = rng.below(n);
            let cond = match rng.below(3) {
                0 => Cond::PriceBelow((world.price[i] + 1 + rng.below(3) as u32).min(MAX_PRICE + 1)),
                1 => match world.price[i].checked_sub(1 + rng.below(3) as u32) {
                    Some(x) if x >= 1 => Cond::PriceAbove(x),
                    _ => continue,
                },
                _ => Cond::Color(world.color[i]),
            };
            return Spec::Filter(cond);
        },
    }
}

/// A random wording of `spec` from the templates of `split`.
pub fn phrase(rng: &mut Rng, spec: &Spec, split: Split) -> String {
    let pool: Vec<&str> =
        templates(spec).iter().filter(|(_, held)| *held == (split == Split::HeldOut)).map(|(t, _)| *t).collect();
    let text = fill(pool[rng.below(pool.len())], spec);
    perturb(rng, text, spec.family() == Family::Chat)
}

/// A random goal of `family` answerable in `world`.
pub fn sample(rng: &mut Rng, world: &World, family: Family, split: Split) -> Goal {
    let spec = sample_spec(rng, world, family);
    Goal { text: phrase(rng, &spec, split), spec: Some(spec) }
}

fn normalize(text: &str) -> String {
    let t = text.to_lowercase().replace('ё', "е");
    let words: Vec<&str> =
        t.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty() && *w != "пожалуйста").collect();
    words.join(" ")
}

/// An arithmetic question in free form: the expression is the question with operation words
/// replaced by symbols («12 умножить на 3» → `12*3`, «Сложи 5 и 7» → `5+7`) and every other
/// word dropped.
fn recognize_calc(text: &str) -> Option<Spec> {
    let t = text.to_lowercase().replace('ё', "е");
    let words: Vec<String> = t
        .split(|c: char| c.is_whitespace() || matches!(c, '?' | '!' | ',' | ':' | ';' | '='))
        .map(|w| w.trim_end_matches('.').to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let num = |w: &str| w.parse::<u32>().ok();
    let verb = |i: usize| -> Option<Expr> {
        let w = |k: usize| words.get(i + k).map(String::as_str);
        let (x, y) = (w(1).and_then(num), w(3).and_then(num));
        match (w(0)?, w(2)?) {
            ("сложи", "и") => Some(Expr::binary(x?, Op::Add, y?)),
            ("вычти", "из") => Some(Expr::binary(y?, Op::Sub, x?)),
            ("умножь", "на") => Some(Expr::binary(x?, Op::Mul, y?)),
            ("раздели", "на") => Some(Expr::binary(x?, Op::Div, y?)),
            ("отними", _) if w(1) == Some("от") => {
                Some(Expr::binary(num(w(2)?)?, Op::Sub, w(3).and_then(num)?))
            }
            ("сумму" | "разность" | "произведение" | "частное", _) => {
                let op = match w(0)? {
                    "сумму" => Op::Add,
                    "разность" => Op::Sub,
                    "произведение" => Op::Mul,
                    _ => Op::Div,
                };
                (w(2)? == "и").then_some(())?;
                Some(Expr::binary(x?, op, y?))
            }
            _ => None,
        }
    };
    if let Some(e) = (0..words.len()).find_map(verb) {
        return Some(Spec::Calc { expr: e, form: CalcForm::Words });
    }
    let mut expr = String::new();
    let mut it = words.iter().peekable();
    while let Some(w) = it.next() {
        match w.as_str() {
            "плюс" => expr.push('+'),
            "минус" => expr.push('-'),
            "умножить" | "умноженное" | "разделить" | "деленное" if it.peek().map(|n| n.as_str()) == Some("на") =>
            {
                it.next();
                expr.push(if w.starts_with('у') { '*' } else { '/' });
            }
            w if w.chars().all(|c| c.is_ascii_digit() || "+-*/()".contains(c)) => expr.push_str(w),
            _ => {}
        }
    }
    let e = Expr::parse(&expr)?;
    e.value().ok()?;
    let form = if text.contains(|c: char| "+-*/".contains(c)) { CalcForm::Compact } else { CalcForm::Words };
    Some(Spec::Calc { expr: e, form })
}

/// Recognises a question written with one of the templates (any case, punctuation, «ё»/«е»,
/// «пожалуйста»), or any arithmetic question — used only to check answers to free-form
/// questions in the CLI; the model itself reads the raw text.
pub fn recognize(text: &str) -> Option<Spec> {
    static INDEX: OnceLock<HashMap<String, Spec>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let n = ITEMS.len();
        let mut specs = Vec::new();
        for item in 0..n {
            specs.extend(Attr::ALL.map(|attr| Spec::Lookup { item, attr }));
        }
        for a in 0..n {
            for b in (0..n).filter(|&b| b != a) {
                for attr in [Attr::Price, Attr::Rating] {
                    specs.extend([true, false].map(|most| Spec::Compare { a, b, attr, most }));
                }
            }
        }
        for x in 1..=MAX_PRICE + 1 {
            specs.extend([Spec::Filter(Cond::PriceBelow(x)), Spec::Filter(Cond::PriceAbove(x))]);
        }
        specs.extend((0..COLORS.len()).map(|c| Spec::Filter(Cond::Color(c))));
        for a in 0..n {
            for b in (0..n).filter(|&b| b != a) {
                specs.extend([TotalKind::Sum, TotalKind::Pricier, TotalKind::Cheaper].map(|kind| Spec::Total {
                    a,
                    b,
                    kind,
                }));
            }
        }
        specs.extend(Chat::ALL.map(Spec::Chat));
        let mut index = HashMap::new();
        for spec in specs {
            for (t, _) in templates(&spec) {
                index.insert(normalize(&fill(t, &spec)), spec);
            }
        }
        index
    });
    index.get(&normalize(text)).copied().or_else(|| recognize_calc(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_family_has_both_splits() {
        let all: [Templates; 26] = [
            LOOKUP_PRICE,
            LOOKUP_COLOR,
            LOOKUP_BRAND,
            LOOKUP_RATING,
            CHEAPER,
            PRICIER,
            RATED_HIGHER,
            RATED_LOWER,
            BELOW,
            ABOVE,
            COLORED,
            CALC_SYMBOLS,
            CALC_WORDS,
            CALC_ADD,
            CALC_SUB,
            CALC_MUL,
            CALC_DIV,
            TOTAL_SUM,
            TOTAL_PRICIER,
            TOTAL_CHEAPER,
            CHAT_HELLO,
            CHAT_HOW,
            CHAT_THANKS,
            CHAT_BYE,
            CHAT_WHO,
            CHAT_SKILLS,
        ];
        for t in all {
            assert!(t.iter().any(|x| x.1) && t.iter().any(|x| !x.1));
        }
    }

    #[test]
    fn letter_case_varies_but_questions_stay_recognisable() {
        assert_eq!(capitalize_items("Что дешевле: лампа или стула?"), "Что дешевле: Лампа или Стула?");
        let mut rng = Rng::new(8);
        let world = World::new(3);
        let (mut caps, mut upper) = (0, 0);
        for _ in 0..2000 {
            let g = sample(&mut rng, &world, Family::Lookup, Split::Train);
            caps += g.text.contains(|c: char| c.is_uppercase()) as usize;
            upper += (g.text == g.text.to_uppercase()) as usize;
            assert_eq!(recognize(&g.text), g.spec, "{}", g.text);
        }
        assert!(upper > 10 && caps > upper, "{caps} {upper}");
    }

    #[test]
    fn sentence_answers_are_checked_by_what_they_state() {
        let w = World::new(5);
        let p = w.price[0];
        let lookup = Spec::Lookup { item: 0, attr: Attr::Price };
        assert!(lookup.accepts(&w, &say::lookup(0, Attr::Price, &super::super::world::fmt_price(p))));
        assert!(lookup.accepts(&w, &format!("{p}")));
        assert!(!lookup.accepts(&w, &format!("Лампа стоит {} ₽.", p + 1)));
        assert!(!lookup.accepts(&w, &format!("Лампа стоит {p} ₽, стул — {} ₽.", p + 1)), "two prices");
        let color = Spec::Lookup { item: 0, attr: Attr::Color };
        assert!(color.accepts(&w, &say::lookup(0, Attr::Color, COLORS[w.color[0]].0)));
        assert!(!color.accepts(&w, &format!("Цвет лампы — {}.", COLORS[(w.color[0] + 1) % COLORS.len()].0)));
        let cmp = Spec::Compare { a: 0, b: 1, attr: Attr::Price, most: false };
        let (win, lose) = if w.price[0] < w.price[1] { (0, 1) } else { (1, 0) };
        assert!(cmp.accepts(&w, &say::compare(win, Attr::Price, false)));
        assert!(cmp.accepts(&w, &format!("{} дешевле, чем {}.", ITEMS[win].nom, ITEMS[lose].nom)));
        assert!(!cmp.accepts(&w, &say::compare(lose, Attr::Price, false)));
        let calc = Spec::Calc { expr: Expr::binary(12, Op::Mul, 3), form: CalcForm::Spaced };
        assert!(calc.accepts(&w, &say::calc("12 * 3", "36")));
        assert!(!calc.accepts(&w, "12 * 3 = 35"));
        let sub = Spec::Calc { expr: Expr::binary(100, Op::Sub, 250), form: CalcForm::Spaced };
        assert!(sub.accepts(&w, "100 - 250 = -150") && !sub.accepts(&w, "100 - 250 = 150"));
        let total = Spec::Total { a: 0, b: 1, kind: TotalKind::Sum };
        let sum = (w.price[0] + w.price[1]).to_string();
        assert!(total.accepts(&w, &say::total(TotalKind::Sum, &sum)));
        assert_eq!(numbers("Вместе 2,5 ₽ и -3"), ["2.5", "-3"]);
    }

    #[test]
    fn goals_are_answerable_and_recognisable() {
        let mut rng = Rng::new(3);
        for s in 0..300u64 {
            let world = World::new(s);
            for family in Family::ALL {
                for split in [Split::Train, Split::HeldOut] {
                    let g = sample(&mut rng, &world, family, split);
                    let spec = g.spec.unwrap();
                    assert!(!g.text.contains('{'), "{}", g.text);
                    let expected = spec.expected(&world);
                    assert!(spec.accepts(&world, &expected), "{} → {expected}", g.text);
                    let said = spec.answer_sentence(&world);
                    assert!(spec.accepts(&world, &said), "{} → {said}", g.text);
                    match spec {
                        // the free-form parser recovers the expression, not the wording
                        Spec::Calc { expr, .. } => match recognize(&g.text) {
                            Some(Spec::Calc { expr: e, .. }) => assert_eq!(e.value(), expr.value(), "{}", g.text),
                            other => panic!("{} → {other:?}", g.text),
                        },
                        _ => assert_eq!(recognize(&g.text), Some(spec), "{}", g.text),
                    }
                }
            }
        }
        let w = World::new(1);
        let spec = recognize("сколько стоит лампа").unwrap();
        assert_eq!(spec, Spec::Lookup { item: 0, attr: Attr::Price });
        assert!(spec.accepts(&w, &w.price[0].to_string()));
        assert_eq!(
            recognize("Что ДЕШЕВЛЕ: лампа или стул"),
            Some(Spec::Compare { a: 0, b: 1, attr: Attr::Price, most: false })
        );
        assert_eq!(recognize("Найди товар зеленого цвета"), Some(Spec::Filter(Cond::Color(2))));
        assert_eq!(recognize("расскажи анекдот"), None);
        let calc = |t: &str| match recognize(t) {
            Some(s @ Spec::Calc { .. }) => s.expected(&w),
            other => panic!("{t} → {other:?}"),
        };
        assert_eq!(calc("сколько будет 5+5"), "10");
        assert_eq!(calc("Сколько будет 7 умножить на 8?"), "56");
        assert_eq!(calc("Посчитай (2+3)*4"), "20");
        assert_eq!(calc("Вычти 5 из 12."), "7");
        assert_eq!(calc("отними от 12 5"), "7");
        assert_eq!(calc("Раздели 10 на 4"), "2.5");
        assert_eq!(recognize("Привет!"), Some(Spec::Chat(Chat::Hello)));
        assert!(Spec::Chat(Chat::Hello).accepts(&w, "привет! чем помочь?"));
        assert!(Spec::Chat(Chat::Hello).accepts(&w, "Здравствуйте!"));
        assert!(!Spec::Chat(Chat::Hello).accepts(&w, "Лампа стоит 12 ₽."));
        let t = recognize("Сколько стоят вместе лампа и стул?").unwrap();
        assert_eq!(t.expected(&w), (w.price[0] + w.price[1]).to_string());
    }
}
