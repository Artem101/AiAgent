#!/usr/bin/env python3
"""Builds the school dataset: Russian language and mathematics, grades 1–7, as chains of
reasoning steps an agent performs — never raw paragraphs.

    scripts/fetch_ru20k.sh && scripts/fetch_school.sh          # sources → data/ru20k, data/school
    scripts/build_school.py [data/school] [data/ru20k]         # → data/school/*.jsonl

Every example is one JSON object per line:

    {"id": "m4_meeting_00017", "grade": 4, "subject": "математика", "topic": "движение навстречу",
     "split": "train",
     "question": "Два велосипедиста выехали навстречу друг другу …",
     "steps": [
       {"act": "THINK",  "text": "Они сближаются: за час расстояние сокращается на 12 + 15 км."},
       {"act": "CALC",   "text": "12 + 15", "result": "27"},
       {"act": "THINK",  "text": "Время встречи = расстояние : скорость сближения."},
       {"act": "CALC",   "text": "81 / 27", "result": "3"},
       {"act": "ANSWER", "text": "Они встретятся через 3 ч."}],
     "check": {"type": "number", "value": "3"}}

Actions: THINK (a step of reasoning, written into the agent's scratchpad), CALC (the agent's
calculator: `+ - * /`, parentheses, decimals — the result is computed here exactly as the
calculator prints it), LOOKUP (the dictionary tool: the result is the entry of
`dictionary.jsonl`), ANSWER. Browser trajectories keep their own format (exported by
`cog_engine export-trajectories`).

Outputs (data/school):
  chains.jsonl       all examples; "split" is train / valid / test (valid and test hold out
                     whole words for language topics and whole question texts for mathematics)
  dictionary.jsonl   the LOOKUP dictionary: {"word", "pos", "entry", …}
  stats.txt          counts per grade, subject and topic
"""

import ast
import collections
import csv
import hashlib
import json
import random
import re
import sys
from fractions import Fraction

SCHOOL = sys.argv[1] if len(sys.argv) > 1 else "data/school"
RU20K = sys.argv[2] if len(sys.argv) > 2 else "data/ru20k"
PER_TOPIC = 4000
csv.field_size_limit(10**7)

# ───────────────────────── the calculator (same program as the agent's Python worker) ─────────
LIMIT, MAX_EXPR, MAX_DIGITS, DECIMALS = 10**15, 64, 18, 4
ALPHABET = set("0123456789+-*/(). ")


def _value(node, src):
    if isinstance(node, ast.Expression):
        return _value(node.body, src)
    if isinstance(node, ast.Constant):
        lit = ast.get_source_segment(src, node)
        assert sum(c.isdigit() for c in lit) <= MAX_DIGITS
        return Fraction(lit)
    if isinstance(node, ast.UnaryOp):
        v = _value(node.operand, src)
        return -v if isinstance(node.op, ast.USub) else v
    a, b = _value(node.left, src), _value(node.right, src)
    op = type(node.op)
    r = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b}[op]()
    assert abs(r.numerator) <= LIMIT and r.denominator <= LIMIT
    return r


def show(x):
    if x.denominator == 1:
        return str(x.numerator)
    scale = 10**DECIMALS
    scaled = (2 * abs(x.numerator) * scale + x.denominator) // (2 * x.denominator)
    whole, frac = divmod(scaled, scale)
    sign = "-" if x < 0 and scaled != 0 else ""
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}." + f"{frac:0{DECIMALS}d}".rstrip("0")


def calc(expr):
    """(printed result, exact value) of a calculator expression."""
    assert len(expr) <= MAX_EXPR and set(expr) <= ALPHABET, expr
    v = _value(ast.parse(expr, mode="eval"), expr)
    return show(v), v


def ru(x):
    """A number as written in a Russian text (decimal comma)."""
    s = show(x) if isinstance(x, Fraction) else str(x)
    return s.replace(".", ",")


def plural(n, forms):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


# ───────────────────────── chain building ─────────────────────────
class Chain:
    def __init__(self, question):
        self.question = question
        self.steps = []

    def think(self, text):
        self.steps.append({"act": "THINK", "text": text})
        return self

    def calc(self, expr):
        out, v = calc(expr)
        self.steps.append({"act": "CALC", "text": expr, "result": out})
        return v

    def lookup(self, word):
        e = DICT.get(word)
        assert e is not None, word
        self.steps.append({"act": "LOOKUP", "text": word, "result": e["entry"]})
        return e

    def answer(self, text, check_type, value):
        self.steps.append({"act": "ANSWER", "text": text})
        self.check = {"type": check_type, "value": value}
        return self


def held(key, share):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000 < share * 10000


def split_of(key):
    return "test" if held("test:" + key, 0.03) else "valid" if held("valid:" + key, 0.03) else "train"


EXAMPLES = []


def topic(grade, subject, code, name, n=PER_TOPIC):
    """Registers a generator `f(rng) -> (Chain, split key) | None` and runs it until `n`
    distinct questions are made."""

    def deco(f):
        rng = random.Random(code)
        seen = set()
        tries = 0
        while len(seen) < n and tries < n * 30:
            tries += 1
            made = f(rng)
            if made is None:
                continue
            ch, key = made
            if ch.question in seen:
                continue
            seen.add(ch.question)
            EXAMPLES.append({
                "id": f"{code}_{len(seen):05d}",
                "grade": grade,
                "subject": subject,
                "topic": name,
                "split": split_of(key),
                "question": ch.question,
                "steps": ch.steps,
                "check": ch.check,
            })
        return f

    return deco


MATH, LANG = "математика", "русский язык"

# ───────────────────────── mathematics ─────────────────────────
NAMES = [  # name, genitive, dative, gender
    ("Маша", "Маши", "Маше", "f"), ("Петя", "Пети", "Пете", "m"), ("Коля", "Коли", "Коле", "m"),
    ("Оля", "Оли", "Оле", "f"), ("Саша", "Саши", "Саше", "m"), ("Аня", "Ани", "Ане", "f"),
    ("Дима", "Димы", "Диме", "m"), ("Катя", "Кати", "Кате", "f"), ("Ваня", "Вани", "Ване", "m"),
    ("Лена", "Лены", "Лене", "f"), ("Миша", "Миши", "Мише", "m"), ("Таня", "Тани", "Тане", "f"),
]
THINGS = [  # (one, few, many)
    ("яблоко", "яблока", "яблок"), ("груша", "груши", "груш"), ("конфета", "конфеты", "конфет"),
    ("карандаш", "карандаша", "карандашей"), ("марка", "марки", "марок"), ("книга", "книги", "книг"),
    ("тетрадь", "тетради", "тетрадей"), ("орех", "ореха", "орехов"), ("гриб", "гриба", "грибов"),
    ("шарик", "шарика", "шариков"), ("открытка", "открытки", "открыток"), ("значок", "значка", "значков"),
]


def past(verb_m, gender):
    """«купил» → «купила» for a girl."""
    if gender == "m":
        return verb_m
    return verb_m[:-1] + "ла" if verb_m.endswith("ёл") else verb_m + "а"


def count(n, forms):
    return f"{n} {plural(n, forms)}"


@topic(1, MATH, "m1_add", "сложение в пределах 20")
def _(r):
    a, b = r.randint(1, 10), r.randint(1, 10)
    if a + b > 20:
        return None
    q = r.choice([f"Сколько будет {a} + {b}?", f"Найди сумму чисел {a} и {b}.", f"Вычисли: {a} + {b}.",
                  f"Сложи {a} и {b}."])
    ch = Chain(q)
    if a < 10 and a + b > 10:
        ch.think(f"Складываю с переходом через десяток: до 10 к числу {a} не хватает {10 - a}, "
                 f"а от {b} после этого остаётся {b - (10 - a)}.")
    else:
        ch.think(f"Нужно найти сумму: складываю {a} и {b}.")
    s = ch.calc(f"{a} + {b}")
    return ch.answer(f"{a} + {b} = {s}", "number", str(s)), q


@topic(1, MATH, "m1_sub", "вычитание в пределах 20")
def _(r):
    a = r.randint(2, 20)
    b = r.randint(1, a)
    q = r.choice([f"Сколько будет {a} - {b}?", f"Найди разность чисел {a} и {b}.", f"Вычти {b} из {a}.",
                  f"Вычисли: {a} - {b}."])
    ch = Chain(q)
    ch.think(f"Из {a} вычитаю {b}: разность = уменьшаемое − вычитаемое.")
    d = ch.calc(f"{a} - {b}")
    return ch.answer(f"{a} - {b} = {d}", "number", str(d)), q


@topic(1, MATH, "m1_compare", "сравнение чисел")
def _(r):
    a, b = r.sample(range(0, 21), 2)
    if r.random() < 0.5:
        q = f"Какое число больше: {a} или {b}?"
        big = max(a, b)
        ch = Chain(q).think(
            f"Сравниваю числа: при счёте {min(a, b)} идёт раньше, чем {big}, значит {big} больше.")
        return ch.answer(f"{big} больше, чем {min(a, b)}.", "number", str(big)), q
    big, small = max(a, b), min(a, b)
    q = f"На сколько {big} больше, чем {small}?"
    ch = Chain(q).think("Чтобы узнать, на сколько одно число больше другого, из большего вычитаю меньшее.")
    d = ch.calc(f"{big} - {small}")
    return ch.answer(f"На {d}.", "number", str(d)), q


@topic(1, MATH, "m1_story", "задачи «было — стало»")
def _(r):
    name, gen, _, g = r.choice(NAMES)
    th = r.choice(THINGS)
    a = r.randint(3, 15)
    kind = r.choice(["gave", "got", "birds_in", "birds_out"])
    if kind == "gave":
        b = r.randint(2, a - 1)
        q = f"У {gen} было {count(a, th)}. {name} {past('отдал', g)} {count(b, th)} другу. Сколько {th[2]} осталось у {gen}?"
        ch = Chain(q).think(f"Было {a}, {b} {past('отдал', g)} — стало меньше, значит вычитаю.")
        v = ch.calc(f"{a} - {b}")
    elif kind == "got":
        b = r.randint(2, 20 - a) if a < 18 else 2
        q = f"У {gen} было {count(a, th)}. Мама дала ещё {b}. Сколько {th[2]} стало у {gen}?"
        ch = Chain(q).think(f"Было {a}, дали ещё {b} — стало больше, значит складываю.")
        v = ch.calc(f"{a} + {b}")
    elif kind == "birds_in":
        b = r.randint(2, 20 - a) if a < 18 else 2
        th = ("птица", "птицы", "птиц")
        q = f"На ветке было {count(a, th)}. Прилетели ещё {b}. Сколько птиц стало на ветке?"
        ch = Chain(q).think(f"Птиц было {a}, прилетели ещё {b} — их стало больше, складываю.")
        v = ch.calc(f"{a} + {b}")
    else:
        b = r.randint(2, a - 1)
        th = ("птица", "птицы", "птиц")
        q = f"На ветке было {count(a, th)}. {b} улетели. Сколько птиц осталось?"
        ch = Chain(q).think(f"Птиц было {a}, {b} улетели — их стало меньше, вычитаю.")
        v = ch.calc(f"{a} - {b}")
    return ch.answer(f"Ответ: {count(v, th)}.", "number", str(v)), q


@topic(1, MATH, "m1_more_less", "на сколько больше / меньше")
def _(r):
    (n1, g1, _, _), (n2, g2, _, _) = r.sample(NAMES, 2)
    th = r.choice(THINGS)
    a, b = r.randint(3, 15), r.randint(2, 6)
    kind = r.choice(["more", "less", "more_inv", "less_inv"])
    if kind == "more":
        q = f"У {g1} {count(a, th)}, а у {g2} на {b} больше. Сколько {th[2]} у {g2}?"
        ch = Chain(q).think(f"«На {b} больше» — это столько же, сколько у {g1}, и ещё {b}: складываю.")
        v = ch.calc(f"{a} + {b}")
    elif kind == "less":
        if b >= a:
            return None
        q = f"У {g1} {count(a, th)}, а у {g2} на {b} меньше. Сколько {th[2]} у {g2}?"
        ch = Chain(q).think(f"«На {b} меньше» — это столько же, сколько у {g1}, но без {b}: вычитаю.")
        v = ch.calc(f"{a} - {b}")
    elif kind == "more_inv":
        if b >= a:
            return None
        q = f"У {g1} {count(a, th)}, это на {b} больше, чем у {g2}. Сколько {th[2]} у {g2}?"
        ch = Chain(q).think(f"У {g1} на {b} больше, значит у {g2} на {b} меньше: вычитаю, а не складываю.")
        v = ch.calc(f"{a} - {b}")
    else:
        q = f"У {g1} {count(a, th)}, это на {b} меньше, чем у {g2}. Сколько {th[2]} у {g2}?"
        ch = Chain(q).think(f"У {g1} на {b} меньше, значит у {g2} на {b} больше: складываю.")
        v = ch.calc(f"{a} + {b}")
    return ch.answer(f"У {g2} {count(v, th)}.", "number", str(v)), q


@topic(2, MATH, "m2_add_sub", "сложение и вычитание в пределах 100")
def _(r):
    a, b = r.randint(11, 89), r.randint(11, 89)
    if r.random() < 0.5 and a + b <= 100:
        q = r.choice([f"Вычисли: {a} + {b}.", f"Сколько будет {a} + {b}?"])
        ch = Chain(q).think(
            f"Складываю по разрядам: десятки {a // 10 * 10} и {b // 10 * 10}, единицы {a % 10} и {b % 10}"
            + (" — единиц больше десяти, будет переход через разряд." if a % 10 + b % 10 >= 10 else "."))
        v = ch.calc(f"{a} + {b}")
        return ch.answer(f"{a} + {b} = {v}", "number", str(v)), q
    a, b = max(a, b), min(a, b)
    q = r.choice([f"Вычисли: {a} - {b}.", f"Сколько будет {a} - {b}?"])
    ch = Chain(q).think(
        f"Вычитаю по разрядам: из {a} вычитаю {b // 10 * 10}, потом {b % 10}"
        + (" — единиц не хватает, занимаю десяток." if a % 10 < b % 10 else "."))
    v = ch.calc(f"{a} - {b}")
    return ch.answer(f"{a} - {b} = {v}", "number", str(v)), q


BOXES = [  # where: prepositional plural, genitive plural, accusative (one, few, many), feminine
    ("коробках", "коробок", ("коробку", "коробки", "коробок"), True),
    ("пакетах", "пакетов", ("пакет", "пакета", "пакетов"), False),
    ("вазах", "ваз", ("вазу", "вазы", "ваз"), True),
]


@topic(2, MATH, "m2_mult_meaning", "смысл умножения и деления")
def _(r):
    th = r.choice(THINGS)
    n, k = r.randint(2, 9), r.randint(2, 9)
    prep, gen, acc, fem = r.choice(BOXES)
    if r.random() < 0.5:
        q = f"В {k} {prep} по {count(n, th)}. Сколько всего {th[2]}?"
        ch = Chain(q).think(f"В каждой из {k} {gen} по {n} — это {k} одинаковых слагаемых по {n}, "
                            f"значит умножаю {n} на {k}.")
        v = ch.calc(f"{n} * {k}")
        return ch.answer(f"Всего {count(v, th)}.", "number", str(v)), q
    t = n * k
    q = (f"{count(t, th)} разложили поровну в {k} {plural(k, acc)}. "
         f"Сколько {th[2]} в {'каждой' if fem else 'каждом'}?")
    ch = Chain(q).think(f"Разложили поровну — значит делю {t} на {k} равных частей.")
    v = ch.calc(f"{t} / {k}")
    return ch.answer(f"По {count(v, th)}.", "number", str(v)), q


UNIT_PAIRS = [  # (big, small, ratio, grade-appropriate range of the big unit)
    ("дм", "см", 10, 9), ("м", "см", 100, 9), ("м", "дм", 10, 9), ("км", "м", 1000, 20), ("кг", "г", 1000, 20),
    ("т", "кг", 1000, 20), ("ц", "кг", 100, 20), ("см", "мм", 10, 20), ("руб.", "коп.", 100, 50),
]


@topic(2, MATH, "m2_units", "единицы величин")
def _(r):
    big, small, k, top = r.choice(UNIT_PAIRS)
    a = r.randint(1, top)
    kind = r.choice(["to_small", "mixed", "to_big"])
    if kind == "to_small":
        q = f"Сколько {small} в {a} {big}?"
        ch = Chain(q).think(f"В 1 {big} {k} {small}, значит умножаю {a} на {k}.")
        v = ch.calc(f"{a} * {k}")
        return ch.answer(f"{a} {big} = {v} {small}", "number", str(v)), q
    if kind == "mixed":
        b = r.randint(1, k - 1)
        q = f"Сколько {small} в {a} {big} {b} {small}?"
        ch = Chain(q).think(f"В 1 {big} {k} {small}: {a} {big} — это {a} раз по {k} {small}, и ещё {b} {small}.")
        v = ch.calc(f"{a} * {k} + {b}")
        return ch.answer(f"{a} {big} {b} {small} = {v} {small}", "number", str(v)), q
    n = a * k
    q = f"Сколько {big} в {n} {small}?"
    ch = Chain(q).think(f"В 1 {big} {k} {small}, значит делю {n} на {k}.")
    v = ch.calc(f"{n} / {k}")
    return ch.answer(f"{n} {small} = {v} {big}", "number", str(v)), q


@topic(2, MATH, "m2_two_step", "задачи в два действия")
def _(r):
    a, b = r.randint(8, 60), r.randint(2, 15)
    kind = r.choice(["class", "shelves"])
    kids = ("ребёнок", "ребёнка", "детей")
    if kind == "class":
        more = r.random() < 0.5
        q = f"В классе {a} мальчиков, а девочек на {b} {'больше' if more else 'меньше'}. Сколько всего детей в классе?"
        ch = Chain(q).think(f"Сначала узнаю, сколько девочек: на {b} {'больше' if more else 'меньше'}, "
                            f"значит {'складываю' if more else 'вычитаю'}.")
        girls = ch.calc(f"{a} {'+' if more else '-'} {b}")
        ch.think("Теперь всех детей: мальчики плюс девочки.")
        v = ch.calc(f"{a} + {girls}")
        return ch.answer(f"В классе {count(v, kids)}.", "number", str(v)), q
    books = ("книга", "книги", "книг")
    q = (f"На первой полке {count(a, books)}, на второй — на {b} больше. "
         f"Сколько книг на двух полках?")
    ch = Chain(q).think(f"Сначала найду, сколько книг на второй полке: на {b} больше, складываю.")
    second = ch.calc(f"{a} + {b}")
    ch.think("Теперь складываю книги на обеих полках.")
    v = ch.calc(f"{a} + {second}")
    return ch.answer(f"На двух полках {count(v, books)}.", "number", str(v)), q


@topic(2, MATH, "m2_perimeter", "периметр")
def _(r):
    if r.random() < 0.6:
        a, b = r.sample(range(2, 30), 2)
        q = f"Найди периметр прямоугольника со сторонами {a} см и {b} см."
        ch = Chain(q).think("Периметр — сумма длин всех сторон. У прямоугольника две пары равных сторон, "
                            "поэтому периметр = (длина + ширина) · 2.")
        v = ch.calc(f"({a} + {b}) * 2")
    else:
        a = r.randint(2, 30)
        q = f"Найди периметр квадрата со стороной {a} см."
        ch = Chain(q).think("У квадрата 4 равные стороны, поэтому периметр = сторона · 4.")
        v = ch.calc(f"{a} * 4")
    return ch.answer(f"Периметр — {v} см.", "number", str(v)), q


@topic(3, MATH, "m3_table", "табличное умножение и деление")
def _(r):
    a, b = r.randint(2, 9), r.randint(2, 9)
    if r.random() < 0.5:
        q = r.choice([f"Сколько будет {a} · {b}?", f"Вычисли: {a} · {b}.", f"Умножь {a} на {b}."])
        ch = Chain(q).think(f"Умножение — это сложение {b} одинаковых слагаемых по {a}.")
        v = ch.calc(f"{a} * {b}")
        return ch.answer(f"{a} · {b} = {v}", "number", str(v)), q
    c = a * b
    q = r.choice([f"Сколько будет {c} : {b}?", f"Вычисли: {c} : {b}.", f"Раздели {c} на {b}."])
    ch = Chain(q).think(f"Деление обратно умножению: ищу число, которое при умножении на {b} даёт {c}.")
    v = ch.calc(f"{c} / {b}")
    return ch.answer(f"{c} : {b} = {v}", "number", str(v)), q


@topic(3, MATH, "m3_remainder", "деление с остатком")
def _(r):
    d = r.randint(2, 12)
    n = r.randint(d + 1, 12 * d + d - 1)
    if n % d == 0:
        return None
    qt = n // d
    q = f"Раздели {n} на {d} с остатком."
    ch = Chain(q).think(f"Ищу самое большое число, не больше {n}, которое делится на {d}: это {qt * d}, "
                        f"частное {qt}. Остаток — сколько осталось до {n}; он должен быть меньше {d}.")
    rem = ch.calc(f"{n} - {qt} * {d}")
    return ch.answer(f"{n} : {d} = {qt} (ост. {rem})", "text", f"{qt} (ост. {rem})"), q


@topic(3, MATH, "m3_order", "порядок действий")
def _(r):
    a, b, c, d = (r.randint(2, 12) for _ in range(4))
    forms = [
        (f"{a} + {b} * {c}", f"{a} + {b} · {c}", "Сначала умножение, потом сложение."),
        (f"({a} + {b}) * {c}", f"({a} + {b}) · {c}", "Сначала действие в скобках, потом умножение."),
        (f"{a * c} - {b * c} / {c}", f"{a * c} - {b * c} : {c}", "Сначала деление, потом вычитание."),
        (f"{a} * {b} - {c} * {d}", f"{a} · {b} - {c} · {d}", "Сначала оба умножения, потом вычитание."),
        (f"({a * b} - {b}) / {b}", f"({a * b} - {b}) : {b}", "Сначала действие в скобках, потом деление."),
    ]
    expr, shown, why = r.choice(forms)
    if calc(expr)[1] < 0:
        return None
    q = f"Вычисли: {shown}."
    ch = Chain(q).think(why)
    v = ch.calc(expr)
    return ch.answer(f"{shown} = {v}", "number", str(v)), q


@topic(3, MATH, "m3_area", "площадь прямоугольника")
def _(r):
    a, b = r.randint(2, 20), r.randint(2, 20)
    if r.random() < 0.6:
        q = f"Длина прямоугольника {a} см, ширина {b} см. Найди его площадь."
        ch = Chain(q).think("Площадь прямоугольника = длина · ширина.")
        v = ch.calc(f"{a} * {b}")
        return ch.answer(f"Площадь — {v} кв. см.", "number", str(v)), q
    s = a * b
    q = f"Площадь прямоугольника {s} кв. см, длина {a} см. Найди ширину."
    ch = Chain(q).think("Площадь = длина · ширина, значит ширина = площадь : длина.")
    v = ch.calc(f"{s} / {a}")
    return ch.answer(f"Ширина — {v} см.", "number", str(v)), q


@topic(3, MATH, "m3_price", "цена, количество, стоимость")
def _(r):
    th = r.choice([("тетрадь", "тетради", "тетрадей"), ("ручка", "ручки", "ручек"), ("булочка", "булочки", "булочек"),
                   ("альбом", "альбома", "альбомов"), ("мяч", "мяча", "мячей")])
    p, k = r.randint(3, 60), r.randint(2, 9)
    kind = r.choice(["cost", "price", "qty"])
    if kind == "cost":
        q = f"{th[0].capitalize()} стоит {p} ₽. Сколько стоят {count(k, th)}?"
        ch = Chain(q).think("Стоимость = цена · количество.")
        v = ch.calc(f"{p} * {k}")
        return ch.answer(f"{count(k, th).capitalize()} стоят {v} ₽.", "number", str(v)), q
    s = p * k
    if kind == "price":
        q = f"За {count(k, th)} заплатили {s} ₽. Сколько стоит одна штука?"
        ch = Chain(q).think("Цена = стоимость : количество.")
        v = ch.calc(f"{s} / {k}")
        return ch.answer(f"Одна штука стоит {v} ₽.", "number", str(v)), q
    q = f"Сколько штук можно купить на {s} ₽, если одна штука стоит {p} ₽?"
    ch = Chain(q).think("Количество = стоимость : цена.")
    v = ch.calc(f"{s} / {p}")
    return ch.answer(f"Можно купить {v} шт.", "number", str(v)), q


PARTS = {2: ("половину", "половина"), 3: ("треть", "треть"), 4: ("четверть", "четверть")}


@topic(3, MATH, "m3_part", "доля числа")
def _(r):
    k = r.randint(2, 10)
    n = k * r.randint(2, 60)
    word = PARTS.get(k) if r.random() < 0.5 else None
    q = f"Найди {word[0]} числа {n}." if word else f"Найди 1/{k} от числа {n}."
    ch = Chain(q).think(f"Чтобы найти одну {k}-ю часть числа, делю его на {k}.")
    v = ch.calc(f"{n} / {k}")
    name = word[1].capitalize() if word else f"1/{k}"
    return ch.answer(f"{name} числа {n} — это {v}.", "number", str(v)), q


@topic(4, MATH, "m4_multidigit", "действия с многозначными числами")
def _(r):
    if r.random() < 0.5:
        a, b = r.randint(100, 9999), r.randint(11, 99)
        q = f"Вычисли: {a} · {b}."
        ch = Chain(q).think("Умножаю многозначное число на двузначное: это удобно сделать калькулятором.")
        v = ch.calc(f"{a} * {b}")
        return ch.answer(f"{a} · {b} = {v}", "number", str(v)), q
    b, v0 = r.randint(11, 99), r.randint(12, 999)
    a = b * v0
    q = f"Вычисли: {a} : {b}."
    ch = Chain(q).think("Делю многозначное число на двузначное.")
    v = ch.calc(f"{a} / {b}")
    return ch.answer(f"{a} : {b} = {v}", "number", str(v)), q


@topic(4, MATH, "m4_motion", "скорость, время, расстояние")
def _(r):
    who = r.choice([("поезд", "ехал"), ("автомобиль", "ехал"), ("велосипедист", "ехал"), ("пешеход", "шёл"),
                    ("теплоход", "плыл")])
    v, t = r.randint(4, 90), r.randint(2, 9)
    s = v * t
    kind = r.choice(["s", "v", "t"])
    if kind == "s":
        q = f"{who[0].capitalize()} {who[1]} {t} ч со скоростью {v} км/ч. Какое расстояние он проделал?"
        ch = Chain(q).think("Расстояние = скорость · время.")
        x = ch.calc(f"{v} * {t}")
        return ch.answer(f"Расстояние — {x} км.", "number", str(x)), q
    if kind == "v":
        q = f"{who[0].capitalize()} за {t} ч проделал {s} км. С какой скоростью он двигался?"
        ch = Chain(q).think("Скорость = расстояние : время.")
        x = ch.calc(f"{s} / {t}")
        return ch.answer(f"Скорость — {x} км/ч.", "number", str(x)), q
    q = f"За сколько часов {who[0]} проделает {s} км со скоростью {v} км/ч?"
    ch = Chain(q).think("Время = расстояние : скорость.")
    x = ch.calc(f"{s} / {v}")
    return ch.answer(f"За {x} ч.", "number", str(x)), q


@topic(4, MATH, "m4_meeting", "движение навстречу и вдогонку")
def _(r):
    v1, v2, t = r.randint(3, 20), r.randint(3, 20), r.randint(2, 6)
    kind = r.choice(["towards", "apart", "chase"])
    if kind == "towards":
        s = (v1 + v2) * t
        q = (f"Два велосипедиста выехали одновременно навстречу друг другу из посёлков, расстояние между "
             f"которыми {s} км. Скорость одного {v1} км/ч, другого {v2} км/ч. Через сколько часов они встретятся?")
        ch = Chain(q).think(f"Они едут навстречу, поэтому за час расстояние между ними сокращается на {v1} + {v2} км "
                            "— это скорость сближения.")
        vs = ch.calc(f"{v1} + {v2}")
        ch.think("Время встречи = расстояние : скорость сближения.")
        x = ch.calc(f"{s} / {vs}")
        return ch.answer(f"Они встретятся через {x} ч.", "number", str(x)), q
    if kind == "apart":
        q = (f"Два пешехода вышли из одного места в противоположных направлениях. Один идёт со скоростью "
             f"{v1} км/ч, другой — {v2} км/ч. Какое расстояние будет между ними через {t} ч?")
        ch = Chain(q).think(f"Они удаляются друг от друга: за час расстояние растёт на {v1} + {v2} км.")
        vs = ch.calc(f"{v1} + {v2}")
        ch.think(f"За {t} ч: скорость удаления · время.")
        x = ch.calc(f"{vs} * {t}")
        return ch.answer(f"Через {t} ч между ними будет {x} км.", "number", str(x)), q
    if v1 == v2:
        return None
    fast, slow = max(v1, v2), min(v1, v2)
    d = (fast - slow) * t
    q = (f"Велосипедист едет со скоростью {slow} км/ч. Следом за ним в {d} км позади выехал мотоциклист со "
         f"скоростью {fast} км/ч. Через сколько часов мотоциклист догонит велосипедиста?")
    ch = Chain(q).think(f"Они едут в одну сторону, поэтому за час расстояние сокращается на {fast} − {slow} км.")
    vs = ch.calc(f"{fast} - {slow}")
    ch.think("Время = начальное расстояние : скорость сближения.")
    x = ch.calc(f"{d} / {vs}")
    return ch.answer(f"Через {x} ч.", "number", str(x)), q


@topic(4, MATH, "m4_time", "единицы времени")
def _(r):
    kind = r.choice(["h_min", "min_s", "day_h", "min_h"])
    if kind == "h_min":
        h, m = r.randint(1, 9), r.randint(1, 59)
        q = f"Сколько минут в {h} ч {m} мин?"
        ch = Chain(q).think(f"В 1 ч 60 мин: {h} ч — это {h} раз по 60 мин, и ещё {m} мин.")
        v = ch.calc(f"{h} * 60 + {m}")
        return ch.answer(f"{h} ч {m} мин = {v} мин", "number", str(v)), q
    if kind == "min_s":
        m = r.randint(2, 30)
        q = f"Сколько секунд в {m} мин?"
        ch = Chain(q).think("В 1 мин 60 с.")
        v = ch.calc(f"{m} * 60")
        return ch.answer(f"{m} мин = {v} с", "number", str(v)), q
    if kind == "day_h":
        d = r.randint(2, 14)
        q = f"Сколько часов в {d} сутках?"
        ch = Chain(q).think("В одних сутках 24 ч.")
        v = ch.calc(f"{d} * 24")
        return ch.answer(f"В {d} сутках {v} ч.", "number", str(v)), q
    h = r.randint(2, 12)
    m = h * 60
    q = f"Сколько часов в {m} мин?"
    ch = Chain(q).think("В 1 ч 60 мин, значит делю на 60.")
    v = ch.calc(f"{m} / 60")
    return ch.answer(f"{m} мин = {v} ч", "number", str(v)), q


@topic(4, MATH, "m4_fraction_of", "часть от целого")
def _(r):
    qd = r.randint(3, 10)
    p = r.randint(2, qd - 1)
    n = qd * r.randint(5, 40)
    q = f"В книге {n} страниц. Маша прочитала {p}/{qd} книги. Сколько страниц она прочитала?"
    ch = Chain(q).think(f"Чтобы найти {p}/{qd} числа, делю его на знаменатель {qd} и умножаю на числитель {p}.")
    v = ch.calc(f"{n} / {qd} * {p}")
    pages = ("страницу", "страницы", "страниц")
    return ch.answer(f"Маша прочитала {v} {plural(v, pages)}.", "number", str(v)), q


EQ_RULES = {
    "x+a": "Неизвестное слагаемое = сумма − известное слагаемое.",
    "x-a": "Неизвестное уменьшаемое = разность + вычитаемое.",
    "a-x": "Неизвестное вычитаемое = уменьшаемое − разность.",
    "x*a": "Неизвестный множитель = произведение : известный множитель.",
    "x/a": "Неизвестное делимое = частное · делитель.",
    "a/x": "Неизвестный делитель = делимое : частное.",
}


@topic(4, MATH, "m4_equation", "простые уравнения")
def _(r):
    kind = r.choice(list(EQ_RULES))
    x, a = r.randint(2, 99), r.randint(2, 99)
    if kind == "x+a":
        b = x + a
        eq, sol, chk = f"x + {a} = {b}", f"{b} - {a}", "x + {a}"
    elif kind == "x-a":
        b = x - a
        if b <= 0:
            return None
        eq, sol, chk = f"x - {a} = {b}", f"{b} + {a}", "x - {a}"
    elif kind == "a-x":
        a = x + r.randint(1, 50)
        b = a - x
        eq, sol, chk = f"{a} - x = {b}", f"{a} - {b}", "{a} - x"
    elif kind == "x*a":
        a = r.randint(2, 12)
        b = x * a
        eq, sol, chk = f"x · {a} = {b}", f"{b} / {a}", "x * {a}"
    elif kind == "x/a":
        a = r.randint(2, 12)
        b = x
        x = b * a
        eq, sol, chk = f"x : {a} = {b}", f"{b} * {a}", "x / {a}"
    else:
        b = r.randint(2, 12)
        a = x * b
        eq, sol, chk = f"{a} : x = {b}", f"{a} / {b}", "{a} / x"
    q = f"Реши уравнение: {eq}."
    ch = Chain(q).think(EQ_RULES[kind])
    v = ch.calc(sol)
    if r.random() < 0.5:
        ch.think("Проверю: подставлю найденное число в уравнение.")
        ch.calc(chk.format(a=a).replace("x", str(v)))
    return ch.answer(f"x = {v}", "number", str(v)), q


def dec(r, lo, hi, places):
    return Fraction(r.randint(lo * 10**places, hi * 10**places), 10**places)


@topic(5, MATH, "m5_decimals", "десятичные дроби")
def _(r):
    a, b = dec(r, 0, 50, r.randint(1, 2)), dec(r, 0, 50, r.randint(1, 2))
    op = r.choice(["+", "-", "*"])
    if op == "-" and a < b:
        a, b = b, a
    if op == "*":
        b = dec(r, 0, 9, 1)
    sym = {"+": "+", "-": "-", "*": "·"}[op]
    q = f"Вычисли: {ru(a)} {sym} {ru(b)}."
    why = {"+": "Складываю десятичные дроби поразрядно: запятая под запятой.",
           "-": "Вычитаю десятичные дроби поразрядно: запятая под запятой.",
           "*": "Умножаю, не обращая внимания на запятые, потом отделяю столько знаков, сколько их в обоих множителях."}[op]
    ch = Chain(q).think(why)
    v = ch.calc(f"{show(a)} {op} {show(b)}")
    return ch.answer(f"{ru(a)} {sym} {ru(b)} = {ru(v)}", "number", show(v)), q


def reduce_steps(ch, num, den):
    """Shortens num/den with CALC steps; returns the reduced pair."""
    from math import gcd
    g = gcd(num, den)
    if g > 1:
        ch.think(f"Сокращаю дробь {num}/{den}: числитель и знаменатель делятся на {g}.")
        num, den = int(ch.calc(f"{num} / {g}")), int(ch.calc(f"{den} / {g}"))
    return num, den


@topic(5, MATH, "m5_fractions", "обыкновенные дроби с одинаковыми знаменателями")
def _(r):
    d = r.randint(3, 20)
    a, b = r.randint(1, d - 1), r.randint(1, d - 1)
    add = r.random() < 0.5
    if add and a + b > d:
        return None
    if not add and a <= b:
        a, b = b, a
    if not add and a == b:
        return None
    q = f"Вычисли: {a}/{d} {'+' if add else '-'} {b}/{d}."
    ch = Chain(q).think(f"Знаменатели одинаковые: {'складываю' if add else 'вычитаю'} числители, знаменатель оставляю.")
    n = int(ch.calc(f"{a} {'+' if add else '-'} {b}"))
    if n == d:
        return ch.answer(f"{a}/{d} + {b}/{d} = {d}/{d} = 1", "text", "1"), q
    n2, d2 = reduce_steps(ch, n, d)
    res = f"{n2}/{d2}"
    return ch.answer(f"{a}/{d} {'+' if add else '-'} {b}/{d} = {res}", "text", res), q


@topic(5, MATH, "m5_percent", "проценты")
def _(r):
    p = r.choice([1, 2, 5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    n = r.randint(1, 60) * 20
    q = r.choice([f"Найди {p}% от {n}.", f"Сколько составляет {p}% от числа {n}?"])
    ch = Chain(q).think(f"1% — это сотая часть числа, значит {p}% от {n} = {n} · {p} : 100.")
    v = ch.calc(f"{n} * {p} / 100")
    return ch.answer(f"{p}% от {n} — это {ru(v)}.", "number", show(v)), q


@topic(5, MATH, "m5_volume", "объём прямоугольного параллелепипеда")
def _(r):
    a, b, c = r.randint(2, 15), r.randint(2, 15), r.randint(2, 15)
    q = f"Длина коробки {a} см, ширина {b} см, высота {c} см. Найди её объём."
    ch = Chain(q).think("Объём прямоугольного параллелепипеда = длина · ширина · высота.")
    v = ch.calc(f"{a} * {b} * {c}")
    return ch.answer(f"Объём — {v} куб. см.", "number", str(v)), q


@topic(5, MATH, "m5_mean", "среднее арифметическое")
def _(r):
    k = r.randint(3, 5)
    xs = [r.randint(1, 50) for _ in range(k)]
    q = f"Найди среднее арифметическое чисел {', '.join(map(str, xs[:-1]))} и {xs[-1]}."
    ch = Chain(q).think(f"Среднее арифметическое = сумма чисел : их количество ({k}).")
    v = ch.calc(f"({' + '.join(map(str, xs))}) / {k}")
    return ch.answer(f"Среднее арифметическое — {ru(v)}.", "number", show(v)), q


@topic(5, MATH, "m5_by_part", "нахождение числа по его части")
def _(r):
    qd = r.randint(2, 10)
    p = r.randint(1, qd - 1)
    whole = qd * r.randint(2, 30)
    part = whole // qd * p
    q = f"Ученик прочитал {part} {plural(part, ('страницу', 'страницы', 'страниц'))}, это {p}/{qd} книги. Сколько страниц в книге?"
    ch = Chain(q).think(f"Известна часть числа. Чтобы найти всё число, делю часть на числитель {p} "
                        f"и умножаю на знаменатель {qd}.")
    v = ch.calc(f"{part} / {p} * {qd}")
    return ch.answer(f"В книге {v} {plural(v, ('страница', 'страницы', 'страниц'))}.", "number", str(v)), q


@topic(5, MATH, "m5_rounding", "округление")
def _(r):
    x = dec(r, 0, 99, 3)
    place, digits = r.choice([("десятых", 1), ("сотых", 2), ("единиц", 0)])
    scaled = x * 10**digits
    down = Fraction(int(scaled), 10**digits)
    nxt = int((scaled - int(scaled)) * 10)
    rounded = down + (Fraction(1, 10**digits) if nxt >= 5 else 0)
    q = f"Округли {ru(x)} до {place}."
    ch = Chain(q).think(f"Смотрю на цифру следующего разряда: это {nxt}. "
                        + ("Она 5 или больше — последнюю оставленную цифру увеличиваю на 1." if nxt >= 5
                           else "Она меньше 5 — оставленные цифры не меняю."))
    return ch.answer(f"{ru(x)} ≈ {ru(rounded)}", "number", show(rounded)), q


def factors(n):
    out, p = [], 2
    while n > 1:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1
    return out


@topic(6, MATH, "m6_gcd_lcm", "НОД и НОК")
def _(r):
    from math import gcd
    g = r.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15])
    a, b = g * r.randint(2, 12), g * r.randint(2, 12)
    if a == b:
        return None
    fa, fb = factors(a), factors(b)
    if r.random() < 0.5:
        common, rest = [], list(fb)
        for f in fa:
            if f in rest:
                common.append(f)
                rest.remove(f)
        q = f"Найди наибольший общий делитель чисел {a} и {b}."
        ch = Chain(q).think(f"Раскладываю на простые множители: {a} = {' · '.join(map(str, fa))}, "
                            f"{b} = {' · '.join(map(str, fb))}. НОД — произведение общих множителей: "
                            f"{', '.join(map(str, common))}.")
        v = ch.calc(" * ".join(map(str, common))) if len(common) > 1 else Fraction(common[0])
        if len(common) == 1:
            ch.think(f"Общий множитель один — {common[0]}.")
        assert v == gcd(a, b)
        return ch.answer(f"НОД({a}, {b}) = {v}", "number", str(v)), q
    missing, rest = [], list(fa)
    for f in fb:
        if f in rest:
            rest.remove(f)
        else:
            missing.append(f)
    big = fa
    q = f"Найди наименьшее общее кратное чисел {a} и {b}."
    ch = Chain(q).think(f"Раскладываю: {a} = {' · '.join(map(str, fa))}, {b} = {' · '.join(map(str, fb))}. "
                        f"НОК — множители числа {a} и недостающие множители числа {b}"
                        + (f": {', '.join(map(str, missing))}." if missing else " (недостающих нет)."))
    v = ch.calc(" * ".join(map(str, big + missing)))
    assert v == a * b // gcd(a, b)
    return ch.answer(f"НОК({a}, {b}) = {v}", "number", str(v)), q


@topic(6, MATH, "m6_divisibility", "признаки делимости")
def _(r):
    n = r.randint(100, 99999)
    k = r.choice([2, 3, 5, 9, 10])
    q = f"Делится ли число {n} на {k}?"
    ch = Chain(q)
    if k in (3, 9):
        ch.think(f"Признак делимости на {k}: число делится на {k}, если сумма его цифр делится на {k}.")
        s = int(ch.calc(" + ".join(str(n))))
        yes = s % k == 0
        ch.think(f"Сумма цифр {s} {'делится' if yes else 'не делится'} на {k}.")
    else:
        last = n % 10
        rule = {2: "последняя цифра чётная", 5: "последняя цифра 0 или 5", 10: "последняя цифра 0"}[k]
        yes = n % k == 0
        ch.think(f"Признак делимости на {k}: {rule}. Последняя цифра — {last}.")
    return ch.answer(f"{'Да' if yes else 'Нет'}, {n} {'делится' if yes else 'не делится'} на {k}.",
                     "text", "да" if yes else "нет"), q


@topic(6, MATH, "m6_proportion", "прямая пропорциональность")
def _(r):
    a, b = r.randint(2, 10), r.randint(1, 9)
    k = r.randint(2, 12)
    c = a * k
    q = f"Из {a} кг яблок получают {b} кг сушёных. Сколько сушёных яблок получится из {c} кг свежих?"
    ch = Chain(q).think(f"Величины прямо пропорциональны: во сколько раз больше свежих яблок, во столько "
                        f"раз больше сушёных. Составляю пропорцию {a} : {b} = {c} : x, x = {c} · {b} : {a}.")
    v = ch.calc(f"{c} * {b} / {a}")
    return ch.answer(f"Получится {ru(v)} кг.", "number", show(v)), q


@topic(6, MATH, "m6_negative", "положительные и отрицательные числа")
def _(r):
    xs = [r.randint(-30, 30) for _ in range(r.randint(2, 4))]
    if all(x >= 0 for x in xs):
        xs[0] = -xs[0] - 1
    expr = str(xs[0]) + "".join(f" + {x}" if x >= 0 else f" - {-x}" for x in xs[1:])
    q = f"Вычисли: {expr}."
    ch = Chain(q).think("Складываю числа с разными знаками: из большего модуля вычитаю меньший и ставлю знак "
                        "числа с большим модулем; вычитание — это сложение с противоположным числом.")
    v = ch.calc(expr)
    return ch.answer(f"{expr} = {v}", "number", str(v)), q


@topic(6, MATH, "m6_percent_change", "увеличение и уменьшение на процент")
def _(r):
    p = r.choice([5, 10, 15, 20, 25, 30, 40, 50])
    n = r.randint(2, 50) * 100
    up = r.random() < 0.5
    q = f"Цена товара {n} ₽ {'повысилась' if up else 'снизилась'} на {p}%. Какой стала цена?"
    ch = Chain(q).think(f"Сначала найду {p}% от {n}.")
    d = ch.calc(f"{n} * {p} / 100")
    ch.think(f"Цена {'выросла' if up else 'уменьшилась'} на {ru(d)} ₽ — {'прибавляю' if up else 'вычитаю'}.")
    v = ch.calc(f"{n} {'+' if up else '-'} {show(d)}")
    return ch.answer(f"Новая цена — {ru(v)} ₽.", "number", show(v)), q


@topic(6, MATH, "m6_ratio", "деление в данном отношении")
def _(r):
    a, b = r.randint(1, 9), r.randint(1, 9)
    part = r.randint(2, 20)
    n = (a + b) * part
    q = f"Раздели число {n} в отношении {a} : {b}."
    ch = Chain(q).think(f"Всего частей {a} + {b}.")
    parts = ch.calc(f"{a} + {b}")
    ch.think(f"Одна часть = {n} : {parts}.")
    one = ch.calc(f"{n} / {parts}")
    x, y = ch.calc(f"{show(one)} * {a}"), ch.calc(f"{show(one)} * {b}")
    return ch.answer(f"{x} и {y}", "text", f"{x} и {y}"), q


@topic(6, MATH, "m6_circle", "длина окружности и площадь круга")
def _(r):
    rad = r.randint(1, 60) if r.random() < 0.7 else dec(r, 1, 30, 1)
    kind = r.choice(["c", "c_d", "s"])
    if kind == "c":
        q = f"Найди длину окружности радиусом {ru(rad)} см (π ≈ 3,14)."
        ch = Chain(q).think("Длина окружности C = 2πr.")
        v = ch.calc(f"2 * 3.14 * {show(Fraction(rad))}")
        return ch.answer(f"C ≈ {ru(v)} см", "number", show(v)), q
    if kind == "c_d":
        d = 2 * Fraction(rad)
        q = f"Найди длину окружности диаметром {ru(d)} см (π ≈ 3,14)."
        ch = Chain(q).think("Длина окружности C = πd.")
        v = ch.calc(f"3.14 * {show(d)}")
        return ch.answer(f"C ≈ {ru(v)} см", "number", show(v)), q
    rad = Fraction(rad)
    q = f"Найди площадь круга радиусом {ru(rad)} см (π ≈ 3,14)."
    ch = Chain(q).think("Площадь круга S = πr², то есть π · r · r.")
    v = ch.calc(f"3.14 * {show(rad)} * {show(rad)}")
    return ch.answer(f"S ≈ {ru(v)} кв. см", "number", show(v)), q


@topic(6, MATH, "m6_fraction_mult", "умножение и деление обыкновенных дробей")
def _(r):
    a, b, c, d = r.randint(1, 9), r.randint(2, 10), r.randint(1, 9), r.randint(2, 10)
    mult = r.random() < 0.5
    q = f"Вычисли: {a}/{b} {'·' if mult else ':'} {c}/{d}."
    ch = Chain(q)
    if mult:
        ch.think("Числитель умножаю на числитель, знаменатель — на знаменатель.")
        n, m = int(ch.calc(f"{a} * {c}")), int(ch.calc(f"{b} * {d}"))
    else:
        ch.think(f"Деление на дробь — это умножение на обратную дробь {d}/{c}.")
        n, m = int(ch.calc(f"{a} * {d}")), int(ch.calc(f"{b} * {c}"))
    n2, m2 = reduce_steps(ch, n, m)
    res = f"{n2}/{m2}" if m2 != 1 else str(n2)
    return ch.answer(f"{a}/{b} {'·' if mult else ':'} {c}/{d} = {res}", "text", res), q


@topic(7, MATH, "m7_linear", "линейные уравнения")
def _(r):
    a = r.choice([k for k in range(-9, 10) if k not in (0, 1)])
    x = r.randint(-15, 15)
    b = r.choice([k for k in range(-30, 31) if k != 0])
    c = a * x + b
    bs = f"+ {b}" if b >= 0 else f"- {-b}"
    q = f"Реши уравнение: {a}x {bs} = {c}."
    ch = Chain(q).think(f"Переношу {b if b >= 0 else '−' + str(-b)} в правую часть с противоположным знаком, "
                        f"затем делю обе части на коэффициент {a}: x = ({c} {'-' if b >= 0 else '+'} {abs(b)}) : {a}.")
    v = ch.calc(f"({c} {'-' if b >= 0 else '+'} {abs(b)}) / {a}")
    if r.random() < 0.4:
        ch.think("Проверка: подставляю x в левую часть.")
        ch.calc(f"{a} * {v} {'+' if b >= 0 else '-'} {abs(b)}" if v >= 0 else f"{a} * ({v}) {'+' if b >= 0 else '-'} {abs(b)}")
    return ch.answer(f"x = {v}", "number", str(v)), q


@topic(7, MATH, "m7_both_sides", "уравнения с неизвестным в обеих частях")
def _(r):
    a, c = r.randint(2, 12), r.randint(1, 11)
    if a == c:
        return None
    x = r.randint(-10, 15)
    b = r.randint(-20, 20)
    d = a * x + b - c * x
    s = lambda k: f"+ {k}" if k >= 0 else f"- {-k}"
    q = f"Реши уравнение: {a}x {s(b)} = {c}x {s(d)}."
    ch = Chain(q).think(f"Собираю слагаемые с x слева, числа справа: {a}x − {c}x = {d} − ({b}), "
                        f"затем делю на коэффициент при x.")
    v = ch.calc(f"({d} - ({b})) / ({a} - {c})")
    return ch.answer(f"x = {v}", "number", str(v)), q


@topic(7, MATH, "m7_function", "значение линейной функции")
def _(r):
    k, b, x = r.randint(-9, 9), r.randint(-20, 20), r.randint(-10, 10)
    if k == 0:
        return None
    bs = f"+ {b}" if b >= 0 else f"- {-b}"
    q = f"Функция задана формулой y = {k}x {bs}. Найди y при x = {x}."
    ch = Chain(q).think(f"Подставляю x = {x} в формулу.")
    xs = str(x) if x >= 0 else f"({x})"
    v = ch.calc(f"{k} * {xs} {'+' if b >= 0 else '-'} {abs(b)}")
    return ch.answer(f"y = {v}", "number", str(v)), q


@topic(7, MATH, "m7_power", "степень с натуральным показателем")
def _(r):
    a, n = r.randint(2, 12), r.randint(2, 4)
    kind = r.choice(["one", "neg", "sum"])
    if kind == "one":
        q = f"Вычисли {a}^{n}."
        ch = Chain(q).think(f"{a}^{n} — это произведение {n} множителей, каждый равен {a}.")
        v = ch.calc(" * ".join([str(a)] * n))
        return ch.answer(f"{a}^{n} = {v}", "number", str(v)), q
    if kind == "neg":
        q = f"Вычисли (−{a})^{n}."
        ch = Chain(q).think(f"Перемножаю {n} множителей −{a}: при {'чётном' if n % 2 == 0 else 'нечётном'} "
                            f"показателе результат {'положительный' if n % 2 == 0 else 'отрицательный'}.")
        v = ch.calc(" * ".join([f"(-{a})"] * n))
        return ch.answer(f"(−{a})^{n} = {v}", "number", str(v)), q
    b, m = r.randint(2, 9), r.randint(2, 3)
    q = f"Вычисли {a}^{n} + {b}^{m}."
    ch = Chain(q).think("Сначала возвожу в степень, потом складываю.")
    x = ch.calc(" * ".join([str(a)] * n))
    y = ch.calc(" * ".join([str(b)] * m))
    v = ch.calc(f"{x} + {y}")
    return ch.answer(f"{a}^{n} + {b}^{m} = {v}", "number", str(v)), q


@topic(7, MATH, "m7_system", "системы линейных уравнений")
def _(r):
    x, y = r.randint(-10, 20), r.randint(-10, 20)
    s, d = x + y, x - y
    q = f"Реши систему уравнений: x + y = {s}, x − y = {d}."
    ch = Chain(q).think(f"Складываю уравнения: y сокращается, получается 2x = {s} + ({d}).")
    vx = ch.calc(f"({s} + ({d})) / 2")
    ch.think(f"Подставляю x в первое уравнение: y = {s} − x.")
    vy = ch.calc(f"{s} - ({vx})")
    return ch.answer(f"x = {vx}, y = {vy}", "text", f"x = {vx}, y = {vy}"), q


@topic(7, MATH, "m7_word_equation", "задачи на составление уравнения")
def _(r):
    kind = r.choice(["times", "more", "sum_diff"])
    if kind == "times":
        k = r.randint(2, 9)
        x = r.randint(3, 60)
        total = x * (k + 1)
        times = plural(k, ("раз", "раза", "раз"))
        q = (f"В двух корзинах {count(total, THINGS[0])}, в первой в {k} {times} больше, чем во второй. "
             f"Сколько яблок в каждой корзине?")
        ch = Chain(q).think(f"Пусть во второй корзине x яблок, тогда в первой {k}x. Вместе x + {k}x = {total}, "
                            f"то есть {k + 1}x = {total}.")
        v = ch.calc(f"{total} / {k + 1}")
        ch.think(f"Во второй {v}, в первой в {k} {times} больше.")
        w = ch.calc(f"{v} * {k}")
        return ch.answer(f"В первой корзине {w}, во второй {v}.", "text", f"{w} и {v}"), q
    if kind == "more":
        d, x = r.randint(2, 40), r.randint(3, 80)
        total = 2 * x + d
        books = ("книга", "книги", "книг")
        q = (f"На двух полках {count(total, books)}. На первой на {d} больше, чем на второй. "
             f"Сколько книг на каждой полке?")
        ch = Chain(q).think(f"Пусть на второй полке x книг, тогда на первой x + {d}. Вместе 2x + {d} = {total}, "
                            f"значит 2x = {total} − {d}.")
        v = ch.calc(f"({total} - {d}) / 2")
        ch.think(f"На второй {v}, на первой на {d} больше.")
        w = ch.calc(f"{v} + {d}")
        return ch.answer(f"На первой полке {w}, на второй {v}.", "text", f"{w} и {v}"), q
    x, y = sorted(r.sample(range(2, 200), 2), reverse=True)
    q = f"Сумма двух чисел равна {x + y}, а их разность — {x - y}. Найди эти числа."
    ch = Chain(q).think(f"Если к сумме прибавить разность, получится удвоенное большее число: "
                        f"({x + y} + {x - y}) : 2.")
    a = ch.calc(f"({x + y} + {x - y}) / 2")
    ch.think("Меньшее число = сумма − большее.")
    b = ch.calc(f"{x + y} - {a}")
    return ch.answer(f"{a} и {b}", "text", f"{a} и {b}"), q


@topic(7, MATH, "m7_triangle", "сумма углов треугольника")
def _(r):
    if r.random() < 0.6:
        a, b = r.randint(20, 100), r.randint(20, 100)
        if a + b >= 170:
            return None
        q = f"Два угла треугольника равны {a}° и {b}°. Найди третий угол."
        ch = Chain(q).think("Сумма углов треугольника равна 180°.")
        v = ch.calc(f"180 - {a} - {b}")
        return ch.answer(f"Третий угол — {v}°.", "number", str(v)), q
    top = r.randrange(20, 160, 2)
    q = f"В равнобедренном треугольнике угол при вершине равен {top}°. Найди углы при основании."
    ch = Chain(q).think("Углы при основании равнобедренного треугольника равны, а сумма всех углов — 180°: "
                        "каждый угол при основании = (180° − угол при вершине) : 2.")
    v = ch.calc(f"(180 - {top}) / 2")
    return ch.answer(f"Углы при основании — по {v}°.", "number", str(v)), q


# ───────────────────────── language data ─────────────────────────
print("loading language data…", file=sys.stderr)


def strip_accent(s):
    return s.replace("'", "").replace("ё", "ё").strip()


def first_form(cell):
    return strip_accent(cell.split(",")[0].split(";")[0])


CYR = re.compile(r"^[а-яё]+$")
BAD = re.compile(
    r"(?<![а-яё])(?:[а-яё]*(?:хуй|хуе|хуё|хуя|хуи|пизд|бляд|блят|мудак|мудил|гандон|залуп|пидор|пидар|педик|шлюх|сучар)[а-яё]*"
    r"|(?:за|у|вы|по|на|от|разъ|съ|въ|при|до|долбо|про|пере|подъ|отъ|о)?[её]б(?:[аулниёеыо][а-яё]*)?"
    r"|сука|суки|суку|сукой|сучка|сучки|бля)(?![а-яё])"
)

freq_rank = {}
with open(f"{RU20K}/words.tsv", encoding="utf-8") as f:
    for line in f:
        rk, w, *_ = line.split("\t")
        freq_rank[w] = int(rk)


def common(w, top=20000):
    return freq_rank.get(w.replace("ё", "е"), 10**9) <= top


NOUNS, VERBS, ADJS = {}, {}, {}
with open(f"{RU20K}/nouns.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        w = r["bare"].strip()
        if CYR.match(w) and r["gender"] in ("m", "f", "n") and not BAD.search(w):
            NOUNS.setdefault(w, r)
with open(f"{RU20K}/verbs.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        w = r["bare"].strip()
        if CYR.match(w) and r["aspect"] in ("perfective", "imperfective") and not BAD.search(w):
            VERBS.setdefault(w, r)
with open(f"{RU20K}/adjectives.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        w = r["bare"].strip()
        if CYR.match(w) and w.endswith(("ый", "ий", "ой")) and not BAD.search(w):
            ADJS.setdefault(w, r)

MORPH = {}
LABEL = {"PREF": "приставка", "ROOT": "корень", "SUFF": "суффикс", "END": "окончание", "POSTFIX": "постфикс",
         "LINK": "соединительная гласная"}
with open(f"{SCHOOL}/tikhonov.txt", encoding="utf-8") as f:
    for line in f:
        w, seg = line.rstrip("\n").split("\t")
        parts = [tuple(p.rsplit(":", 1)) for p in seg.split("/")]
        if all(len(p) == 2 and p[1] in LABEL for p in parts) and "".join(p[0] for p in parts) == w:
            MORPH[w] = parts

# UD sentences
SENTS = []
with open(f"{SCHOOL}/syntagrus.conllu", encoding="utf-8") as f:
    text, toks = None, []
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("# text = "):
            text = line[9:]
        elif not line:
            if text and toks:
                SENTS.append((text, toks))
            text, toks = None, []
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            feats = dict(x.split("=", 1) for x in c[5].split("|")) if c[5] != "_" else {}
            toks.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3], "feats": feats,
                         "head": int(c[6]), "deprel": c[7]})
print(f"  {len(NOUNS)} nouns, {len(VERBS)} verbs, {len(ADJS)} adjectives, {len(MORPH)} words with morphemes, "
      f"{len(SENTS)} UD sentences", file=sys.stderr)


def clean_sentence(text, toks, max_words=12):
    words = [t for t in toks if t["upos"] != "PUNCT"]
    if not 3 <= len(words) <= max_words or re.search(r"[A-Za-z0-9«»\"…]", text):
        return False
    return not BAD.search(text.lower())


SHORT_SENTS = [(t, k) for t, k in SENTS if clean_sentence(t, k)]
print(f"  {len(SHORT_SENTS)} short clean sentences", file=sys.stderr)

# ───────────────────────── dictionary (LOOKUP) ─────────────────────────
GENDER = {"m": "м. р.", "f": "ж. р.", "n": "ср. р."}
GENDER_WORD = {"m": "мужского", "f": "женского", "n": "среднего"}
POS_NAME = {"NOUN": "существительное", "ADJ": "прилагательное", "VERB": "глагол", "ADV": "наречие",
            "PRON": "местоимение", "DET": "местоимение", "NUM": "числительное", "ADP": "предлог",
            "CCONJ": "союз", "SCONJ": "союз", "PART": "частица", "INTJ": "междометие"}
SPECIAL_DECL = {"путь", "время", "имя", "знамя", "племя", "пламя", "бремя", "семя", "стремя", "темя", "вымя"}


def declension(w, r):
    if r["indeclinable"] == "1" or r["pl_only"] == "1" or w in SPECIAL_DECL:
        return None
    g = r["gender"]
    if w.endswith(("а", "я")) and g in ("f", "m"):
        return 1
    if g == "m" or (g == "n" and w.endswith(("о", "е", "ё"))):
        return 2
    if g == "f" and w.endswith("ь"):
        return 3
    return None


RAZNO = {"хотеть", "бежать", "чтить", "есть", "дать", "создать", "надоесть", "съесть"}


def conjugation(w, r):
    if w in RAZNO or w.endswith(("хотеть", "бежать", "дать", "есть")):
        return None, None
    third = first_form(r["presfut_pl3"])
    if third.endswith(("ут", "ют", "утся", "ются")):
        return 1, third
    if third.endswith(("ат", "ят", "атся", "ятся")):
        return 2, third
    return None, third


def morph_text(w):
    parts = MORPH.get(w)
    if not parts:
        return None, None
    seg = "-".join(p for p, _ in parts)
    by = collections.OrderedDict()
    for p, lab in parts:
        by.setdefault(lab, []).append(p)
    desc = []
    for lab in ("PREF", "ROOT", "LINK", "SUFF", "END", "POSTFIX"):
        if lab in by:
            name = LABEL[lab]
            many = len(by[lab]) > 1
            if many:
                name = {"приставка": "приставки", "корень": "корни", "суффикс": "суффиксы",
                        "соединительная гласная": "соединительные гласные", "окончание": "окончания",
                        "постфикс": "постфиксы"}[name]
            desc.append(f"{name} {', '.join(by[lab])}")
    return seg, desc


DICT = {}


def add_entry(w, pos, gram, extra=None, zero_end=False):
    seg, desc = morph_text(w)
    entry = f"{w} — {pos}" + (f", {gram}" if gram else "")
    info = {"word": w, "pos": pos, "gram": gram}
    if seg:
        if zero_end and not any(lab == "END" for _, lab in MORPH[w]):
            seg += "-∅"
            desc.append("окончание нулевое")
        entry += f"; состав: {seg} ({', '.join(desc)})"
        info["morphemes"] = [[p, LABEL[lab]] for p, lab in MORPH[w]]
    if extra:
        entry += f"; {extra}"
        info["extra"] = extra
    info["entry"] = entry
    DICT[w] = info


for w, r in NOUNS.items():
    gram = [GENDER[r["gender"]], "одуш." if r["animate"] == "1" else "неодуш."]
    d = declension(w, r)
    if r["indeclinable"] == "1":
        gram.append("несклоняемое")
    elif d:
        gram.append(f"{d}-е скл.")
    elif w in SPECIAL_DECL:
        gram.append("разносклоняемое")
    add_entry(w, "существительное", ", ".join(gram), zero_end=r["indeclinable"] != "1")
for w, r in VERBS.items():
    gram = ["сов. вид" if r["aspect"] == "perfective" else "несов. вид"]
    c, third = conjugation(w, r)
    if c:
        gram.append(f"{'I' if c == 1 else 'II'} спряж.")
    elif w in RAZNO:
        gram.append("разноспрягаемый")
    extra = [f"они {third}"] if third else []
    pair = first_form(r["partner"]) if r["partner"] else ""
    if pair and CYR.match(pair):
        extra.append(f"видовая пара: {pair}")
    add_entry(w, "глагол", ", ".join(gram), "; ".join(extra) or None)
for w, r in ADJS.items():
    forms = [first_form(r["decl_f_nom"]), first_form(r["decl_n_nom"]), first_form(r["decl_pl_nom"])]
    extra = []
    if all(forms):
        extra.append(f"формы: {', '.join([w] + forms)}")
    comp = first_form(r["comparative"]) if r["comparative"] else ""
    if comp and CYR.match(comp):
        extra.append(f"сравн. степень: {comp}")
    short = first_form(r["short_m"]) if r["short_m"] else ""
    if short and CYR.match(short):
        extra.append(f"краткая форма: {short}")
    add_entry(w, "прилагательное", None, "; ".join(extra) or None)
# other parts of speech: lemmas whose part of speech in SynTagRus is (almost) always the same
lemma_pos = collections.defaultdict(collections.Counter)
for _, toks in SENTS:
    for t in toks:
        if t["upos"] in POS_NAME:
            lemma_pos[t["lemma"].lower()][t["upos"]] += 1
for lemma, c in lemma_pos.items():
    if lemma in DICT or not CYR.match(lemma) or BAD.search(lemma):
        continue
    (pos, n), total = c.most_common(1)[0], sum(c.values())
    if pos in ("ADV", "PRON", "DET", "NUM", "ADP", "CCONJ", "SCONJ", "PART", "INTJ") and n >= 3 and n >= 0.95 * total:
        add_entry(lemma, POS_NAME[pos], None)
print(f"  dictionary: {len(DICT)} entries", file=sys.stderr)


def lemma_split(w):
    return split_of("lemma:" + w)


QUESTION_WORD = {"m": "какой?", "f": "какая?", "n": "какое?"}


def noun_question(r):
    return "кто?" if r["animate"] == "1" else "что?"


# ───────────────────────── Russian language ─────────────────────────
VOWELS = set("аеёиоуыэюя")
IOTATED = set("еёюя")


def letters_sounds(w):
    """(letters, sounds) by the school rules; None when the word has sounds the rules below
    cannot count (silent consonants, -тся, сч, long consonants)."""
    if re.search(r"стн|здн|рдц|лнц|вств|стл|нтск|тся|ться|сч|зч|жч|(.)\1", w):
        return None
    sounds = 0
    for i, ch in enumerate(w):
        if ch in "ьъ":
            continue
        if ch in IOTATED and (i == 0 or w[i - 1] in VOWELS or w[i - 1] in "ьъ"):
            sounds += 2
        else:
            sounds += 1
    return len(w), sounds


LEMMAS_COMMON = [w for w in list(NOUNS) + list(VERBS) + list(ADJS) if common(w, 15000)]
COMMON_NOUNS = [w for w in LEMMAS_COMMON if w in NOUNS]
COMMON_VERBS = [w for w in LEMMAS_COMMON if w in VERBS]
COMMON_ADJS = [w for w in LEMMAS_COMMON if w in ADJS]
COMMON_MORPH = [w for w in LEMMAS_COMMON if w in MORPH]
# the rest of the dictionary: words in the 50 000 most frequent forms of the subtitle list
TOP50 = {}
with open(f"{RU20K}/ru_50k.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        TOP50.setdefault(line.split()[0].replace("ё", "е"), i)
ALL_NOUNS = [w for w in NOUNS if w.replace("ё", "е") in TOP50]
ALL_VERBS = [w for w in VERBS if w.replace("ё", "е") in TOP50]
ALL_ADJS = [w for w in ADJS if w.replace("ё", "е") in TOP50]


@topic(1, LANG, "r1_syllables", "слоги")
def _(r):
    w = r.choice(LEMMAS_COMMON)
    v = [c for c in w if c in VOWELS]
    q = f"Сколько слогов в слове «{w}»?"
    ch = Chain(q).think(f"Сколько в слове гласных, столько и слогов. Гласные: {', '.join(v)}.")
    n = len(v)
    return ch.answer(f"В слове «{w}» {n} {plural(n, ('слог', 'слога', 'слогов'))}.", "number", str(n)), "lemma:" + w


@topic(1, LANG, "r1_letters_sounds", "буквы и звуки")
def _(r):
    w = r.choice(LEMMAS_COMMON)
    ls = letters_sounds(w)
    if not ls or len(w) > 9:
        return None
    nl, ns = ls
    notes = []
    if any(c in "ьъ" for c in w):
        notes.append("ь и ъ звука не обозначают")
    if any(c in IOTATED and (i == 0 or w[i - 1] in VOWELS or w[i - 1] in "ьъ") for i, c in enumerate(w)):
        notes.append("е, ё, ю, я в начале слова, после гласной или после ь, ъ обозначают два звука")
    q = f"Сколько букв и звуков в слове «{w}»?"
    ch = Chain(q).think(f"Считаю буквы: {nl}. " + (("Звуков: " + "; ".join(notes) + f" — получается {ns}.")
                                                     if notes else f"Каждая буква здесь обозначает один звук: {ns}."))
    return ch.answer(f"{nl} {plural(nl, ('буква', 'буквы', 'букв'))}, {ns} "
                     f"{plural(ns, ('звук', 'звука', 'звуков'))}.", "text", f"{nl}/{ns}"), "lemma:" + w


PROPER = collections.defaultdict(list)  # sentences with a proper name to capitalise
for text, toks in SHORT_SENTS:
    names = [t for t in toks if t["upos"] == "PROPN" and t["feats"].get("NameType") in ("Geo", "Giv", "Sur", "Zoo")
             and t["id"] > 1 and CYR.match(t["form"].lower()) and t["form"][:1].isupper() and t["form"][1:].islower()]
    if len(names) == 1:
        PROPER[names[0]["feats"]["NameType"]].append((text, names[0]))
KIND = {"Geo": "название города, страны, реки или другого места", "Giv": "имя человека", "Sur": "фамилия",
        "Zoo": "кличка животного"}


@topic(1, LANG, "r1_capital", "большая буква в именах собственных")
def _(r):
    kind = r.choice(list(PROPER))
    text, t = r.choice(PROPER[kind])
    lower = text.replace(t["form"], t["form"].lower(), 1)
    q = r.choice([f"Какое слово в предложении «{lower}» нужно написать с большой буквы?",
                  f"Найди ошибку в предложении «{lower}»."])
    ch = Chain(q).think(f"С большой буквы пишутся имена собственные: имена, фамилии, клички животных, названия "
                        f"городов, стран и рек. Здесь «{t['form'].lower()}» — {KIND[kind]}.")
    return ch.answer(f"«{t['form']}» — с большой буквы: {text}", "text", t["form"]), "sent:" + text


@topic(2, LANG, "r2_pos", "части речи: существительное, прилагательное, глагол")
def _(r):
    kind = r.choice(["n", "a", "v"])
    w = r.choice({"n": COMMON_NOUNS, "a": COMMON_ADJS, "v": COMMON_VERBS}[kind])
    if w not in DICT or sum((w in NOUNS, w in ADJS, w in VERBS)) > 1:
        return None
    q = r.choice([f"Какая часть речи слово «{w}»?", f"Определи часть речи: «{w}»."])
    ch = Chain(q).think("Задам к слову вопрос и сверюсь со словарём.")
    ch.lookup(w)
    if kind == "n":
        ans = f"Существительное: обозначает предмет, отвечает на вопрос «{noun_question(NOUNS[w])}»."
        val = "существительное"
    elif kind == "a":
        ans = "Прилагательное: обозначает признак предмета, отвечает на вопрос «какой?»."
        val = "прилагательное"
    else:
        qv = "что сделать?" if VERBS[w]["aspect"] == "perfective" else "что делать?"
        ans = f"Глагол: обозначает действие, отвечает на вопрос «{qv}»."
        val = "глагол"
    return ch.answer(ans, "text", val), "lemma:" + w


@topic(2, LANG, "r2_gender", "род существительных")
def _(r):
    w = r.choice(COMMON_NOUNS)
    rr = NOUNS[w]
    if rr["pl_only"] == "1" or w not in DICT:
        return None
    g = rr["gender"]
    mine = {"m": "он мой", "f": "она моя", "n": "оно моё"}[g]
    q = r.choice([f"Какого рода существительное «{w}»?", f"Определи род слова «{w}»."])
    ch = Chain(q).think("Подставлю слова «он мой», «она моя», «оно моё» и сверюсь со словарём.")
    ch.lookup(w)
    return ch.answer(f"{GENDER_WORD[g].capitalize()} рода ({mine}).", "text", GENDER_WORD[g]), "lemma:" + w


CASE = {"Nom": ("именительный", "кто? что?"), "Gen": ("родительный", "кого? чего?"),
        "Dat": ("дательный", "кому? чему?"), "Acc": ("винительный", "кого? что?"),
        "Ins": ("творительный", "кем? чем?"), "Loc": ("предложный", "о ком? о чём?")}
CASE_Q = {"Nom": ("кто", "что"), "Gen": ("кого", "чего"), "Dat": ("кому", "чему"), "Acc": ("кого", "что"),
          "Ins": ("кем", "чем"), "Loc": ("ком", "чём")}


def with_prep(toks, t):
    preps = [x for x in toks if x["head"] == t["id"] and x["deprel"] == "case" and x["upos"] == "ADP"]
    return preps[0]["form"].lower() if preps else None


@topic(3, LANG, "r3_case", "падеж существительного в предложении")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    nouns = [t for t in toks if t["upos"] == "NOUN" and t["feats"].get("Case") in CASE and t["head"] > 0
             and CYR.match(t["form"].lower())]
    if not nouns:
        return None
    t = r.choice(nouns)
    head = next(x for x in toks if x["id"] == t["head"])
    if head["upos"] == "PUNCT":
        return None
    case = t["feats"]["Case"]
    anim = t["feats"].get("Animacy") == "Anim"
    qw = CASE_Q[case][0 if anim else 1]
    prep = with_prep(toks, t)
    if case == "Loc" and not prep:
        return None
    ask = f"{prep + ' ' if prep else ''}{qw}?"
    q = f"Определи падеж слова «{t['form']}» в предложении «{text}»."
    ch = Chain(q).think(f"Нахожу слово, от которого оно зависит, и задаю вопрос: {head['form']} ({ask}) "
                        f"{(prep + ' ') if prep else ''}{t['form']}. Вопрос «{CASE[case][1]}» — у "
                        f"{CASE[case][0][:-2]}ого падежа.")
    return ch.answer(f"{CASE[case][0].capitalize()} падеж.", "text", CASE[case][0]), "sent:" + text


TENSE = {"Past": "прошедшее", "Pres": "настоящее", "Fut": "будущее"}


@topic(3, LANG, "r3_tense", "время глагола")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    verbs = [t for t in toks if t["upos"] == "VERB" and t["feats"].get("VerbForm") == "Fin"
             and t["feats"].get("Tense") in TENSE and t["feats"].get("Mood") == "Ind"]
    if not verbs:
        return None
    t = r.choice(verbs)
    tense, perf = t["feats"]["Tense"], t["feats"].get("Aspect") == "Perf"
    qv = {"Past": "что сделал?" if perf else "что делал?", "Pres": "что делает?",
          "Fut": "что сделает?" if perf else "что будет делать?"}[tense]
    why = {"Past": "действие уже было", "Pres": "действие происходит сейчас", "Fut": "действие будет потом"}[tense]
    q = f"Определи время глагола «{t['form']}» в предложении «{text}»."
    ch = Chain(q).think(f"Задаю вопрос: «{qv}» — {why}.")
    return ch.answer(f"{TENSE[tense].capitalize()} время.", "text", TENSE[tense]), "sent:" + text


@topic(3, LANG, "r3_morphemes", "состав слова", n=8000)
def _(r):
    w = r.choice(COMMON_MORPH)
    q = r.choice([f"Разбери слово «{w}» по составу.", f"Какой состав у слова «{w}»?"])
    ch = Chain(q).think("Окончание — изменяемая часть слова, корень — общая часть родственных слов, приставка "
                        "стоит перед корнем, суффикс — после корня. Сверюсь с морфемным словарём.")
    e = ch.lookup(w)
    seg, desc = morph_text(w)
    if w in NOUNS and not any(lab == "END" for _, lab in MORPH[w]) and NOUNS[w]["indeclinable"] != "1":
        desc.append("окончание нулевое")
    return ch.answer("; ".join(desc).capitalize() + ".", "text", seg), "lemma:" + w


@topic(3, LANG, "r3_root", "корень слова")
def _(r):
    w = r.choice(COMMON_MORPH)
    roots = [p for p, lab in MORPH[w] if lab == "ROOT"]
    if len(roots) != 1:
        return None
    q = f"Выдели корень в слове «{w}»."
    ch = Chain(q).think("Корень — общая часть родственных слов. Проверю по морфемному словарю.")
    ch.lookup(w)
    return ch.answer(f"Корень — «{roots[0]}».", "text", roots[0]), "lemma:" + w


@topic(4, LANG, "r4_declension", "склонение существительных")
def _(r):
    w = r.choice(COMMON_NOUNS if r.random() < 0.5 else ALL_NOUNS)
    rr = NOUNS[w]
    d = declension(w, rr)
    if not d or w not in DICT:
        return None
    why = {1: "существительные женского и мужского рода с окончанием -а, -я относятся к 1-му склонению",
           2: "существительные мужского рода с нулевым окончанием и среднего рода с окончанием -о, -е "
              "относятся ко 2-му склонению",
           3: "существительные женского рода на мягкий знак относятся к 3-му склонению"}[d]
    q = f"Определи склонение существительного «{w}»."
    ch = Chain(q).think("Склонение зависит от рода и окончания. Узнаю род по словарю.")
    ch.lookup(w)
    ch.think(f"«{w}» — {GENDER_WORD[rr['gender']]} рода; {why}.")
    return ch.answer(f"{d}-е склонение.", "text", str(d)), "lemma:" + w


@topic(4, LANG, "r4_conjugation", "спряжение глаголов")
def _(r):
    w = r.choice(COMMON_VERBS if r.random() < 0.5 else ALL_VERBS)
    c, third = conjugation(w, VERBS[w])
    if not c or w not in DICT:
        return None
    end = re.search(r"(ут|ют|ат|ят)(ся)?$", third).group(1)
    q = f"Определи спряжение глагола «{w}»."
    ch = Chain(q).think("Поставлю глагол в форму 3-го лица множественного числа: окончание -ут, -ют — I спряжение, "
                        "-ат, -ят — II спряжение. Форму возьму из словаря.")
    ch.lookup(w)
    ch.think(f"Они {third}: окончание -{end}.")
    return ch.answer(f"{'I' if c == 1 else 'II'} спряжение.", "text", "I" if c == 1 else "II"), "lemma:" + w


PRON = {("1", "Sing"): "я", ("2", "Sing"): "ты", ("3", "Sing"): "он (она, оно)", ("1", "Plur"): "мы",
        ("2", "Plur"): "вы", ("3", "Plur"): "они"}


@topic(4, LANG, "r4_person", "лицо и число глаголов")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    verbs = [t for t in toks if t["upos"] == "VERB" and t["feats"].get("VerbForm") == "Fin"
             and t["feats"].get("Tense") in ("Pres", "Fut") and t["feats"].get("Person") in ("1", "2", "3")]
    if not verbs:
        return None
    t = r.choice(verbs)
    p, n = t["feats"]["Person"], t["feats"].get("Number")
    if (p, n) not in PRON:
        return None
    num = "единственное" if n == "Sing" else "множественное"
    q = f"Определи лицо и число глагола «{t['form']}» в предложении «{text}»."
    ch = Chain(q).think(f"Подставляю местоимение: {PRON[(p, n)]} {t['form'].lower()}.")
    return ch.answer(f"{p}-е лицо, {num} число.", "text", f"{p} {num}"), "sent:" + text


def school_pos(t):
    u, f = t["upos"], t["feats"]
    if u == "VERB" or u == "AUX":
        vf = f.get("VerbForm")
        return {"Part": "причастие", "Conv": "деепричастие"}.get(vf, "глагол")
    if u == "ADJ" and f.get("NumType") == "Ord":
        return "числительное"
    return POS_NAME.get(u)


POS_WHY = {
    "существительное": "оно обозначает предмет и отвечает на вопрос «кто?» или «что?»",
    "прилагательное": "оно обозначает признак предмета и отвечает на вопрос «какой?»",
    "глагол": "оно обозначает действие и отвечает на вопрос «что делать?» или «что сделать?»",
    "причастие": "оно обозначает признак предмета по действию, отвечает на вопрос «какой?» и образовано от глагола",
    "деепричастие": "оно обозначает добавочное действие и отвечает на вопрос «что делая?» или «что сделав?»",
    "наречие": "оно обозначает признак действия, отвечает на вопрос «как?», «где?» или «когда?» и не изменяется",
    "местоимение": "оно указывает на предмет или признак, но не называет его",
    "числительное": "оно обозначает количество или порядок предметов при счёте",
    "предлог": "это служебное слово: оно связывает слова и стоит перед существительным или местоимением",
    "союз": "это служебное слово: оно связывает однородные члены или части сложного предложения",
    "частица": "это служебное слово: оно придаёт слову или предложению дополнительный оттенок",
    "междометие": "оно выражает чувства или побуждения, но не называет их",
}


@topic(5, LANG, "r5_pos_context", "части речи в предложении", n=8000)
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    cands = [t for t in toks if school_pos(t) and CYR.match(t["form"].lower())]
    if not cands:
        return None
    t = r.choice(cands)
    pos = school_pos(t)
    q = f"Какой частью речи является слово «{t['form']}» в предложении «{text}»?"
    ch = Chain(q).think(f"Смотрю, что обозначает слово «{t['form']}» и на какой вопрос отвечает: {POS_WHY[pos]}.")
    return ch.answer(f"{pos.capitalize()}.", "text", pos), "sent:" + text


@topic(5, LANG, "r5_members", "подлежащее и сказуемое")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    root = [t for t in toks if t["deprel"] == "root" and t["upos"] == "VERB" and t["feats"].get("VerbForm") == "Fin"]
    if not root:
        return None
    root = root[0]
    subj = [t for t in toks if t["head"] == root["id"] and t["deprel"] == "nsubj" and t["upos"] in ("NOUN", "PRON")]
    if len(subj) != 1 or any(t["head"] == root["id"] and t["deprel"] in ("aux", "aux:pass", "cop", "conj")
                             for t in toks):
        return None
    s = subj[0]
    sq = "кто?" if s["feats"].get("Animacy") == "Anim" or s["upos"] == "PRON" else "что?"
    q = f"Найди подлежащее и сказуемое в предложении «{text}»."
    ch = Chain(q).think(f"Подлежащее отвечает на вопрос «кто?» или «что?», сказуемое говорит, что делает "
                        f"подлежащее: {sq[:-1]}? — {s['form'].lower()}; что делает? — {root['form'].lower()}.")
    return ch.answer(f"Подлежащее — «{s['form']}», сказуемое — «{root['form']}».", "text",
                     f"{s['form'].lower()}|{root['form'].lower()}"), "sent:" + text


@topic(6, LANG, "r6_aspect", "вид глагола")
def _(r):
    w = r.choice(COMMON_VERBS if r.random() < 0.5 else ALL_VERBS)
    if w not in DICT:
        return None
    perf = VERBS[w]["aspect"] == "perfective"
    q = f"Определи вид глагола «{w}»."
    ch = Chain(q).think("Задаю вопрос: «что делать?» — несовершенный вид, «что сделать?» — совершенный. "
                        "Проверю по словарю.")
    ch.lookup(w)
    name = "совершенный" if perf else "несовершенный"
    return ch.answer(f"{name.capitalize()} вид (отвечает на вопрос «{'что сделать?' if perf else 'что делать?'}»).",
                     "text", name), "lemma:" + w


@topic(6, LANG, "r6_comparative", "степени сравнения прилагательных")
def _(r):
    w = r.choice(ALL_ADJS)
    comp = first_form(ADJS[w]["comparative"]) if ADJS[w]["comparative"] else ""
    if not comp or not CYR.match(comp) or w not in DICT:
        return None
    q = f"Образуй сравнительную степень прилагательного «{w}»."
    ch = Chain(q).think("Простая сравнительная степень образуется суффиксами -ее, -е, -ше; у некоторых слов она "
                        "особая. Проверю по словарю.")
    ch.lookup(w)
    return ch.answer(f"{comp.capitalize()} (более {w}).", "text", comp), "lemma:" + w


@topic(7, LANG, "r7_participle", "причастие")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    parts = [t for t in toks if t["upos"] == "VERB" and t["feats"].get("VerbForm") == "Part"
             and t["feats"].get("Voice") in ("Act", "Pass") and t["feats"].get("Tense") in ("Pres", "Past")]
    if len(parts) != 1:
        return None
    t = parts[0]
    voice = "действительное" if t["feats"]["Voice"] == "Act" else "страдательное"
    tense = "настоящего" if t["feats"]["Tense"] == "Pres" else "прошедшего"
    short = t["feats"].get("Variant") == "Short"
    q = f"Найди причастие в предложении «{text}» и определи его признаки."
    ch = Chain(q).think(f"Причастие обозначает признак предмета по действию и отвечает на вопрос «какой?». "
                        f"Это «{t['form']}»: признак {'того, кто действует' if voice == 'действительное' else 'того, над чем действуют'} "
                        f"— {voice}; время — {tense}.")
    kind = f"{'краткое ' if short else ''}{voice} причастие {tense} времени"
    return ch.answer(f"«{t['form']}» — {kind}.", "text", f"{t['form'].lower()}|{voice}|{tense}"), "sent:" + text


@topic(7, LANG, "r7_gerund", "деепричастие")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    gs = [t for t in toks if t["upos"] == "VERB" and t["feats"].get("VerbForm") == "Conv" and t["feats"].get("Aspect")]
    if len(gs) != 1:
        return None
    t = gs[0]
    perf = t["feats"]["Aspect"] == "Perf"
    q = f"Найди деепричастие в предложении «{text}» и определи его вид."
    ch = Chain(q).think(f"Деепричастие обозначает добавочное действие: «{t['form']}» отвечает на вопрос "
                        f"«{'что сделав?' if perf else 'что делая?'}».")
    name = "совершенного" if perf else "несовершенного"
    return ch.answer(f"«{t['form']}» — деепричастие {name} вида.", "text", f"{t['form'].lower()}|{name}"), "sent:" + text


@topic(7, LANG, "r7_service", "служебные части речи")
def _(r):
    text, toks = r.choice(SHORT_SENTS)
    groups = collections.OrderedDict((("предлоги", []), ("союзы", []), ("частицы", [])))
    for t in toks:
        name = {"ADP": "предлоги", "CCONJ": "союзы", "SCONJ": "союзы", "PART": "частицы"}.get(t["upos"])
        if name and CYR.match(t["form"].lower()):
            groups[name].append(t["form"].lower())
    if sum(map(len, groups.values())) < 2:
        return None
    q = f"Найди служебные части речи в предложении «{text}»."
    ch = Chain(q).think("Служебные части речи — предлоги, союзы и частицы: они не отвечают на вопросы и не являются "
                        "членами предложения.")
    parts = [f"{k}: {', '.join(v)}" for k, v in groups.items() if v]
    return ch.answer("; ".join(parts).capitalize() + ".", "text", "; ".join(parts)), "sent:" + text


# ───────────────────────── write ─────────────────────────
with open(f"{SCHOOL}/chains.jsonl", "w", encoding="utf-8") as f:
    for e in EXAMPLES:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(f"{SCHOOL}/dictionary.jsonl", "w", encoding="utf-8") as f:
    for w in sorted(DICT):
        f.write(json.dumps(DICT[w], ensure_ascii=False) + "\n")

by_topic = collections.Counter((e["grade"], e["subject"], e["id"].rsplit("_", 1)[0], e["topic"]) for e in EXAMPLES)
acts = collections.Counter(s["act"] for e in EXAMPLES for s in e["steps"])
splits = collections.Counter(e["split"] for e in EXAMPLES)
lines = [f"examples: {len(EXAMPLES)} ({', '.join(f'{k} {v}' for k, v in splits.most_common())})",
         f"steps: {sum(acts.values())} ({', '.join(f'{k} {v}' for k, v in acts.most_common())})",
         f"dictionary entries: {len(DICT)}", ""]
for (g, s, code, name), n in sorted(by_topic.items()):
    lines.append(f"{g} класс | {s:<12} | {code:<20} | {name:<50} | {n}")
open(f"{SCHOOL}/stats.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), file=sys.stderr)
