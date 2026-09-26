#!/usr/bin/env python3
"""Builds the «ru20k» dataset: the 20 000 most frequent Russian words, taught three ways.

    scripts/fetch_ru20k.sh && scripts/fetch_ru_corpus.sh      # sources → data/ru20k, data/ru
    scripts/build_ru20k.py [data/ru20k] [data/ru]              # → data/ru20k/*.tsv, *.txt

Outputs (all UTF-8, one record per line):
  words.tsv         rank, word, corpus frequency, number of example sentences, lemma, part of speech
  sentences.tsv     split \t sentence — sentences chosen so that every word of the list occurs in
                    up to 5 of them (usage in context; continuation task)
  qa.tsv            split \t kind \t question \t answer — grammar of the words: plural, cases,
                    declension, gender, conjugation, past tense, aspect, imperative, adjective
                    forms and comparative (from openrussian.org paradigms)
  dialog.tsv        split \t turn_1 \t … \t turn_k \t reply — cleaned dialogue excerpts
  stats.txt         coverage report
Held-out split: 10% of the lemmas (qa), 1% of the sentences and dialogues ("valid").
"""

import collections
import csv
import hashlib
import random
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/ru20k"
UD = sys.argv[2] if len(sys.argv) > 2 else "data/ru"
N_WORDS = 20000
PER_WORD = 5
MAX_REPLY = 120
MAX_TURN = 150
random.seed(20)
csv.field_size_limit(10**7)

WORD = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
CYR = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$")
# Obscene words: a line containing one is dropped. Roots that are safe to match anywhere in a
# word, and the «еб» root only at the start of a word or after a verbal prefix (so that «требует»,
# «хлебу», «небо», «себе» pass).
BAD = re.compile(
    r"(?<![а-яё])(?:[а-яё]*(?:хуй|хуе|хуё|хуя|хуи|пизд|бляд|блят|мудак|мудил|гандон|залуп|пидор|пидар|педик|шлюх|сучар)[а-яё]*"
    r"|(?:за|у|вы|по|на|от|разъ|съ|въ|при|до|долбо|про|пере|подъ|отъ|о)?[её]б(?:[аулниёеыо][а-яё]*)?"
    r"|сука|суки|суку|сукой|сучка|сучки|бля)(?![а-яё])"
)


def words_of(text):
    return WORD.findall(text.lower().replace("ё", "е"))


def held_out(key, share):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1000 < share * 1000


def clean_line(line):
    line = line.strip()
    if line.startswith("-"):
        line = line[1:].strip()
    return re.sub(r"\s+", " ", line)


def usable(text, max_len):
    if not (2 <= len(text) <= max_len) or BAD.search(text.lower()):
        return False
    letters = sum(c.isalpha() for c in text)
    cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    return letters > 0 and cyr >= 0.8 * letters and not text.endswith(",")


def blocks(path):
    cur = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    yield cur
                cur = []
            else:
                cur.append(line)
    if cur:
        yield cur


# 1. the word list
freq = []
with open(f"{SRC}/ru_50k.txt", encoding="utf-8") as f:
    for line in f:
        w, c = line.split()
        w = w.lower().replace("ё", "е")
        # one-letter words are prepositions and conjunctions; longer ones need a vowel (subtitle
        # debris like «врн», «щрн» has none); obscene words are not taught
        if CYR.match(w) and (w in "авиокуся" if len(w) == 1 else re.search("[аеиоуыэюя]", w)) and not BAD.search(w):
            freq.append((w, int(c)))
freq = freq[:N_WORDS]
rank = {w: i for i, (w, _) in enumerate(freq)}
print(f"words: {len(freq)} (most frequent: {' '.join(w for w, _ in freq[:12])} …)", file=sys.stderr)

# 2. dialogues
pairs = set()
for path, name in [(f"{SRC}/dialogues.txt", "fiction"), (f"{SRC}/anekdots.txt", "jokes")]:
    n = 0
    for b in blocks(path):
        turns = [clean_line(t) for t in b]
        reply, context = turns[-1], turns[-4:-1]
        if not usable(reply, MAX_REPLY) or not context or not all(usable(t, MAX_TURN) for t in context):
            continue
        pairs.add(tuple(context) + (reply,))
        n += 1
    print(f"dialogues from {name}: {n}", file=sys.stderr)
pairs = sorted(pairs)
random.shuffle(pairs)

# 3. sentences covering the words
candidates = []
for path in [f"{UD}/train.txt", f"{SRC}/syntagrus.txt"]:
    with open(path, encoding="utf-8") as f:
        candidates += [clean_line(l) for l in f]
candidates += [p[-1] for p in pairs[:600000]]
candidates = [s for s in dict.fromkeys(candidates) if usable(s, 200)]
random.shuffle(candidates)
count = collections.Counter()
chosen = []
for s in candidates:
    ws = words_of(s)
    if not 3 <= len(ws) <= 25:
        continue
    new = {w for w in ws if w in rank and count[w] < PER_WORD}
    # rare words first: a sentence is taken if it adds a still-missing word
    if new:
        chosen.append(s)
        for w in new:
            count[w] += 1
sentences = [("valid" if held_out(s, 0.01) else "train", s) for s in chosen]

# 4. grammar questions from the dictionary
def forms(cell):
    """First variant of a dictionary cell, stress marks removed."""
    return cell.replace("'", "").split(",")[0].split(";")[0].strip()


lemma_of = {}
qa = []


def add(kind, lemma, templates, answer, held):
    if not answer or len(answer) > 90:
        return
    q = random.choice(templates).format(w=lemma)
    qa.append(("heldout" if held else "train", kind, q, answer))


def relevant(bare, cells):
    """A lemma is taught when it or one of its forms is among the 20k words."""
    ws = {forms(c).replace("ё", "е") for c in cells if c}
    ws.add(bare.replace("ё", "е"))
    hit = [w for w in ws if w in rank]
    for w in hit:
        lemma_of.setdefault(w, bare)
    return bool(hit)


CASES = ["nom", "gen", "dat", "acc", "inst", "prep"]
CASE_RU = {"gen": "родительном", "dat": "дательном", "acc": "винительном", "inst": "творительном", "prep": "предложном"}
with open(f"{SRC}/nouns.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        bare = r["bare"].strip()
        cells = [r[f"{n}_{c}"] for n in ("sg", "pl") for c in CASES]
        if not CYR.match(bare.lower()) or not relevant(bare, cells) or r["indeclinable"] == "1":
            continue
        held = held_out(bare, 0.10)
        sg = [forms(r[f"sg_{c}"]) for c in CASES]
        pl = [forms(r[f"pl_{c}"]) for c in CASES]
        if r["pl_only"] != "1" and all(sg):
            add("declension", bare, ["Просклоняй слово «{w}».", "Как склоняется «{w}»?", "Склонение слова «{w}»?"], ", ".join(sg), held)
            for c in ["gen", "dat", "inst", "prep"]:
                add(f"case_{c}", bare, [f"Как будет «{{w}}» в {CASE_RU[c]} падеже?", f"«{{w}}» в {CASE_RU[c]} падеже?"], sg[CASES.index(c)], held)
        if r["sg_only"] != "1" and pl[0]:
            add("plural", bare, ["Какое множественное число у слова «{w}»?", "Как будет «{w}» во множественном числе?",
                                 "Множественное число от «{w}»?"], pl[0], held)
        g = {"m": "мужского", "f": "женского", "n": "среднего"}.get(r["gender"])
        if g:
            add("gender", bare, ["Какого рода слово «{w}»?", "Род слова «{w}»?"], f"{g} рода", held)

with open(f"{SRC}/verbs.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        bare = r["bare"].strip()
        keys = ["presfut_sg1", "presfut_sg2", "presfut_sg3", "presfut_pl1", "presfut_pl2", "presfut_pl3"]
        cells = [r[k] for k in keys] + [r[k] for k in ("past_m", "past_f", "past_n", "past_pl", "imperative_sg")]
        if not CYR.match(bare.lower()) or not relevant(bare, cells):
            continue
        held = held_out(bare, 0.10)
        pres = [forms(r[k]) for k in keys]
        if all(pres):
            p = ["я", "ты", "он", "мы", "вы", "они"]
            add("conjugation", bare, ["Как спрягается глагол «{w}»?", "Проспрягай глагол «{w}».", "Спряжение глагола «{w}»?"],
                ", ".join(f"{a} {b}" for a, b in zip(p, pres)), held)
            add("first_person", bare, ["Как сказать «{w}» от первого лица?", "«{w}»: я …?"], f"я {pres[0]}", held)
        past = [forms(r[k]) for k in ("past_m", "past_f", "past_n", "past_pl")]
        if all(past):
            add("past", bare, ["Прошедшее время глагола «{w}»?", "Как будет «{w}» в прошедшем времени?"], ", ".join(past), held)
        if r["aspect"] in ("perfective", "imperfective"):
            asp = "совершенный" if r["aspect"] == "perfective" else "несовершенный"
            add("aspect", bare, ["Какой вид у глагола «{w}»?", "«{w}» — это какой вид?"], f"{asp} вид", held)
            partner = forms(r["partner"]) if r["partner"] else ""
            if partner and CYR.match(partner):
                add("aspect_pair", bare, ["Видовая пара к глаголу «{w}»?", "Парный глагол к «{w}»?"], partner, held)
        imp = forms(r["imperative_sg"])
        if imp:
            add("imperative", bare, ["Повелительное наклонение от «{w}»?", "Как приказать: «{w}»?"], imp, held)

with open(f"{SRC}/adjectives.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        bare = r["bare"].strip()
        cells = [r[k] for k in r if k.startswith("decl_") or k.startswith("short_")]
        if not CYR.match(bare.lower()) or not relevant(bare, cells):
            continue
        held = held_out(bare, 0.10)
        f_, n_, pl_ = forms(r["decl_f_nom"]), forms(r["decl_n_nom"]), forms(r["decl_pl_nom"])
        if f_ and n_ and pl_:
            add("adj_forms", bare, ["Род и число прилагательного «{w}»?", "Формы прилагательного «{w}»?"],
                f"{bare}, {f_}, {n_}, {pl_}", held)
            add("adj_fem", bare, ["Женский род от «{w}»?", "Как будет «{w}» в женском роде?"], f_, held)
            add("adj_plural", bare, ["Множественное число прилагательного «{w}»?"], pl_, held)
        comp = forms(r["comparative"])
        if comp and CYR.match(comp):
            add("comparative", bare, ["Сравнительная степень от «{w}»?", "Как сравнить: «{w}» → …?"], comp, held)

# 5. write everything
with open(f"{SRC}/words.tsv", "w", encoding="utf-8") as f:
    for i, (w, c) in enumerate(freq):
        f.write(f"{i + 1}\t{w}\t{c}\t{count[w]}\t{lemma_of.get(w, '')}\n")
with open(f"{SRC}/sentences.tsv", "w", encoding="utf-8") as f:
    for split, s in sentences:
        f.write(f"{split}\t{s}\n")
random.shuffle(qa)
with open(f"{SRC}/qa.tsv", "w", encoding="utf-8") as f:
    for row in qa:
        f.write("\t".join(row) + "\n")
with open(f"{SRC}/dialog.tsv", "w", encoding="utf-8") as f:
    for p in pairs:
        split = "valid" if held_out(p[-1] + p[-2], 0.01) else "train"
        f.write(split + "\t" + "\t".join(p) + "\n")

covered = [sum(count[w] >= k for w, _ in freq) for k in (1, 3, 5)]
taught = sum(1 for w, _ in freq if w in lemma_of)
kinds = collections.Counter(k for _, k, _, _ in qa)
report = [
    f"words: {len(freq)}",
    f"words with ≥1 / ≥3 / ≥5 example sentences: {covered[0]} / {covered[1]} / {covered[2]}",
    f"words whose lemma has grammar questions: {taught}",
    f"sentences: {len(sentences)} ({sum(s == 'valid' for s, _ in sentences)} valid)",
    f"grammar questions: {len(qa)} ({sum(s == 'heldout' for s, *_ in qa)} on held-out lemmas): "
    + ", ".join(f"{k} {v}" for k, v in kinds.most_common()),
    f"dialogue pairs: {len(pairs)}",
]
open(f"{SRC}/stats.txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
print("\n".join(report), file=sys.stderr)
