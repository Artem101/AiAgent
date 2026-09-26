#!/usr/bin/env bash
# Downloads the sources of the school dataset (then run scripts/build_school.py).
#
#   scripts/fetch_school.sh [out_dir]        (default: data/school)
#
# Sources (all public, fetched from GitHub):
#   syntagrus.conllu   — UD_Russian-SynTagRus with full annotation: lemma, part of speech,
#                        case, gender, number, tense, aspect, person, participles and gerunds,
#                        syntax (subject / predicate). UniversalDependencies (CC BY-NC-SA 4.0)
#   tikhonov.txt       — ~96 000 words split into typed morphemes (prefix, root, suffix, ending,
#                        postfix) from A. N. Tikhonov's morpheme dictionary, as prepared by
#                        AlexeySorokin/NeuralMorphemeSegmentation (research data; the dictionary
#                        itself is copyrighted — used locally for training, never committed)
# The paradigms of openrussian.org come from scripts/fetch_ru20k.sh (data/ru20k).
set -euo pipefail

OUT="${1:-data/school}"
RAW="https://raw.githubusercontent.com"
mkdir -p "$OUT"
get() { echo "  $1" >&2; curl -fsSL --retry 3 "$1"; }

: > "$OUT/syntagrus.conllu"
for part in train-a train-b train-c dev test; do
  get "$RAW/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-$part.conllu" >> "$OUT/syntagrus.conllu"
done

: > "$OUT/tikhonov.txt"
for part in train test; do
  get "$RAW/AlexeySorokin/NeuralMorphemeSegmentation/master/data/${part}_Tikhonov_reformat.txt" >> "$OUT/tikhonov.txt"
done

wc -l "$OUT"/syntagrus.conllu "$OUT"/tikhonov.txt >&2
