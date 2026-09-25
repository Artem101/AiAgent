#!/usr/bin/env bash
# Downloads a compact open Russian corpus and extracts plain sentences.
#
#   scripts/fetch_ru_corpus.sh [out_dir]        (default: data/ru)
#
# Source: Universal Dependencies treebanks UD_Russian-GSD (Wikipedia/news) and
# UD_Russian-Taiga (web, social media, fiction), both CC BY-SA 4.0,
# https://universaldependencies.org. Only the `# text = …` lines are kept:
#   <out_dir>/train.txt — GSD train + Taiga train-a (one sentence per line)
#   <out_dir>/valid.txt — GSD test + Taiga test (held out)
set -euo pipefail

OUT="${1:-data/ru}"
BASE="https://raw.githubusercontent.com/UniversalDependencies"
TRAIN=(
  "UD_Russian-GSD/master/ru_gsd-ud-train.conllu"
  "UD_Russian-GSD/master/ru_gsd-ud-dev.conllu"
  "UD_Russian-Taiga/master/ru_taiga-ud-train-a.conllu"
)
VALID=(
  "UD_Russian-GSD/master/ru_gsd-ud-test.conllu"
  "UD_Russian-Taiga/master/ru_taiga-ud-test.conllu"
)

mkdir -p "$OUT"
extract() { # conllu on stdin → sentences on stdout
  sed -n 's/^# text = //p' | tr -d '\r' | awk 'length($0) > 0'
}
fetch() { # list of paths → concatenated sentences
  for path in "$@"; do
    echo "  $BASE/$path" >&2
    curl -fsSL --retry 3 "$BASE/$path" | extract
  done
}

echo "train:" >&2
fetch "${TRAIN[@]}" > "$OUT/train.txt"
echo "valid:" >&2
fetch "${VALID[@]}" > "$OUT/valid.txt"
for f in train valid; do
  printf '%s: %s sentences, %s bytes\n' "$OUT/$f.txt" "$(wc -l < "$OUT/$f.txt")" "$(wc -c < "$OUT/$f.txt")"
done
