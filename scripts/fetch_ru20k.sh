#!/usr/bin/env bash
# Downloads the sources of the «ru20k» dataset (then run scripts/build_ru20k.py).
#
#   scripts/fetch_ru20k.sh [out_dir]        (default: data/ru20k)
#
# Sources (all public, fetched from GitHub):
#   ru_50k.txt              — the 50 000 most frequent Russian words of OpenSubtitles 2018,
#                             hermitdave/FrequencyWords (CC BY-SA 4.0)
#   nouns.csv, verbs.csv,   — full paradigms of ~54 000 lemmas from openrussian.org,
#   adjectives.csv,           Badestrand/russian-dictionary (CC BY-SA 4.0)
#   others.csv
#   syntagrus.txt           — sentences of UD_Russian-SynTagRus (fiction, news, science),
#                             UniversalDependencies (CC BY-NC-SA 4.0)
#   dialogues.txt           — ~1 M dialogue excerpts from fiction and
#   anekdots.txt              ~90 000 dialogues from jokes, Koziev/NLP_Datasets
# The UD Russian GSD/Taiga sentences come from scripts/fetch_ru_corpus.sh (data/ru).
set -euo pipefail

OUT="${1:-data/ru20k}"
RAW="https://raw.githubusercontent.com"
mkdir -p "$OUT"
get() { echo "  $1" >&2; curl -fsSL --retry 3 "$1"; }

get "$RAW/hermitdave/FrequencyWords/master/content/2018/ru/ru_50k.txt" > "$OUT/ru_50k.txt"
for f in nouns verbs adjectives others; do
  get "$RAW/Badestrand/russian-dictionary/master/$f.csv" > "$OUT/$f.csv"
done
get "$RAW/Badestrand/russian-dictionary/master/LICENSE" > "$OUT/openrussian.LICENSE"

: > "$OUT/syntagrus.txt"
for part in train-a train-b train-c dev; do
  get "$RAW/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-$part.conllu" \
    | sed -n 's/^# text = //p' | tr -d '\r' >> "$OUT/syntagrus.txt"
done

TMP="$(mktemp -d)"
get "$RAW/Koziev/NLP_Datasets/master/Conversations/Data/dialogues.zip" > "$TMP/d.zip"
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extract('dialogues.txt', sys.argv[2])" "$TMP/d.zip" "$OUT"
get "$RAW/Koziev/NLP_Datasets/master/Conversations/Data/extract_dialogues_from_anekdots.tar.xz" \
  | tar -xJ -O > "$OUT/anekdots.txt"
rm -rf "$TMP"

wc -l "$OUT"/*.txt "$OUT"/*.csv >&2
