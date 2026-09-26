#!/usr/bin/env python3
"""Convergence data of the unified model: training log + snapshot evaluations → CSV for
scripts/convergence_svg.py.

    scripts/unified_curves.py train.log evals.txt > docs/img/unified.csv
    scripts/convergence_svg.py docs/img/unified.csv docs/img/unified.svg

train.log: the output of `cog_engine train-unified` (held-out losses every --eval-every steps).
evals.txt: lines `step <N> agent <success %> grammar <held-out accuracy %>` (from unified-eval
runs on checkpoints saved during training).
"""

import re
import sys

log, evals = sys.argv[1], sys.argv[2]
rows = []
agent_rows = []
step = None
for line in open(log, encoding="utf-8"):
    m = re.match(r"step\s+(\d+)", line)
    if m:
        step = int(m.group(1))
    if line.strip().startswith("valid |") and step is not None:
        # one color per entity across panels (grammar is on both); text continuation is left out
        # to stay within the validated palette
        for name, series in [("dialogue", "диалоги"), ("grammar", "грамматика")]:
            v = re.search(rf"{name} ([\d.]+)", line)
            if v:
                rows.append(("uloss", series, step, float(v.group(1))))
for line in open(evals, encoding="utf-8"):
    f = line.split()
    if len(f) >= 6 and f[0] == "step":
        agent_rows.append(("uagent", "агент", int(f[1]), float(f[3])))
        agent_rows.append(("uagent", "грамматика", int(f[1]), float(f[5])))
print("panel,series,step,value")
for r in agent_rows + rows:
    print(",".join(str(x) for x in r))
