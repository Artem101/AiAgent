#!/usr/bin/env python3
"""Renders convergence curves (CSV → SVG, stdlib only).

    scripts/convergence_svg.py docs/img/convergence.csv docs/img/convergence.svg

CSV columns: panel,series,step,value. Panels are stacked vertically in order of appearance;
a series named `ref:<label>` is drawn as a thin muted reference line. Colors follow a
validated categorical palette (fixed slot order, light and dark variants); every series is
direct-labelled at its end and every point has a hover tooltip.
"""

import csv
import sys
from collections import OrderedDict

PANELS = {
    "agent": ("Агент: успешные эпизоды (200 задач, обучающие формулировки, симулятор), %", None),
    "text": ("Продолжение текста: верный следующий токен (500 отложенных предложений), %", None),
    "agent3": ("Агент: успешные эпизоды, все шесть семейств (200 задач, симулятор), %", None),
    "steps3": ("Агент: точные действия на отдельных состояниях (500 состояний), %", None),
    "calc3": ("Арифметика: верный ответ через калькулятор (эпизоды семейства calc), %", None),
    "text3": ("Продолжение текста: верный следующий токен (500 отложенных предложений), %", None),
}
# Fixed slot order (blue, orange, aqua, yellow). Each entity keeps its color on every panel;
# a panel shows at most three series (the validated all-pairs cap) and orange never meets
# yellow on one panel.
SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
W, PH, ML, MR, MT, MB = 880, 250, 56, 210, 44, 36


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def nice_max(v):
    for m in (5, 10, 20, 25, 40, 50, 60, 80, 100):
        if v <= m:
            return m
    return v


def main(src, dst):
    data = OrderedDict()
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            panel = data.setdefault(row["panel"], OrderedDict())
            panel.setdefault(row["series"], []).append((int(row["step"]), float(row["value"])))
    series_names = []
    for panel in data.values():
        for s in panel:
            if not s.startswith("ref:") and s not in series_names:
                series_names.append(s)
    if len(series_names) > len(SLOTS_LIGHT):
        sys.exit("more series than validated slots — fold or facet")
    slot = {s: i for i, s in enumerate(series_names)}
    for pid, panel in data.items():
        used = {slot[s] for s in panel if not s.startswith("ref:")}
        if len(used) > 3 or {1, 3} <= used:
            sys.exit(f"panel {pid}: more than 3 series, or orange next to yellow — facet instead")
    height = len(data) * (PH + MT + MB) + 8
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        'font-family="system-ui,-apple-system,Segoe UI,sans-serif" role="img">',
        "<style>",
        ":root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;"
        + "".join(f"--s{i}:{c};" for i, c in enumerate(SLOTS_LIGHT))
        + "}",
        "@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;"
        "--grid:#2c2c2a;--axis:#383835;" + "".join(f"--s{i}:{c};" for i, c in enumerate(SLOTS_DARK)) + "}}",
        "text{fill:var(--ink2);font-size:12px}.t{fill:var(--ink);font-size:14px;font-weight:600}"
        ".l{fill:var(--ink);font-size:12px}.g{stroke:var(--grid);stroke-width:1}.a{stroke:var(--axis);stroke-width:1}"
        ".ref{stroke:var(--muted);stroke-width:1.5}",
        "</style>",
        f'<rect width="{W}" height="{height}" fill="var(--surface)"/>',
    ]
    y0 = 0
    for pid, panel in data.items():
        title, ymax = PANELS.get(pid, (pid, None))
        steps = [st for pts in panel.values() for st, _ in pts]
        xmax = max(steps)
        ymax = ymax or nice_max(max(v for pts in panel.values() for _, v in pts) * 1.1)
        top, left, pw = y0 + MT, ML, W - ML - MR
        x = lambda st: left + pw * st / xmax
        y = lambda v: top + PH * (1 - v / ymax)
        out.append(f'<text class="t" x="{left}" y="{y0 + 24}">{esc(title)}</text>')
        for i in range(6):  # recessive horizontal grid + y ticks
            v = ymax * i / 5
            out.append(f'<line class="{"a" if i == 0 else "g"}" x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            out.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        xticks = sorted(set(steps) | {0})
        for st in xticks:
            out.append(f'<text x="{x(st):.1f}" y="{top + PH + 18}" text-anchor="middle">{st}</text>')
        out.append(f'<text x="{left + pw}" y="{top + PH + 32}" text-anchor="end">шаг обучения</text>')
        labels = []
        for name, pts in panel.items():
            pts.sort()
            path = " ".join(f"{'M' if i == 0 else 'L'}{x(s):.1f},{y(v):.1f}" for i, (s, v) in enumerate(pts))
            if name.startswith("ref:"):
                out.append(f'<path class="ref" d="{path}" fill="none"/>')
                labels.append((y(pts[-1][1]), x(pts[-1][0]), name[4:], None, pts[-1][1]))
                continue
            c = f"var(--s{slot[name]})"
            out.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for s, v in pts:
                out.append(
                    f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="4" fill="{c}" stroke="var(--surface)" stroke-width="2">'
                    f"<title>{esc(name)} · шаг {s}: {v:.1f}%</title></circle>"
                )
            labels.append((y(pts[-1][1]), x(pts[-1][0]), name, c, pts[-1][1]))
        # direct labels at the right edge; spread vertically, leader lines back to the line ends
        labels.sort()
        placed = []
        for ly, *_ in labels:
            placed.append(max(ly, placed[-1] + 18) if placed else ly)
        overflow = placed[-1] - (top + PH) if placed else 0
        if overflow > 0:  # keep the label column inside the panel
            placed = [p - overflow for p in placed]
            for i in range(len(placed) - 2, -1, -1):
                placed[i] = min(placed[i], placed[i + 1] - 18)
        for (ly, lx, name, c, v), ty in zip(labels, placed):
            tx = left + pw + 14
            out.append(f'<line x1="{lx + 5:.1f}" y1="{ly:.1f}" x2="{tx - 2:.1f}" y2="{ty:.1f}" stroke="var(--axis)" stroke-width="1"/>')
            if c:
                out.append(f'<circle cx="{tx + 5}" cy="{ty:.1f}" r="4" fill="{c}"/>')
            out.append(f'<text class="l" x="{tx + 14}" y="{ty + 4:.1f}">{esc(name)} · {v:.1f}%</text>')
        y0 += PH + MT + MB
    out.append("</svg>")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
