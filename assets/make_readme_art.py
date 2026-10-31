"""Generate the animated SVGs used in the README.

    python assets/make_readme_art.py

Writes race-{light,dark}.svg and terminal.svg next to
this file. Numbers come from benchmarks/headline.json, so the race chart
always matches the published results. Animations are plain CSS keyframes,
which GitHub renders inside <img>; they stop for viewers who ask for reduced motion.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

THEMES = {
    "light": dict(bg="#ffffff", ink="#1b1d2a", muted="#6a6f85", rule="#e4e5ee",
                  base="#c9cbd8", accent="#5b3fa8", accent2="#d0577f", glass="#f4f2fa"),
    "dark": dict(bg="#0d1117", ink="#e8e9f2", muted="#8b90a8", rule="#262a36",
                 base="#3a3f50", accent="#a48cf0", accent2="#f08aae", glass="#161b26"),
}

REDUCED = "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"
FONT = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def race(t, rows):
    W = 960
    top, rh = 70, 54
    H = top + len(rows) * rh + 40
    x0, bw = 330, 470
    vmax = max(r["omnia_items_per_s"] for r in rows)
    out = []
    for i, r in enumerate(rows):
        y = top + i * rh
        n, o = r["normal_items_per_s"], r["omnia_items_per_s"]
        d = i * .15
        out.append(f'''<text x="{x0 - 16}" y="{y + 15}" text-anchor="end" font-size="14" fill="{t["ink"]}">{r["label"]}</text>
<rect class="bar" style="animation-delay:{d:.2f}s" x="{x0}" y="{y}" width="{n / vmax * bw:.1f}" height="10" rx="2" fill="{t["base"]}"/>
<rect class="bar" style="animation-delay:{d + .25:.2f}s" x="{x0}" y="{y + 14}" width="{o / vmax * bw:.1f}" height="10" rx="2" fill="{t["accent"]}"/>
<text class="lab" style="animation-delay:{d + 1.3:.2f}s" x="{x0 + bw + 20}" y="{y + 21}" font-size="20" font-weight="700" fill="{t["accent"]}">{o / n:.1f}×</text>''')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Training throughput, original file vs .omnia">
<style>
text{{{FONT}}}
@keyframes grow{{0%{{transform:scaleX(0)}}25%,88%{{transform:scaleX(1)}}100%{{transform:scaleX(0)}}}}
@keyframes show{{0%,20%{{opacity:0}}30%,86%{{opacity:1}}100%{{opacity:0}}}}
.bar{{transform-box:fill-box;transform-origin:left;animation:grow 7s cubic-bezier(.3,.7,.2,1) infinite}}
.lab{{animation:show 7s infinite}}
{REDUCED}
</style>
<rect width="{W}" height="{H}" fill="{t["bg"]}"/>
<text x="40" y="34" font-size="18" font-weight="700" fill="{t["ink"]}">Training throughput, same GPU, same loop</text>
<rect x="{W - 230}" y="22" width="12" height="12" rx="2" fill="{t["base"]}"/><text x="{W - 212}" y="33" font-size="13" fill="{t["muted"]}">original file</text>
<rect x="{W - 110}" y="22" width="12" height="12" rx="2" fill="{t["accent"]}"/><text x="{W - 92}" y="33" font-size="13" fill="{t["muted"]}">.omnia</text>
{"".join(out)}
<text x="40" y="{H - 14}" font-size="12" fill="{t["muted"]}">NVIDIA A40 · ResNet-18 AMP · batch 64 · 4 workers · shuffled tiles · benchmarks/headline.json</text>
</svg>
'''


def terminal():
    W, H = 960, 330
    lines = [
        ("$ ", "curl -fsSL https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.sh | sh", 0.0),
        ("$ ", "omnia convert brain.nii.gz", 1.6),
        ("", "INFO: Converted brain.nii.gz (nifti): 18.2 MB -> 17.9 MB", 3.0),
        ("$ ", "omnia verify brain.omnia", 4.0),
        ("", "INFO: Verification passed: 160 tiles, 0 errors", 5.2),
        ("$ ", "python train.py   # OmniaDataset -> DataLoader", 6.2),
        ("", "# on an A40 GPU: 1,032 -> 2,024 slices/s, 2.0x faster", 7.6),
    ]
    out, css = [], []
    for i, (p, txt, d) in enumerate(lines):
        y = 82 + i * 34
        n = len(p + txt)
        col = "#e8e9f2" if p else "#8fd6a8"
        typed = bool(p)
        css.append(f".l{i}{{animation:{'type' if typed else 'pop'} 11s {d}s steps({n if typed else 1},end) infinite;"
                   f"--w:{n * 9.65:.0f}px}}")
        out.append(f'<clipPath id="c{i}"><rect class="l{i}" x="34" y="{y - 18}" height="26"/></clipPath>'
                   f'<text x="34" y="{y}" clip-path="url(#c{i})" fill="{col}">'
                   f'<tspan fill="#a48cf0">{p}</tspan>{txt.replace("&", "&amp;").replace(">", "&gt;").replace("<", "&lt;")}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Terminal: install, convert, verify, train">
<style>
text{{{MONO};font-size:16px;white-space:pre}}
@keyframes type{{0%{{width:0}}12%,92%{{width:var(--w)}}100%{{width:0}}}}
@keyframes pop{{0%{{width:0}}1%,92%{{width:900px}}100%{{width:0}}}}
rect[class^=l]{{width:0}}
{"".join(css)}
@media (prefers-reduced-motion: reduce){{rect[class^=l]{{animation:none!important;width:900px}}}}
</style>
<rect width="{W}" height="{H}" rx="12" fill="#0d1117"/>
<rect width="{W}" height="36" rx="12" fill="#161b26"/><rect y="24" width="{W}" height="12" fill="#161b26"/>
<circle cx="24" cy="18" r="6" fill="#ff5f57"/><circle cx="44" cy="18" r="6" fill="#febc2e"/><circle cx="64" cy="18" r="6" fill="#28c840"/>
<text x="{W / 2}" y="23" text-anchor="middle" fill="#8b90a8" style="font-size:13px">omnia</text>
{"".join(out)}
</svg>
'''


def main():
    data = json.loads((ROOT / "benchmarks/headline.json").read_text(encoding="utf-8"))
    short = {"Aperio SVS, full-res (JPEG 2000)": "Aperio SVS · JPEG 2000",
             "Aperio SVS 20x (JPEG)": "Aperio SVS · 20x",
             "Aperio SVS, full-res 40x (JPEG)": "Aperio SVS · 40x"}
    rows = [dict(r, label=short.get(r["format"], r["format"])) for r in data["rows"] if r["speedup"] >= 1.5]
    rows.sort(key=lambda r: -r["speedup"])
    for name, t in THEMES.items():
        (HERE / f"race-{name}.svg").write_text(race(t, rows), encoding="utf-8")
    (HERE / "terminal.svg").write_text(terminal(), encoding="utf-8")
    for r in rows:
        print(f"{r['label']:<32} {r['speedup']:.2f}x")


if __name__ == "__main__":
    main()
