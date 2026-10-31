"""Generate the animated SVGs used in the README.

    python assets/make_readme_art.py

Writes hero-{light,dark}.svg, race-{light,dark}.svg and terminal.svg next to
this file. Numbers come from benchmarks/headline.json, so the race chart
always matches the published results. Animations are plain CSS keyframes,
which GitHub renders inside <img>; they stop for viewers who ask for reduced motion.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

THEMES = {
    "light": dict(bg="#ffffff", ink="#1b1d2a", muted="#6a6f85", rule="#e4e5ee",
                  base="#c9cbd8", accent="#5b3fa8", accent2="#d0577f", glass="#f4f2fa"),
    "dark": dict(bg="#0d1117", ink="#e8e9f2", muted="#8b90a8", rule="#262a36",
                 base="#3a3f50", accent="#a48cf0", accent2="#f08aae", glass="#161b26"),
}

# H&E stain palette: hematoxylin (nuclei, purple) and eosin (stroma, pink).
STAIN = ["#e9c6d8", "#dca4c2", "#c983ad", "#b56aa0", "#8e5c9e", "#6b4f99", "#f2dbe6", "#d6b0cf"]

REDUCED = "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"
FONT = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def hero(t):
    W, H = 960, 300
    rnd = random.Random(7)
    cols, rows, s, gap = 8, 6, 22, 3
    gx, gy = 70, 72
    tiles, css = [], []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            x, y = gx + c * (s + gap), gy + r * (s + gap)
            col = rnd.choice(STAIN)
            d = (i * 0.11) % 4.8
            # every tile flies from the slide into its slot in the .omnia column
            tx = 586 - x - s / 2
            ty = 150 - y - s / 2
            css.append(f".t{i}{{animation:fly 5.2s {d:.2f}s cubic-bezier(.6,0,.2,1) infinite;"
                       f"--tx:{tx}px;--ty:{ty}px}}")
            tiles.append(f'<rect class="t{i}" x="{x}" y="{y}" width="{s}" height="{s}" rx="3" fill="{col}"/>')
    # the .omnia container: stacked blocks, each with a CRC tick
    blocks = []
    for k in range(12):
        y = 60 + k * 14
        blocks.append(f'<rect x="596" y="{y}" width="102" height="11" rx="2" fill="{t["glass"]}" stroke="{t["rule"]}"/>'
                      f'<circle class="crc" style="animation-delay:{k * .38:.2f}s" cx="712" cy="{y + 5.5}" r="3" fill="{t["accent"]}"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Slide tiles converted into an .omnia container">
<style>
text{{{FONT}}}
.mono{{{MONO}}}
@keyframes fly{{0%,30%{{transform:none;opacity:1}}62%{{transform:translate(var(--tx),var(--ty)) scale(.3);opacity:0}}63%,85%{{transform:none;opacity:0}}100%{{transform:none;opacity:1}}}}
@keyframes tick{{0%,40%{{opacity:.15}}50%,90%{{opacity:1}}100%{{opacity:.15}}}}
@keyframes flow{{to{{stroke-dashoffset:-24}}}}
@keyframes rise{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}
[class^=t]{{transform-box:fill-box;transform-origin:center}}
.crc{{animation:tick 4.6s ease-in-out infinite}}
.flow{{stroke-dasharray:6 6;animation:flow 1s linear infinite}}
.n{{animation:rise .9s ease-out both}}
{"".join(css)}
{REDUCED}
</style>
<rect width="{W}" height="{H}" fill="{t["bg"]}"/>
<text x="70" y="52" font-size="13" fill="{t["muted"]}" letter-spacing="1.5">SVS · NDPI · MRXS · SCN · TIFF · DICOM · NIfTI</text>
{"".join(tiles)}
<path class="flow" d="M300 150 C 400 150, 480 150, 580 150" stroke="{t["accent"]}" stroke-width="2" fill="none"/>
<text x="440" y="138" text-anchor="middle" class="mono" font-size="13" fill="{t["muted"]}">omnia convert</text>
{"".join(blocks)}
<text x="596" y="250" class="mono" font-size="13" fill="{t["ink"]}">slide.omnia</text>
<text x="596" y="268" font-size="12" fill="{t["muted"]}">zstd · CRC per tile · O(1) seek</text>
<g class="n" style="animation-delay:.4s">
<text x="780" y="140" font-size="64" font-weight="700" fill="{t["accent"]}">8.5×</text>
<text x="782" y="168" font-size="14" fill="{t["ink"]}">faster training</text>
<text x="782" y="188" font-size="12" fill="{t["muted"]}">best case, A40 GPU</text>
</g>
</svg>
'''


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
        (HERE / f"hero-{name}.svg").write_text(hero(t), encoding="utf-8")
        (HERE / f"race-{name}.svg").write_text(race(t, rows), encoding="utf-8")
    (HERE / "terminal.svg").write_text(terminal(), encoding="utf-8")
    for r in rows:
        print(f"{r['label']:<32} {r['speedup']:.2f}x")


if __name__ == "__main__":
    main()
