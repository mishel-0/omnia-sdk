"""Generate the logo and the animated "how it works" diagram used in the README.

    python assets/make_brand.py

Writes logo-{light,dark}.svg, mark.svg and how-it-works-{light,dark}.svg next to this file.
The GPU-utilisation figures in the diagram come from benchmarks/mlarray_results_rtx4090.json
(CMU-1.svs, 4 workers), so the animation always matches the published results.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

THEMES = {
    "light": dict(bg="#ffffff", ink="#1b1d2a", muted="#6a6f85", rule="#dcdde8", box="#f5f4fa",
                  base="#b9bccd", accent="#5b3fa8", accent2="#8d74d6", ok="#2a7d5b"),
    "dark": dict(bg="#0d1117", ink="#e8e9f2", muted="#8b90a8", rule="#2a2f3d", box="#151a24",
                 base="#4a5062", accent="#a48cf0", accent2="#7358c9", ok="#5fc59a"),
}
FONT = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
REDUCED = "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"


def mark(t, size=64, x=0, y=0):
    """A 3x3 grid of tiles with the centre missing: an 'O' built from tiles."""
    s, g = size * 0.28, size * 0.08
    shades = [t["accent"], t["accent2"], t["accent"], t["accent2"], None, t["accent2"],
              t["accent"], t["accent2"], t["accent"]]
    out = []
    for i, c in enumerate(shades):
        if c is None:
            continue
        cx, cy = x + (i % 3) * (s + g), y + (i // 3) * (s + g)
        out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{s:.1f}" height="{s:.1f}" rx="{s * .22:.1f}" fill="{c}"/>')
    return "".join(out)


def logo(t):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 84" width="300" height="84" role="img" aria-label="omnia-sdk">
<style>text{{{FONT}}}</style>
{mark(t, 64, 10, 10)}
<text x="92" y="56" font-size="40" font-weight="700" letter-spacing="-1" fill="{t["ink"]}">omnia</text>
<text x="218" y="56" font-size="40" font-weight="300" letter-spacing="-1" fill="{t["muted"]}">sdk</text>
</svg>
'''


def lane(t, y, title, sub, src_label, mid_label, dots, util, util_delay, dur=6):
    out = [f'<text x="40" y="{y - 14}" font-size="15" font-weight="700" fill="{t["ink"]}">{title}</text>',
           f'<text x="40" y="{y + 4}" font-size="12" fill="{t["muted"]}">{sub}</text>']
    top = y + 14
    # source: a pyramid of three stacked levels
    for k, (w, h) in enumerate(((120, 18), (90, 18), (60, 18))):
        out.append(f'<rect x="{40 + (120 - w) / 2}" y="{top + k * 22}" width="{w}" height="{h}" rx="3" '
                   f'fill="{t["box"]}" stroke="{t["rule"]}"/>')
    out.append(f'<text x="100" y="{top + 82}" text-anchor="middle" class="m" font-size="12" fill="{t["muted"]}">{src_label}</text>')
    # lane the tiles travel along
    out.append(f'<line x1="200" y1="{top + 30}" x2="690" y2="{top + 30}" stroke="{t["rule"]}" stroke-dasharray="3 5"/>')
    for i in range(dots):
        d = i * dur / dots
        out.append(f'<rect class="tile" style="animation:go {dur}s linear {d:.2f}s infinite" x="200" y="{top + 24}" '
                   f'width="12" height="12" rx="2.5" fill="{t["accent"]}"/>')
    # the step between disk and GPU
    out.append(f'<rect x="300" y="{top + 8}" width="120" height="44" rx="6" fill="{t["box"]}" stroke="{t["rule"]}"/>'
               f'<text x="360" y="{top + 35}" text-anchor="middle" class="m" font-size="13" fill="{t["ink"]}">{mid_label}</text>')
    # GPU box with a utilisation bar
    out.append(f'<rect x="700" y="{top}" width="220" height="60" rx="6" fill="{t["box"]}" stroke="{t["rule"]}"/>'
               f'<text x="714" y="{top + 22}" font-size="12" fill="{t["muted"]}">GPU utilisation</text>'
               f'<rect x="714" y="{top + 32}" width="192" height="10" rx="5" fill="{t["rule"]}"/>'
               f'<rect class="bar" style="animation-delay:{util_delay}s;--w:{192 * util:.0f}px" x="714" y="{top + 32}" '
               f'width="0" height="10" rx="5" fill="{t["ok"] if util > .5 else t["base"]}"/>'
               f'<text x="906" y="{top + 22}" text-anchor="end" class="m" font-size="12" font-weight="700" fill="{t["ink"]}">{util * 100:.0f}%</text>')
    return "".join(out)


def how(t, data):
    r = {x["reader"]: x for x in next(s for s in data["results"] if s["sample"] == "CMU-1.svs")["readers"]}
    a, b = r["openslide read_region"]["gpu_share_w4"], r["omnia"]["gpu_share_w4"]
    H = 410
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {H}" width="960" height="{H}" role="img" aria-label="Reading tiles from the original slide vs from .omnia: GPU utilisation {a * 100:.0f}% vs {b * 100:.0f}%">
<style>
text{{{FONT}}}
.m{{{MONO}}}
@keyframes go{{0%{{transform:translateX(0);opacity:0}}6%{{opacity:1}}94%{{opacity:1}}100%{{transform:translateX(480px);opacity:0}}}}
@keyframes fill{{0%,8%{{width:0}}40%,88%{{width:var(--w)}}100%{{width:0}}}}
.bar{{animation:fill 6s ease-out infinite}}
{REDUCED}
</style>
<rect width="960" height="{H}" fill="{t["bg"]}"/>
{lane(t, 62, "Original file", "every epoch re-reads and re-decodes each tile", "slide.svs", "decode (CPU)", 5, a, 0)}
{lane(t, 222, ".omnia", "decoded once at conversion; each tile is one seek", "slide.omnia", "seek + zstd", 24, b, 0)}
<text x="40" y="{H - 18}" font-size="12" fill="{t["muted"]}">CMU-1.svs, ResNet-18 AMP, batch 64, 4 DataLoader workers, RTX 4090. Dot density is read throughput. Source: benchmarks/mlarray_results_rtx4090.json</text>
</svg>
'''


def main():
    data = json.loads((ROOT / "benchmarks/mlarray_results_rtx4090.json").read_text(encoding="utf-8"))
    for name, t in THEMES.items():
        (HERE / f"logo-{name}.svg").write_text(logo(t), encoding="utf-8")
        (HERE / f"how-it-works-{name}.svg").write_text(how(t, data), encoding="utf-8")
    (HERE / "mark.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 84 84" width="84" height="84">{mark(THEMES["light"], 64, 10, 10)}</svg>\n',
        encoding="utf-8")


if __name__ == "__main__":
    main()
