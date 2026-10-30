#!/usr/bin/env python3
"""Which tile codec trains fastest? .omnia zstd levels/batching vs JPEG vs raw vs openslide.

    python benchmarks/bench_codecs.py SLIDE [SLIDE ...] [--tiles 3000] [--out codecs.json]

For each slide it cuts N tissue tiles (256 px, level 0), stores them in every
variant below, then measures what a training job actually pays:

  size            bytes on disk and the ratio against raw pixels
  encode          seconds to write the variant
  decode_1t       random-order single-thread decode, tiles/s, cold container
  train_Wn        a real ResNet-18 training loop (AMP, channels_last) fed by a
                  shuffled DataLoader with W workers, images/s; compared with the
                  same loop on pre-loaded GPU tensors (the ceiling) to give the
                  share of the GPU the data path can keep busy
  fidelity        (lossy only) PSNR and ResNet-18 feature cosine similarity against
                  the original tiles, i.e. how much the codec changes what a model sees

Variants:  raw (uint8 .npy memmap) · zstd L1/L3/L9 with 1 or 16 tiles per batch ·
jpeg q75/q90/q95 · openslide (decode the .svs on every read, the usual baseline).
Batching 16 tiles per compressed block improves ratio but a shuffled read has to
decompress the whole block for one tile: the benchmark shows that cost.

Prints a table and a JSON block between RESULTS_JSON_BEGIN/END markers.
"""
import argparse
import io
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TS = 256


# ── tile sampling ────────────────────────────────────────────────────────

def sample_tiles(slide_path: Path, n: int, seed: int = 0):
    """n tissue tile origins (level-0 coordinates) and the tiles themselves."""
    import openslide
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    lvl = slide.level_count - 1
    while lvl > 0 and min(slide.level_dimensions[lvl]) < 64:
        lvl -= 1
    tw, th = slide.level_dimensions[lvl]
    thumb = np.asarray(slide.read_region((0, 0), lvl, (tw, th)).convert("RGB")).astype(np.float32)
    sat = thumb.max(2) - thumb.min(2)
    ds = slide.level_downsamples[lvl]
    cell = max(1, int(round(TS / ds)))
    rng = np.random.default_rng(seed)
    cands = [(x, y) for y in range(0, th - cell, cell) for x in range(0, tw - cell, cell)
             if sat[y:y + cell, x:x + cell].mean() > 25]
    if not cands:
        raise RuntimeError(f"{slide_path.name}: no tissue found")
    rng.shuffle(cands)
    coords = []
    for x, y in cands[:n]:
        coords.append((min(int(x * ds), w - TS), min(int(y * ds), h - TS)))
    tiles = np.empty((len(coords), TS, TS, 3), np.uint8)
    for i, (x, y) in enumerate(coords):
        tiles[i] = np.asarray(slide.read_region((x, y), 0, (TS, TS)).convert("RGB"))
    return slide, coords, tiles


# ── variants: write + a reader factory ───────────────────────────────────

class RawReader:
    def __init__(self, path):
        self.path, self.m = path, None
    def get(self, i):
        if self.m is None:
            self.m = np.load(self.path, mmap_mode="r")
        return np.array(self.m[i])


class ZstdReader:
    def __init__(self, path):
        self.path, self.c = path, None
    def get(self, i):
        if self.c is None:
            from omnia_sdk import OmniaContainer
            self.c = OmniaContainer(self.path)
            self.c.open()
        return self.c.get_slice(i)


class JpegReader(ZstdReader):
    def get(self, i):
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(super().get(i).tobytes())).convert("RGB"))


class SlideReader:
    def __init__(self, slide_path, coords):
        self.p, self.coords, self.s = slide_path, coords, None
    def get(self, i):
        if self.s is None:
            import openslide
            self.s = openslide.OpenSlide(str(self.p))
        x, y = self.coords[i]
        return np.asarray(self.s.read_region((x, y), 0, (TS, TS)).convert("RGB"))


def build_variants(tiles, slide_path, coords, workdir: Path):
    from omnia_sdk import OmniaContainer
    from PIL import Image
    out = {}

    def timed(fn):
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t

    p = workdir / "raw.npy"
    enc = timed(lambda: np.save(p, tiles))
    out["raw"] = dict(kind="lossless", size=p.stat().st_size, encode_s=enc, reader=RawReader(p))

    for level, batch in [(1, 1), (3, 1), (3, 16), (9, 16)]:
        p = workdir / f"zstd_l{level}_b{batch}.omnia"
        enc = timed(lambda: OmniaContainer.write(p, iter(tiles), batch_size=batch, zstd_level=level))
        out[f"zstd_L{level}_b{batch}"] = dict(kind="lossless", size=p.stat().st_size, encode_s=enc, reader=ZstdReader(p))

    for q in (75, 90, 95):
        p = workdir / f"jpeg_q{q}.omnia"

        def gen(q=q):
            for t in tiles:
                b = io.BytesIO()
                Image.fromarray(t).save(b, "JPEG", quality=q)
                yield b.getvalue()
        enc = timed(lambda: OmniaContainer.write_native(p, gen(), pixel_codec="jpeg", native_codec="jpeg"))
        out[f"jpeg_q{q}"] = dict(kind="lossy", size=p.stat().st_size, encode_s=enc, reader=JpegReader(p))

    out["openslide"] = dict(kind="source", size=slide_path.stat().st_size, encode_s=0.0,
                            reader=SlideReader(slide_path, coords))
    return out


# ── measurements ─────────────────────────────────────────────────────────

def fresh(reader):
    """A reader with no open file handle. Handles must never be shared with
    DataLoader workers forked later: they would share one file position."""
    import copy
    r = copy.copy(reader)
    for attr in ("m", "c", "s"):
        if hasattr(r, attr):
            setattr(r, attr, None)
    return r


def decode_rate(reader, n, count=1500, seed=1):
    reader = fresh(reader)
    idx = np.random.default_rng(seed).integers(0, n, size=min(count, n * 2)).tolist()
    for i in idx[:20]:
        reader.get(i)
    t = time.perf_counter()
    for i in idx:
        reader.get(i)
    return len(idx) / (time.perf_counter() - t)


class DS:
    def __init__(self, reader, n):
        self.reader, self.n = reader, n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(np.ascontiguousarray(self.reader.get(i).transpose(2, 0, 1)))


def make_model(device):
    import torch
    import torchvision
    m = torchvision.models.resnet18(num_classes=6).to(device).to(memory_format=torch.channels_last)
    return m, torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)


def train_epochs(batches, device, steps_hint=None):
    """Run train steps over an iterable of uint8 NCHW batches; return images/s per epoch."""
    import torch
    model, opt = make_model(device)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
    rates = []
    for epoch in range(3):  # 1 warm-up + 2 measured
        seen, t0 = 0, None
        for x in batches():
            x = x.to(device, non_blocking=True).float()
            x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
            y = torch.randint(0, 6, (x.shape[0],), device=device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if t0 is None:  # start the clock after the first batch of the epoch
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                continue
            seen += x.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        if epoch > 0:
            rates.append(seen / (time.perf_counter() - t0))
    return statistics.median(rates)


def loader_rate(reader, n, workers, batch, device):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(DS(reader, n), batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=device == "cuda", drop_last=True, persistent_workers=workers > 0)
    try:
        return train_epochs(lambda: dl, device)
    finally:
        del dl


def gpu_ceiling(batch, device, steps=40):
    import torch
    x = torch.randint(0, 256, (batch, 3, TS, TS), dtype=torch.uint8, device=device)
    return train_epochs(lambda: (x for _ in range(steps)), device)


def fidelity(original, reader, n=200):
    """PSNR (dB) and ResNet-18 feature cosine similarity of decoded vs original tiles."""
    import torch
    import torchvision
    idx = list(range(min(n, len(original))))
    reader = fresh(reader)
    dec = np.stack([reader.get(i) for i in idx])
    org = original[idx]
    mse = ((dec.astype(np.float32) - org.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    psnr = float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-9))))
    try:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        pretrained = True
    except Exception:
        net = torchvision.models.resnet18()
        pretrained = False
    net.fc = torch.nn.Identity()
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def emb(a):
        x = (torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255 - mean) / std
        with torch.no_grad():
            return torch.cat([net(x[i:i + 50]) for i in range(0, len(x), 50)])
    cos = torch.nn.functional.cosine_similarity(emb(org), emb(dec)).mean().item()
    return {"psnr_db": round(psnr, 2), "feature_cosine": round(cos, 5), "pretrained_features": pretrained}


def bench_slide(slide_path: Path, ntiles, workers_list, batch, device):
    print(f"\n=== {slide_path.name} ===", flush=True)
    slide, coords, tiles = sample_tiles(slide_path, ntiles)
    n = len(tiles)
    print(f"{n} tissue tiles of {TS}px, raw {tiles.nbytes / 1e6:.0f} MB", flush=True)
    work = Path(tempfile.mkdtemp(prefix="omnia_codec_"))
    variants = build_variants(tiles, slide_path, coords, work)
    ceiling = gpu_ceiling(batch, device)
    print(f"GPU ceiling (no data loading): {ceiling:.0f} img/s", flush=True)
    rows = []
    for name, v in variants.items():
        row = {"variant": name, "kind": v["kind"], "size_mb": round(v["size"] / 1e6, 2),
               "ratio_vs_raw": round(tiles.nbytes / v["size"], 2) if name != "openslide" else None,
               "encode_s": round(v["encode_s"], 2),
               "decode_1t_tiles_s": round(decode_rate(v["reader"], n), 1)}
        for w in workers_list:
            r = loader_rate(v["reader"], n, w, batch, device)
            row[f"train_w{w}_img_s"] = round(r, 1)
            row[f"gpu_share_w{w}"] = round(min(r / ceiling, 1.0), 3)
        if v["kind"] == "lossy":
            row.update(fidelity(tiles, v["reader"]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"slide": slide_path.name, "tiles": n, "gpu_ceiling_img_s": round(ceiling, 1), "variants": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slides", nargs="+", type=Path)
    ap.add_argument("--tiles", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/codec_results.json"))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpus = os.cpu_count() or 1
    workers = args.workers or sorted({min(4, cpus), min(8, cpus)})
    env = {"python": platform.python_version(), "machine": platform.platform(), "cpus": cpus,
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "torch": torch.__version__, "batch": args.batch, "workers": workers}
    for mod in ("omnia_sdk", "openslide", "PIL", "zstandard"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    print(json.dumps(env), flush=True)

    results = []
    for s in args.slides:
        try:
            results.append(bench_slide(s, args.tiles, workers, args.batch, device))
        except Exception as e:  # one bad slide must not lose the others
            print(f"FAILED {s.name}: {e!r}", flush=True)
            results.append({"slide": s.name, "error": repr(e)})
    payload = {"environment": env, "results": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(payload, separators=(",", ":")))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
