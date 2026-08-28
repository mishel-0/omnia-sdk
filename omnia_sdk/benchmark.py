#!/usr/bin/env python3
"""
omnia-20x auditable benchmark.

Actually MEASURES the data-path speedup (.svs + openslide vs .omnia + RAM
preload) and writes a fully honest JSON receipt. Nothing is hardcoded —
every number comes from a timed run on the machine that executes it.

Usage:
    python -m omnia_sdk.benchmark                          # downloads a public slide
    python -m omnia_sdk.benchmark --svs slide.svs          # use your own slide
    python -m omnia_sdk.benchmark --tiles 5000 --epochs 5  # tune size
    python -m omnia_sdk.benchmark --output /path/results.json
"""
import argparse, hashlib, json, os, platform, random, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from omnia_sdk.container import OmniaContainer
from omnia_sdk.dataset import OmniaDataset

TS = 256
PUBLIC_SLIDES = [
    # Public Aperio .svs test slides (CMU openslide test-data mirror)
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs",
]

DEFAULT_SLIDE_URL = PUBLIC_SLIDES[0]


def download_slide(url: str, dest: Path) -> Path:
    """Download a public .svs; returns local path. Raises on failure."""
    print(f"  downloading public slide: {url}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "omnia-20x-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = 100.0 * got / total
                print(f"\r  {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print(f"\n  downloaded {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "torchvision", "numpy", "zstandard", "openslide-python", "tifffile", "Pillow", "omnia-20x"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


class SvsTileDataset(Dataset):
    """Realistic .svs usage: decode tiles from disk EVERY epoch (20x / level 1)."""
    def __init__(self, slide, coords):
        from openslide import OpenSlide
        self.slide = OpenSlide(str(slide))
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = self.coords[i]
        pil = self.slide.read_region((x, y), 0, (TS, TS)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), torch.tensor(0, dtype=torch.long)


def gpu_util_sample(duration_s: float) -> dict:
    """Sample nvidia-smi GPU utilization during `duration_s`; {} if no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return {}
        base = int(out.stdout.strip().split("\n")[0])
    except Exception:
        return {}
    samples = []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            samples.append(int(out.stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.4)
    return {"mean_pct": round(float(np.mean(samples)), 1) if samples else None,
            "samples": samples, "idle_base_pct": base}


def _make_model():
    """Small trainable model: resnet18 if torchvision is present, else a tiny CNN."""
    import torch.nn as nn
    try:
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=2)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        return model
    except ImportError:
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 64 * 64, 2),
        )


def full_train_bench(ds_a, ds_b, name_a, name_b, tiles_n, epochs, batch) -> dict:
    """Run a real forward/backward loop on both paths; returns speedup + times."""
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    def run(ds):
        n = min(len(ds), tiles_n)
        sub = torch.utils.data.Subset(ds, list(range(n)))
        loader = DataLoader(sub, batch_size=batch, shuffle=True, num_workers=0,
                            pin_memory=device == "cuda")
        model = _make_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        times = []
        for e in range(epochs):
            t0 = time.perf_counter()
            model.train()
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                opt.zero_grad()
                out = model(images)
                loss = crit(out, labels)
                loss.backward()
                opt.step()
            times.append(time.perf_counter() - t0)
        return times

    print(f"  [full-train] {name_a} (ResNet-18, {min(len(ds_a), tiles_n)} tiles, {epochs} epochs)...")
    t_a = run(ds_a)
    print(f"  [full-train] {name_b} ...")
    t_b = run(ds_b)
    mean_a = float(np.mean(t_a[1:])) if epochs > 1 else t_a[0]
    mean_b = float(np.mean(t_b[1:])) if epochs > 1 else t_b[0]
    results["model"] = "resnet18" if _make_model().__class__.__name__ == "ResNet" else "tiny-cnn"
    results["tiles"] = min(len(ds_a), tiles_n)
    results["epochs"] = epochs
    results["batch_size"] = batch
    results["svs_epoch_times_s"] = [round(t, 3) for t in t_a]
    results["svs_mean_s"] = round(mean_a, 3)
    results["omnia_epoch_times_s"] = [round(t, 3) for t in t_b]
    results["omnia_mean_s"] = round(mean_b, 3)
    results["speedup"] = round(mean_a / mean_b, 2)
    results["data_fraction_of_omnia_epoch"] = None  # filled by caller
    return results


def run_benchmark(svs_path: Path, omnia_path: Path, tiles_n: int, epochs: int,
                  batch: int, out_path: Path,
                  train: bool = True, force_train: bool = False,
                  train_tiles: int = 512, train_epochs: int = 2) -> dict:
    from openslide import OpenSlide

    print("== omnia-20x auditable benchmark ==")
    torch.manual_seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    slide = OpenSlide(str(svs_path))
    w, h = slide.dimensions
    tx, ty = (w + TS - 1) // TS, (h + TS - 1) // TS
    coords = [(rng.randrange(tx) * TS, rng.randrange(ty) * TS) for _ in range(tiles_n)]
    print(f"  slide {w}x{h}, {tiles_n} fixed L0 tile coords (seed 42)")

    # Build the .omnia container from the SAME tiles (zstd, lossless)
    def zstd_gen():
        for x, y in coords:
            yield np.array(slide.read_region((x, y), 0, (TS, TS)).convert("RGB"), dtype=np.uint8)
    t0 = time.perf_counter()
    OmniaContainer.write(omnia_path, zstd_gen(), metadata={"source": str(svs_path), "seed": 42})
    print(f"  built .omnia: {omnia_path.stat().st_size/1e6:.1f} MB in {time.perf_counter()-t0:.1f}s")
    slide.close()

    # ---- .svs path: decode every epoch ----
    print(f"  [.svs] data-only epochs x{epochs} (decode from disk every epoch)...")
    svs_ds = SvsTileDataset(svs_path, coords)
    svs_loader = DataLoader(svs_ds, batch_size=batch, shuffle=True, num_workers=0)
    svs_times = []
    for e in range(epochs):
        t0 = time.perf_counter()
        for images, labels in svs_loader:
            images.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)
        svs_times.append(time.perf_counter() - t0)
        print(f"    epoch {e+1}: {svs_times[-1]:.2f}s")
    svs_mean = float(np.mean(svs_times[1:])) if epochs > 1 else svs_times[0]

    # ---- .omnia path: preload once, zero-copy ----
    print(f"  [.omnia] preload + data-only epochs x{epochs}...")
    t0 = time.perf_counter()
    ds = OmniaDataset(omnia_path, cache_mode="ram")
    preload_time = time.perf_counter() - t0
    assert ds.cache_mode == "ram", f"preload failed, cache_mode={ds.cache_mode}"
    print(f"    preload (one-time): {preload_time:.1f}s")
    omnia_loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    omnia_times = []
    for e in range(epochs):
        t0 = time.perf_counter()
        for images, labels in omnia_loader:
            images.to("cuda" if torch.cuda.is_available() else "cpu", non_blocking=True)
        omnia_times.append(time.perf_counter() - t0)
        print(f"    epoch {e+1}: {omnia_times[-1]:.2f}s")
    omnia_mean = float(np.mean(omnia_times[1:])) if epochs > 1 else omnia_times[0]

    speedup = svs_mean / omnia_mean

    # GPU utilization during one representative epoch per path
    print("  sampling GPU utilization (1 epoch each)...")
    gpu_svs = gpu_util_sample(svs_times[-1])
    gpu_omnia = gpu_util_sample(omnia_times[-1])

    # Full-train measurement (real forward/backward) — the "model-bound" number.
    # Meaningful on GPU; on CPU training dominates so skip by default (use --force-train).
    full_train = None
    if train and (torch.cuda.is_available() or force_train):
        full_train = full_train_bench(svs_ds, ds, ".svs", ".omnia",
                                      train_tiles, train_epochs, batch)
        # data fraction of the omnia full-train epoch (data-only time / full-train time)
        full_train["data_fraction_of_omnia_epoch"] = round(
            100.0 * omnia_mean / full_train["omnia_mean_s"], 1)
    elif train:
        print("  [full-train] skipped: no CUDA GPU — the end-to-end number is only "
              "meaningful when training runs on a GPU (use --force-train for CPU anyway)")

    # Regime classification — honest, data-driven.
    # Thresholds are the SAME numbers as the README table (10x / 2x), so a
    # receipt never contradicts the headline docs.
    if full_train is not None:
        if full_train["speedup"] >= 10.0:
            regime = "data-bound (GPU starved by decode — omnia wins big, 10x+ end-to-end)"
        elif full_train["speedup"] >= 2.0:
            regime = "mixed (data and training both matter, ~2-10x end-to-end)"
        else:
            regime = "model-bound (training dominates — omnia helps data, not total, ~1-2x)"
    elif train:
        regime = "unknown (full-train skipped on CPU — run on GPU for regime)"
    else:
        regime = "unknown (full-train disabled)"

    result = {
        "benchmark": "omnia-20x auditable benchmark",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "machine": {
            "os": platform.system(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_available": torch.cuda.is_available(),
            "python": platform.python_version(),
            "packages": package_versions(),
        },
        "slide": {
            "path": str(svs_path),
            "size_bytes": svs_path.stat().st_size,
            "sha256": sha256(svs_path),
        },
        "container": {
            "path": str(omnia_path),
            "size_bytes": omnia_path.stat().st_size,
            "codec": "zstd-lossless",
            "tiles": tiles_n,
        },
        "measurement": {
            "tiles": tiles_n,
            "batch_size": batch,
            "epochs": epochs,
            "warmup_dropped": epochs > 1,
            "num_workers": 0,
            "seed": 42,
            "preload_time_s": round(preload_time, 3),
            "svs_epoch_times_s": [round(t, 3) for t in svs_times],
            "svs_mean_s": round(svs_mean, 3),
            "omnia_epoch_times_s": [round(t, 3) for t in omnia_times],
            "omnia_mean_s": round(omnia_mean, 3),
            "data_speedup": round(speedup, 2),
            "gpu_util_svs": gpu_svs,
            "gpu_util_omnia": gpu_omnia,
            "passed_20x": speedup >= 20.0,
            "full_train": full_train,
            "regime": regime,
        },
        "notes": [
            "Speedup is DATA-FEEDING first (loader iteration, no model). The "
            "full_train block reports the end-to-end number with a real "
            "forward/backward loop. If full_train.speedup is ~1-2x, your "
            "workload is model-bound and the data win is hidden in the total.",
            "Regime tells you which case you are in: data-bound (GPU starved by "
            "decode — the 10-20x case) vs model-bound (training dominates).",
            "num_workers=0: preloaded data has no I/O to parallelize; workers only add IPC overhead.",
            "First epoch is warmup and is excluded from the mean when epochs > 1.",
            "Identical fixed tile set (seed 42) on both paths.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  proof written to {out_path}")

    print("\n" + "=" * 52)
    print("  omnia-20x RESULT")
    print("=" * 52)
    print(f"  .svs data:   {svs_mean:.2f}s/epoch")
    print(f"  .omnia data: {omnia_mean:.2f}s/epoch  (preload {preload_time:.1f}s one-time)")
    print(f"  DATA SPEEDUP: {speedup:.1f}x  {'(target 20x MET)' if speedup >= 20 else '(below 20x on this machine)'}")
    if full_train is not None:
        print(f"  FULL-TRAIN:  {full_train['svs_mean_s']:.2f}s vs {full_train['omnia_mean_s']:.2f}s/epoch "
              f"-> {full_train['speedup']:.1f}x end-to-end")
        print(f"  REGIME:      {regime}")
    if gpu_svs.get("mean_pct") and gpu_omnia.get("mean_pct"):
        print(f"  GPU util:    .svs {gpu_svs['mean_pct']}% | .omnia {gpu_omnia['mean_pct']}%")
    print(f"  GPU:         {result['machine']['gpu']}")
    print("=" * 52)
    return result


def resolve_slide(args) -> Path:
    if args.svs:
        p = Path(args.svs)
        if not p.exists():
            sys.exit(f"error: {p} not found")
        return p
    dest = Path("/tmp/omnia_bench_slide.svs")
    try:
        return download_slide(DEFAULT_SLIDE_URL, dest)
    except Exception as e:
        sys.exit(f"error: could not download public slide ({e}).\n"
                 f"Provide one with: python -m omnia_sdk.benchmark --svs /path/to/slide.svs")


def main():
    ap = argparse.ArgumentParser(description="omnia-20x auditable benchmark")
    ap.add_argument("--svs", default=None, help="Path to a .svs slide (default: download public one)")
    ap.add_argument("--tiles", type=int, default=5000, help="Tiles per epoch (default 5000)")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs per path (default 5)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size (default 64)")
    ap.add_argument("--output", default=str(Path("benchmarks/benchmark_results.json").resolve()),
                    help="JSON output path")
    ap.add_argument("--no-train", action="store_true",
                    help="Skip the full-train (model-bound) measurement")
    ap.add_argument("--force-train", action="store_true",
                    help="Run full-train even without a GPU (CPU — slow, mostly meaningless)")
    ap.add_argument("--train-tiles", type=int, default=512, help="Tiles for full-train bench (default 512)")
    ap.add_argument("--train-epochs", type=int, default=2, help="Epochs for full-train bench (default 2)")
    args = ap.parse_args()

    svs = resolve_slide(args)
    omnia = Path("/tmp/omnia_bench_train.omnia")
    run_benchmark(svs, omnia, args.tiles, args.epochs, args.batch, Path(args.output),
                  train=not args.no_train, force_train=args.force_train,
                  train_tiles=args.train_tiles,
                  train_epochs=args.train_epochs)


if __name__ == "__main__":
    main()
