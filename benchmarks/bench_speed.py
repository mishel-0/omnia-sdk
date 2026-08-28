#!/usr/bin/env python3
"""
Deep speed bug test: .svs + openslide vs .omnia (zstd / jpeg), RAM preload.

Measures per-epoch DATA FEEDING (the part .omnia makes 10x faster):
  - .svs:  openslide decodes every tile EVERY epoch (realistic usage)
  - .omnia: preload once at init, zero-copy views per epoch

Also: num_workers comparison, preload cost, and a full ResNet-18
train-step sanity (if torchvision is available).

Usage: python bench_speed.py [--tiles 10000] [--epochs 3]
"""
import argparse, io, random, time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from openslide import OpenSlide
from PIL import Image
from omnia_sdk.container import OmniaContainer
from omnia_sdk.dataset import OmniaDataset

SVS = "/Users/misheladnan/Desktop/omnia-20x/data/TCGA.svs"
TS = 256


class SvsSubset(Dataset):
    """Realistic .svs usage: decode tiles on every epoch (no preload)."""
    def __init__(self, slide, coords):
        self.slide = slide
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = self.coords[i]
        pil = self.slide.read_region((x, y), 0, (TS, TS)).convert("RGB")
        arr = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), torch.tensor(0, dtype=torch.long)


def build_containers(slide, coords, jpeg_q=85):
    def zstd_gen():
        for x, y in coords:
            yield np.array(slide.read_region((x, y), 0, (TS, TS)).convert("RGB"), dtype=np.uint8)

    def jpeg_gen():
        for x, y in coords:
            buf = io.BytesIO()
            Image.fromarray(np.array(slide.read_region((x, y), 0, (TS, TS)).convert("RGB"))).save(
                buf, "JPEG", quality=jpeg_q)
            yield buf.getvalue()

    t0 = time.perf_counter()
    OmniaContainer.write("/tmp/bench.zstd.omnia", zstd_gen(), metadata={"source": "bench"})
    tz = time.perf_counter() - t0
    t0 = time.perf_counter()
    OmniaContainer.write_native("/tmp/bench.jpeg.omnia", jpeg_gen(),
                                metadata={"source": "bench"}, pixel_codec="jpeg", native_codec="jpeg")
    tj = time.perf_counter() - t0
    print(f"  build: zstd {tz:.1f}s, jpeg {tj:.1f}s")
    print(f"  sizes: zstd {Path('/tmp/bench.zstd.omnia').stat().st_size/1e6:.1f} MB, "
          f"jpeg {Path('/tmp/bench.jpeg.omnia').stat().st_size/1e6:.1f} MB")


def bench_epochs(ds, name, batch=64, epochs=3, nw=0):
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=nw)
    times = []
    for e in range(epochs):
        t0 = time.perf_counter()
        for images, labels in loader:
            pass  # data feeding only
        times.append(time.perf_counter() - t0)
    avg = float(np.mean(times[1:])) if epochs > 1 else times[0]
    print(f"  {name:24s}: {avg:6.2f}s/epoch  ({', '.join(f'{t:.1f}' for t in times)})")
    return avg


def train_sanity(ds, name, epochs=2, batch=64, max_tiles=1024):
    """Full forward/backward sanity on a small subset (CPU)."""
    try:
        import torch.nn as nn
        from torchvision.models import resnet18
    except ImportError:
        print("  train sanity skipped (no torchvision)")
        return
    n = min(len(ds), max_tiles)
    sub = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=batch, shuffle=True)
    model = resnet18(weights=None, num_classes=2)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for e in range(epochs):
        for images, labels in loader:
            out = model(images)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
    dt = (time.perf_counter() - t0) / epochs
    print(f"  {name} train sanity ({n} tiles, CPU): {dt:.1f}s/epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    print(f"== deep speed test: {args.tiles} tiles, {args.epochs} epochs, batch 64 ==")
    rng = random.Random(7)
    slide = OpenSlide(SVS)
    w, h = slide.dimensions
    tx, ty = (w + TS - 1) // TS, (h + TS - 1) // TS
    coords = [(rng.randrange(tx) * TS, rng.randrange(ty) * TS) for _ in range(args.tiles)]
    print(f"  slide {w}x{h}, sampled {args.tiles} L0 tile coords")

    build_containers(slide, coords)
    slide.close()

    print("\n== .svs + openslide (decode EVERY epoch) ==")
    svs = bench_epochs(SvsSubset(OpenSlide(SVS), coords), ".svs nw=0", epochs=args.epochs)

    print("\n== .omnia zstd (preload once, zero-copy) ==")
    t0 = time.perf_counter()
    dz = OmniaDataset("/tmp/bench.zstd.omnia", cache_mode="ram")
    z_pre = time.perf_counter() - t0
    tz = bench_epochs(dz, "zstd nw=0", epochs=args.epochs)
    tz2 = bench_epochs(dz, "zstd nw=2", epochs=args.epochs, nw=2)

    print("\n== .omnia jpeg (preload once, zero-copy) ==")
    t0 = time.perf_counter()
    dj = OmniaDataset("/tmp/bench.jpeg.omnia", cache_mode="ram")
    j_pre = time.perf_counter() - t0
    tj = bench_epochs(dj, "jpeg nw=0", epochs=args.epochs)

    print("\n== results ==")
    print(f"  preload (one-time): zstd {z_pre:.1f}s, jpeg {j_pre:.1f}s")
    print(f"  speedup vs .svs:    zstd {svs/tz:.1f}x, jpeg {svs/tj:.1f}x")
    print(f"  zstd nw2 vs nw0:    {tz2/tz:.2f}x (nw2 slower if >1)")

    print("\n== train sanity (full forward/backward) ==")
    train_sanity(SvsSubset(OpenSlide(SVS), coords[:1024]), ".svs")
    train_sanity(dz, ".omnia zstd")


if __name__ == "__main__":
    main()
