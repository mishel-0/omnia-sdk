#!/usr/bin/env python3
"""Per-format benchmark: .omnia against each format's usual reader, under real training.

    omnia benchmark formats SAMPLE [SAMPLE ...] [--note "source, licence" ...] [--out results.json]

For every sample (a slide, a DICOM series or DICOM whole-slide folder, or a NIfTI file) it
converts to .omnia, then measures what a training job pays, for the native reader(s) and
for .omnia, on the same items:

  convert_s / size      conversion time and bytes (source vs .omnia)
  decode_1t             random-order, single-thread read, items/s, cold container
  train_w{N}            a ResNet-18 training loop (AMP) fed by a shuffled DataLoader with N
                        workers, items/s, and the share of the GPU-only ceiling it reaches
  bit_exact             for lossless volumes: slices identical to the native reader's

Slices of a volume are min-max scaled and resized to 224 px on the GPU; slide tiles use the
usual RGB normalisation. Slides are converted twice: lossless zstd and JPEG q90.
Where .omnia loses, the table shows it: this is data, not a verdict.
"""
import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import OmniaContainer, convert
from .codec_bench import (JpegReader, ZstdReader, fresh, decode_rate, gpu_ceiling,
                          train_epochs, TS)
from .converters import detect_format, _dicom_files

MAX_ITEMS = 2000


# ── items and readers ────────────────────────────────────────────────────

class Items:
    """Maps item j to a reader index; readers expose .get(i) -> ndarray (HW, HWC)."""
    def __init__(self, reader, index):
        self.reader, self.index = reader, list(index)
    def __len__(self):
        return len(self.index)
    def get(self, j):
        return self.reader.get(self.index[j])
    def fresh(self):
        return Items(fresh(self.reader), self.index)


def fresh_items(it):
    return it.fresh()


class NiftiProxy:      # nibabel reading one slice from the gzip file, as most pipelines do
    def __init__(self, path):
        self.path, self.img = path, None
    def get(self, i):
        import nibabel as nib
        if self.img is None:
            self.img = nib.load(str(self.path))
        return np.asanyarray(self.img.dataobj[:, :, i])


class NiftiRam:        # nibabel with the whole volume decompressed into RAM first
    def __init__(self, path):
        import nibabel as nib
        self.vol = np.asanyarray(nib.load(str(path)).dataobj)
    def get(self, i):
        return self.vol[:, :, i]


class DicomFiles:      # one .dcm per slice, read with pydicom on demand
    def __init__(self, files, order):
        self.files, self.order = files, order
    def get(self, i):
        import pydicom
        return pydicom.dcmread(str(self.files[self.order[i]])).pixel_array


class OpenSlideTiles:  # the same tiles .omnia stored, read from the slide with read_region
    def __init__(self, path, level, tw, th, tx, ds):
        self.path, self.level, self.tw, self.th, self.tx, self.ds = path, level, tw, th, tx, ds
        self.s = None
    def get(self, i):
        import openslide
        if self.s is None:
            self.s = openslide.OpenSlide(str(self.path))
        x, y = (i % self.tx) * self.tw, (i // self.tx) * self.th
        return np.asarray(self.s.read_region((int(x * self.ds), int(y * self.ds)), self.level,
                                             (self.tw, self.th)).convert("RGB"))


class WsiDicomTiles:
    def __init__(self, path, level_id, tx):
        self.path, self.level_id, self.tx, self.s = path, level_id, tx, None
    def get(self, i):
        from wsidicom import WsiDicom
        if self.s is None:
            self.s = WsiDicom.open(str(self.path))
        return np.asarray(self.s.read_tile(self.level_id, (i % self.tx, i // self.tx)).convert("RGB"))


# ── GPU-side preparation ─────────────────────────────────────────────────

def make_prep(kind, device):
    import torch
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if kind == "rgb":
        return lambda x: (x.float() / 255 - mean) / std
    def volume(x):
        x = x.float()
        lo, hi = x.amin((1, 2, 3), keepdim=True), x.amax((1, 2, 3), keepdim=True)
        x = (x - lo) / (hi - lo + 1e-6)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        return (x - mean) / std
    return volume


class DS:
    def __init__(self, items, kind):
        self.items, self.kind = items, kind
    def __len__(self):
        return len(self.items)
    def __getitem__(self, j):
        import torch
        a = np.asarray(self.items.get(j))
        if self.kind == "rgb":
            return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        if a.dtype == np.uint16:
            a = a.astype(np.int32)
        return torch.from_numpy(np.ascontiguousarray(a))[None]


def loader_rate(items, kind, workers, batch, device):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(DS(items, kind), batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=device == "cuda", drop_last=True, persistent_workers=workers > 0)
    try:
        return train_epochs(lambda: dl, device, prep=make_prep(kind, device))
    finally:
        del dl


def ceiling(items, kind, batch, device):
    import torch
    sample = torch.from_numpy(np.asarray(fresh(items).get(0)))
    if sample.dtype == torch.uint16:
        sample = sample.to(torch.int32)
    if kind == "volume":
        sample = sample[None]
    elif kind == "rgb":
        sample = sample.permute(2, 0, 1)
    x = sample[None].repeat(batch, 1, 1, 1).to(device)
    return train_epochs(lambda: (x for _ in range(40)), device, prep=make_prep(kind, device))


# ── one sample ───────────────────────────────────────────────────────────

def measure(name, items, kind, workers_list, batch, device, ceil, n_decode=800):
    items = fresh(items)  # never hand DataLoader workers a reader that already holds open files
    row = {"reader": name, "items": len(items),
           "decode_1t_items_s": round(decode_rate(items, len(items), count=n_decode), 1)}
    for w in workers_list:
        r = loader_rate(fresh(items), kind, w, batch, device)
        row[f"train_w{w}_items_s"] = round(r, 1)
        row[f"gpu_share_w{w}"] = round(min(r / ceil, 1.0), 3)
    return row


def tissue_indices(container, n_max, seed=0):
    """Interior tiles of the first stored level that contain tissue (not blank glass)."""
    lvl = container.levels[0]
    tx, ty, start = lvl["tx"], lvl["ty"], lvl["start"]
    cand = []
    for y in range(ty - 1):
        for x in range(tx - 1):
            cand.append(start + y * tx + x)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    keep = []
    for i in cand:
        t = container.get_slice(i)
        if t.ndim == 3 and t.std() > 12:
            keep.append(i)
            if len(keep) >= n_max:
                break
    return keep


def bench_one(path: Path, note: str, workers_list, batch, device):
    fmt = detect_format(path)
    print(f"\n=== {path.name} ({fmt}) ===", flush=True)
    work = Path(tempfile.mkdtemp(prefix="omnia_fmt_"))
    src_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
    res = {"sample": path.name, "format": fmt, "note": note, "source_mb": round(src_bytes / 1e6, 2), "variants": []}

    variants = [("omnia", {})]
    if fmt == "whole-slide":
        variants.append(("omnia_jpeg_q90", {"codec": "jpeg", "quality": 90}))
    kind = "rgb" if fmt in ("whole-slide", "dicom-wsi") else "volume"

    omnia_paths = {}
    for vname, kw in variants:
        out = work / f"{vname}.omnia"
        t = time.perf_counter()
        extra = {}
        if fmt == "whole-slide":
            import openslide
            extra["min_level"] = 1 if openslide.OpenSlide(str(path)).level_count > 1 else 0
        elif fmt == "dicom-wsi":
            extra["min_level"] = 1
        stats = convert(path, out, **{**extra, **kw})
        omnia_paths[vname] = out
        res["variants"].append({"variant": vname, "convert_s": round(time.perf_counter() - t, 2),
                                "omnia_mb": round(out.stat().st_size / 1e6, 2)})
        print(f"{vname}: {res['variants'][-1]}", flush=True)

    base = omnia_paths["omnia"]
    with OmniaContainer(base) as c:
        if kind == "rgb":
            idx = tissue_indices(c, MAX_ITEMS)
            lvl0 = c.levels[0]
        else:
            idx = list(range(c.num_slices))
    if not idx:
        raise RuntimeError("no usable items")

    # readers: native ones first, then the .omnia ones, over the same items
    natives = []
    if fmt == "nifti":
        natives = [("nibabel (slice from .nii.gz)", Items(NiftiProxy(path), idx)),
                   ("nibabel (volume in RAM)", Items(NiftiRam(path), idx))]
    elif fmt == "dicom-series":
        files = _dicom_files(path)
        import pydicom
        order = sorted(range(len(files)), key=lambda i: (
            float(getattr(pydicom.dcmread(str(files[i]), stop_before_pixels=True), "ImagePositionPatient", [0, 0, i])[2]),
            i))
        natives = [("pydicom (one file per slice)", Items(DicomFiles(files, order), idx))]
    elif fmt == "whole-slide":
        import openslide
        s = openslide.OpenSlide(str(path))
        # container level 0 is slide level `min_level`=1; find it by dimensions
        lv = next(i for i in range(s.level_count) if s.level_dimensions[i] == (lvl0["w"], lvl0["h"]))
        local = [i - lvl0["start"] for i in idx]
        natives = [("openslide read_region", Items(OpenSlideTiles(path, lv, lvl0["tw"], lvl0["th"], lvl0["tx"],
                                                                  s.level_downsamples[lv]), local))]
        idx_native_local = local
    elif fmt == "dicom-wsi":
        from wsidicom import WsiDicom
        w = WsiDicom.open(str(path))
        level = list(w.pyramid)[1] if len(list(w.pyramid)) > 1 else list(w.pyramid)[0]
        tx = (level.size.width + level.tile_size.width - 1) // level.tile_size.width
        local = [i - lvl0["start"] for i in idx]
        natives = [("wsidicom read_tile", Items(WsiDicomTiles(path, level.level, tx), local))]
        w.close()

    omnia_readers = []
    for vname, p in omnia_paths.items():
        cls = JpegReader if vname.endswith("jpeg_q90") else ZstdReader
        omnia_readers.append((vname, Items(cls(p), idx)))

    ref = natives[0][1] if natives else omnia_readers[0][1]
    ceil = ceiling(ref, kind, batch, device)
    res["gpu_ceiling_items_s"] = round(ceil, 1)
    print(f"GPU ceiling: {ceil:.0f} items/s", flush=True)

    if kind == "volume" and natives:
        with OmniaContainer(base) as c:
            res["bit_exact"] = all(np.array_equal(natives[0][1].get(j), c.get_slice(idx[j]))
                                   for j in (0, len(idx) // 2, len(idx) - 1))
    elif kind == "rgb" and natives and "omnia" in omnia_paths:
        with OmniaContainer(base) as c:
            res["tiles_identical_to_native"] = all(
                np.array_equal(natives[0][1].get(j), c.get_slice(idx[j])) for j in range(min(5, len(idx))))

    res["readers"] = []
    for name, items in natives + omnia_readers:
        row = measure(name, items, kind, workers_list, batch, device, ceil)
        res["readers"].append(row)
        print(json.dumps(row), flush=True)
    return res


def print_table(results, workers_list):
    w = workers_list[0]
    for r in results:
        if "error" in r:
            print(f"\n{r['sample']}: FAILED {r['error']}")
            continue
        print(f"\n{r['sample']}  [{r['format']}]  source {r['source_mb']} MB   GPU ceiling {r['gpu_ceiling_items_s']} items/s")
        for v in r["variants"]:
            print(f"  {v['variant']}: {v['omnia_mb']} MB, converted in {v['convert_s']} s")
        if "bit_exact" in r:
            print(f"  bit-exact vs native: {r['bit_exact']}")
        if "tiles_identical_to_native" in r:
            print(f"  tiles identical to native: {r['tiles_identical_to_native']}")
        print(f"  {'reader':34} {'decode/s':>9} " + " ".join(f"{'train w'+str(x):>10} {'GPU%':>5}" for x in workers_list))
        for row in r["readers"]:
            cells = " ".join(f"{row[f'train_w{x}_items_s']:>10} {row[f'gpu_share_w{x}'] * 100:>4.0f}%" for x in workers_list)
            print(f"  {row['reader']:34} {row['decode_1t_items_s']:>9} {cells}")


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--note", action="append", default=[], help="source + licence, one per sample, in order")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("format_results.json"))
    args = ap.parse_args(argv)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpus = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else sorted({min(4, cpus), min(8, cpus)})
    notes = args.note + [""] * (len(args.samples) - len(args.note))
    env = {"python": platform.python_version(), "machine": platform.platform(), "cpus": cpus,
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__,
           "batch": args.batch, "workers": workers, "model": "ResNet-18, AMP, channels_last"}
    for mod in ("numpy", "nibabel", "pydicom", "openslide", "wsidicom", "zstandard", "PIL"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    print(json.dumps(env), flush=True)

    results = []
    for p, n in zip(args.samples, notes):
        try:
            results.append(bench_one(p, n, workers, args.batch, device))
        except Exception as e:  # one failing format must not lose the others
            import traceback
            traceback.print_exc()
            print(f"FAILED {p.name}: {e!r}", flush=True)
            results.append({"sample": p.name, "error": repr(e), "note": n})
    payload = {"environment": env, "results": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print_table(results, workers)
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(payload, separators=(",", ":")))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
