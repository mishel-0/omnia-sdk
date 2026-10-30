#!/usr/bin/env python3
"""Per-format benchmark: .omnia against each format's normal reader.

    python benchmarks/bench_formats.py SAMPLE [SAMPLE ...] --out results.json

For every sample it records, on the machine it runs on:
  * conversion time and size (source vs .omnia)
  * sequential read of every slice/tile into RAM, native reader vs .omnia
  * random-access read of N slices/tiles, native reader vs .omnia
  * whether the first and last slice are bit-identical to the native reader

Numbers are medians of --repeats runs after one warm-up, and the file records
the hardware and library versions. Where .omnia is slower or larger it says so;
the output is data, not a verdict. Samples must be public with a known licence
(record it in the --note for each one).
"""
import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omnia_sdk import OmniaContainer, convert  # noqa: E402
from omnia_sdk.converters import detect_format  # noqa: E402


def _median_time(fn, repeats):
    fn()  # warm-up (page cache, imports)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def _native_reader(path: Path, fmt: str):
    """Return (n_items, read_fn(i)) for the format's usual library."""
    if fmt == "nifti":
        import nibabel as nib
        img = nib.load(str(path))
        n = img.shape[2]
        return n, lambda i: np.asanyarray(img.dataobj[:, :, i])
    if fmt == "dicom-series":
        import pydicom
        from omnia_sdk.converters import _dicom_files
        files = _dicom_files(path)
        return len(files), lambda i: pydicom.dcmread(str(files[i])).pixel_array
    if fmt == "whole-slide":
        import openslide
        slide = openslide.OpenSlide(str(path))
        w, h = slide.level_dimensions[1] if slide.level_count > 1 else slide.dimensions
        lvl = 1 if slide.level_count > 1 else 0
        ds = slide.level_downsamples[lvl]
        cols, rows = (w + 255) // 256, (h + 255) // 256
        def read(i):
            x, y = (i % cols) * 256, (i // cols) * 256
            return np.asarray(slide.read_region((int(x * ds), int(y * ds)), lvl, (256, 256)).convert("RGB"))
        return cols * rows, read
    if fmt == "dicom-wsi":
        from wsidicom import WsiDicom
        slide = WsiDicom.open(str(path))
        lv = slide.pyramid[1] if len(slide.pyramid) > 1 else slide.pyramid[0]
        tx = (lv.size.width + lv.tile_size.width - 1) // lv.tile_size.width
        ty = (lv.size.height + lv.tile_size.height - 1) // lv.tile_size.height
        return tx * ty, lambda i: np.asarray(slide.read_tile(lv.level, (i % tx, i // tx)).convert("RGB"))
    raise ValueError(fmt)


def bench_one(path: Path, note: str, repeats: int, random_n: int) -> dict:
    fmt = detect_format(path)
    out = Path(tempfile.mkdtemp()) / "x.omnia"
    t = time.perf_counter()
    stats = convert(path, out)
    convert_s = time.perf_counter() - t

    n_native, read_native = _native_reader(path, fmt)
    with OmniaContainer(out) as c:
        n_omnia = c.num_slices
        # Whole-slide containers keep levels >= min_level; compare like for like
        # on the first `n` items the two have in common only for volumes.
        n = min(n_native, n_omnia)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=min(random_n, n)).tolist()
        seq = list(range(n))

        res = {
            "sample": path.name, "format": fmt, "note": note,
            "source_mb": round(stats["source_bytes"] / 1e6, 2) if "source_bytes" in stats else round(stats.get("svs_bytes", 0) / 1e6, 2),
            "omnia_mb": round(stats["compressed_bytes"] / 1e6, 2),
            "items_compared": n,
            "convert_s": round(convert_s, 2),
        }
        raw = {
            "sequential": (_median_time(lambda: [read_native(i) for i in seq], repeats),
                           _median_time(lambda: [c.get_slice(i) for i in seq], repeats)),
            "random": (_median_time(lambda: [read_native(i) for i in idx], repeats),
                       _median_time(lambda: [c.get_slice(i) for i in idx], repeats)),
        }
        for k, (nat, om) in raw.items():
            res[f"{k}_s"] = {"native": round(nat, 4), "omnia": round(om, 4)}
            res[f"speedup_{k}"] = round(nat / om, 2) if om > 0 else None
        if fmt in ("nifti", "dicom-series"):
            res["bit_exact"] = all(np.array_equal(read_native(i), c.get_slice(i)) for i in (0, n // 2, n - 1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--note", action="append", default=[], help="source + licence, one per sample, in order")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--random", type=int, default=200, help="random reads per sample")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/format_results.json"))
    args = ap.parse_args()

    notes = args.note + [""] * (len(args.samples) - len(args.note))
    results = [bench_one(p, n, args.repeats, args.random) for p, n in zip(args.samples, notes)]
    env = {"python": platform.python_version(), "machine": platform.platform(), "processor": platform.processor()}
    for mod in ("numpy", "nibabel", "pydicom", "openslide", "wsidicom", "zstandard"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    args.out.write_text(json.dumps({"environment": env, "results": results}, indent=2))
    print(f"{'sample':32} {'fmt':13} {'src MB':>8} {'omnia MB':>9} {'seq x':>7} {'rand x':>7}")
    for r in results:
        print(f"{r['sample'][:32]:32} {r['format']:13} {r['source_mb']:>8} {r['omnia_mb']:>9} {r['speedup_sequential']:>7} {r['speedup_random']:>7}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
