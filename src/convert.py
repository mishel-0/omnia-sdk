#!/usr/bin/env python3
"""
Convert raw uncompressed DICOM CT studies to .omnia Zstd containers.
Usage: python3 convert.py /path/to/lidc_raw/ /path/to/output/
"""
import sys, time, json
from pathlib import Path
import pydicom
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from container import OmniaContainer


def convert_study(dicom_dir: Path, out_dir: Path, verbose: bool = True) -> dict:
    """Convert one study to .omnia. Returns stats.
    
    Properly handles shape/dtype mismatches: only CT slices with matching
    512x512 int16 shape are included. Mismatched slices are logged and skipped.
    """
    dcm_files = sorted(dicom_dir.rglob("*.dcm"))

    # Filter to CT modality, collect valid slices
    valid_slices: list[np.ndarray] = []
    ref_shape: tuple | None = None
    ref_dtype: np.dtype | None = None
    skipped = 0

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            if getattr(ds, "Modality", "") != "CT":
                continue
        except Exception:
            skipped += 1
            continue

        try:
            ds = pydicom.dcmread(str(f), force=True)
            arr = ds.pixel_array
        except Exception:
            skipped += 1
            if verbose:
                print(f"  WARN: Could not read pixels from {f.name}")
            continue

        # Establish reference on first valid slice
        if ref_shape is None:
            ref_shape = arr.shape
            ref_dtype = arr.dtype
            valid_slices.append(arr)
        elif arr.shape == ref_shape and arr.dtype == ref_dtype:
            valid_slices.append(arr)
        else:
            skipped += 1
            if verbose:
                print(f"  WARN: {f.name} shape={arr.shape} dtype={arr.dtype} "
                      f"!= expected {ref_shape} {ref_dtype}, skipped")

    if not valid_slices:
        return {"study": dicom_dir.name, "slices": 0, "error": "No valid CT slices"}

    # Write .omnia
    out_path = out_dir / f"{dicom_dir.name}.omnia"
    stats = OmniaContainer.write(out_path, valid_slices)
    stats["study"] = dicom_dir.name
    stats["skipped"] = skipped

    if verbose:
        mb_orig = stats["original_bytes"] / 1e6
        mb_comp = stats["compressed_bytes"] / 1e6
        skip_info = f" ({skipped} skipped)" if skipped else ""
        print(f"  {stats['study']:<20s} {stats['slices']:>4d} slices  "
              f"{mb_orig:>7.1f} MB → {mb_comp:>6.1f} MB  {stats['ratio']:.2f}x{skip_info}")

    return stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 convert.py <raw_dicom_dir/> <output_dir/>")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    studies = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(studies)} study directories")

    all_stats = []
    total_orig = 0
    total_comp = 0
    total_skipped = 0
    t0 = time.perf_counter()

    for study_dir in studies:
        result = convert_study(study_dir, out_dir)
        all_stats.append(result)
        if result["slices"] > 0:
            total_orig += result["original_bytes"]
            total_comp += result["compressed_bytes"]
            total_skipped += result.get("skipped", 0)
        else:
            print(f"  {study_dir.name:<20s} SKIPPED ({result.get('error', 'no data')})")

    elapsed = time.perf_counter() - t0
    overall_ratio = total_orig / total_comp if total_comp else 0

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Total original:    {total_orig/1e6:.1f} MB")
    print(f"  Total compressed:  {total_comp/1e6:.1f} MB")
    print(f"  Overall ratio:     {overall_ratio:.2f}x lossless (Zstd)")
    print(f"  Studies converted: {sum(1 for s in all_stats if s['slices'] > 0)}/{len(studies)}")
    if total_skipped:
        print(f"  Skipped files:     {total_skipped} (shape/dtype mismatch)")

    summary = {
        "elapsed_s": round(elapsed, 1),
        "total_original_mb": round(total_orig / 1e6, 1),
        "total_compressed_mb": round(total_comp / 1e6, 1),
        "overall_ratio": round(overall_ratio, 3),
        "total_skipped": total_skipped,
        "codec": "zstd",
        "studies": all_stats,
    }
    summary_path = out_dir / "conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
