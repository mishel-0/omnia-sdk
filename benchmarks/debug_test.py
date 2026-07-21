#!/usr/bin/env python3
"""
Comprehensive debug test for .omnia Zstd training edition.
Tests: DICOM reading, .omnia conversion, lossless verify, header format, batch convert.
"""
import sys, os, json, struct, zstd, time
from pathlib import Path
import numpy as np
import pydicom

sys.path.insert(0, "/workspace/omnia-training/src")
from container import OmniaContainer

RAW_DIR = Path("/workspace/lidc_raw")
OMNIA_DIR = Path("/workspace/omnia_training_data")
OMNIA_DIR.mkdir(exist_ok=True)


def test_1_dicom_reading():
    print("\n" + "=" * 60)
    print("TEST 1: DICOM File Reading")
    print("=" * 60)
    dcm_files = sorted(RAW_DIR.rglob("*.dcm"))
    print(f"  Total DICOM files: {len(dcm_files)}")
    ct_files = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
        if getattr(ds, "Modality", "") == "CT":
            ct_files.append(f)
    print(f"  CT files: {len(ct_files)}")
    assert len(ct_files) > 0, "No CT files found!"
    ds = pydicom.dcmread(str(ct_files[0]), force=True)
    arr = ds.pixel_array
    print(f"  First: {ct_files[0].name}")
    print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}, Range: {arr.min()}..{arr.max()}")
    ts = str(ds.file_meta.TransferSyntaxUID)
    print(f"  TransferSyntax: {ts}")
    assert arr.shape == (512, 512)
    assert ts == "1.2.840.10008.1.2.1"
    print("  PASS: Raw uncompressed DICOM")
    return ct_files[0], arr


def test_2_single_conversion(sample_file, sample_arr):
    print("\n" + "=" * 60)
    print("TEST 2: Single Slice .omnia Conversion")
    print("=" * 60)
    out_path = Path("/tmp/test_single.omnia")
    stats = OmniaContainer.write(out_path, [sample_arr])
    print(f"  Original: {stats['original_bytes']} bytes")
    print(f"  Compressed: {stats['compressed_bytes']} bytes")
    print(f"  Ratio: {stats['ratio']:.2f}x")
    with OmniaContainer(out_path) as reader:
        reloaded = reader.get_slice(0)
        identical = np.array_equal(sample_arr, reloaded)
        max_diff = int(np.abs(sample_arr.astype(np.int32) - reloaded.astype(np.int32)).max())
        print(f"  Lossless: {'PASS' if identical else 'FAIL'}  Max diff: {max_diff}")
        assert identical, "Lossless check FAILED!"
    out_path.unlink()
    print("  PASS")


def test_3_study_conversion():
    print("\n" + "=" * 60)
    print("TEST 3: Full Study Conversion (1108 slices)")
    print("=" * 60)
    study = RAW_DIR / "LIDC-IDRI-0010"
    assert study.exists()
    dcm_files = sorted(study.rglob("*.dcm"))
    ct_arrays = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
        if getattr(ds, "Modality", "") == "CT":
            ct_arrays.append(pydicom.dcmread(str(f), force=True).pixel_array)
    print(f"  CT slices: {len(ct_arrays)}")
    t0 = time.perf_counter()
    out_path = OMNIA_DIR / "LIDC-IDRI-0010.omnia"
    stats = OmniaContainer.write(out_path, ct_arrays)
    elapsed = time.perf_counter() - t0
    print(f"  Original: {stats['original_bytes']/1e6:.1f} MB")
    print(f"  Compressed: {stats['compressed_bytes']/1e6:.1f} MB")
    print(f"  Ratio: {stats['ratio']:.2f}x")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(ct_arrays)*1000:.2f}ms/slice)")
    print(f"  Verifying lossless...")
    errors = 0
    with OmniaContainer(out_path) as reader:
        for i, orig in enumerate(ct_arrays):
            reloaded = reader.get_slice(i)
            if not np.array_equal(orig, reloaded):
                errors += 1
    assert errors == 0, f"{errors}/{len(ct_arrays)} slices had errors!"
    print(f"  PASS: {stats['ratio']:.2f}x, {errors} errors")


def test_4_format_header():
    print("\n" + "=" * 60)
    print("TEST 4: Binary Format Header")
    print("=" * 60)
    f = OMNIA_DIR / "LIDC-IDRI-0010.omnia"
    with open(f, "rb") as fh:
        magic = fh.read(4)
        ver, codec, _ = struct.unpack("B B H", fh.read(4))
        ms = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(zstd.decompress(fh.read(ms)))
        px_start = fh.tell()
        fh.seek(0, 2)
        fsize = fh.tell()
    print(f"  Magic: {magic}  Version: {ver}  Codec: {codec}")
    print(f"  Slices: {meta['n']}  Dtype: {meta['dtype']}  Shape: {meta['shape']}")
    assert magic == b"OMN2"
    assert ver == 3
    assert codec == 0
    assert px_start + sum(meta['sz']) == fsize
    print("  PASS: Format valid")


def test_5_all_studies():
    print("\n" + "=" * 60)
    print("TEST 5: Batch Convert All Studies")
    print("=" * 60)
    studies = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    total_orig = 0
    total_comp = 0
    t0 = time.perf_counter()
    for study_dir in studies:
        ct_arrays = []
        for f in sorted(study_dir.rglob("*.dcm")):
            try:
                ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
                if getattr(ds, "Modality", "") == "CT":
                    ct_arrays.append(pydicom.dcmread(str(f), force=True).pixel_array)
            except: pass
        if not ct_arrays:
            print(f"  {study_dir.name:<20s} SKIPPED")
            continue
        out_path = OMNIA_DIR / f"{study_dir.name}.omnia"
        stats = OmniaContainer.write(out_path, ct_arrays)
        stats["study"] = study_dir.name
        total_orig += stats["original_bytes"]
        total_comp += stats["compressed_bytes"]
        print(f"  {stats['study']:<20s} {stats['slices']:>4d} slices  "
              f"{stats['original_bytes']/1e6:>7.1f} MB -> {stats['compressed_bytes']/1e6:>6.1f} MB  {stats['ratio']:.2f}x")
    elapsed = time.perf_counter() - t0
    ratio = total_orig / total_comp if total_comp else 0
    print(f"\n  Total: {total_orig/1e6:.1f} MB -> {total_comp/1e6:.1f} MB = {ratio:.2f}x")
    print(f"  Time: {elapsed:.0f}s")
    assert total_comp > 0
    print("  PASS")


def test_6_random_access():
    print("\n" + "=" * 60)
    print("TEST 6: Random Access Performance")
    print("=" * 60)
    f = OMNIA_DIR / "LIDC-IDRI-0010.omnia"
    with OmniaContainer(f) as reader:
        n = reader.num_slices
        print(f"  {n} slices")
        t0 = time.perf_counter()
        for i in range(n):
            reader.get_slice(i)
        seq = time.perf_counter() - t0
        print(f"  Sequential: {seq:.3f}s ({seq/n*1000:.3f}ms/slice)")
        
        import random
        random.seed(42)
        idx = random.sample(range(n), min(100, n))
        t0 = time.perf_counter()
        for i in idx:
            reader.get_slice(i)
        rand = time.perf_counter() - t0
        print(f"  Random 100: {rand:.3f}s ({rand/len(idx)*1000:.3f}ms/slice)")
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("OMNIA TRAINING EDITION - COMPREHENSIVE DEBUG")
    print(f"  Mode: Zstd codec")
    print(f"  Data: {RAW_DIR}")
    print("=" * 60)
    
    tests = [
        ("DICOM Reading", lambda: test_1_dicom_reading()),
        ("Single Conversion", lambda: test_2_single_conversion(*test_1_dicom_reading())),
        ("Study Conversion", test_3_study_conversion),
        ("Format Header", test_4_format_header),
        ("Batch Convert", test_5_all_studies),
        ("Random Access", test_6_random_access),
    ]
    
    passed = failed = 0
    for name, func in tests:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)
