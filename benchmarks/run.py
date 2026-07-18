#!/usr/bin/env python3
"""Omnia SDK — Benchmark Suite"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import glob
import numpy as np
from omnia import OmniaCompressor

BENCH_DIR = "/tmp/omnia-bench"
RESULTS = []


def bench(name, fn):
    t0 = time.time()
    result = fn()
    t = time.time() - t0
    RESULTS.append({"name": name, "time_s": round(t, 3), **result})
    print(f"  {name:30s} {result.get('ratio', 0):.1f}x  {t:.2f}s")
    return result


print(f"\n{'='*50}")
print(f"  OMNIA SDK BENCHMARK")
print(f"{'='*50}")

# 1. Find test data
test_files = sorted(glob.glob(
    "/Users/misheladnan/Desktop/omnia-AI/Omnia-AI/data/lidc_test/*/lidc_idri/*/1.3.6.1.4.1*/CT_*/*.dcm"))
if not test_files:
    print("  No test DICOMs found")
    sys.exit(1)

# 2. Test with 10 slices
os.makedirs(f"{BENCH_DIR}/10", exist_ok=True)
for f in test_files[:10]:
    os.system(f"cp '{f}' {BENCH_DIR}/10/")

comp = OmniaCompressor()

print(f"\n  10 slices:")
bench("compress", lambda: comp.compress(f"{BENCH_DIR}/10/", f"{BENCH_DIR}/10.omnia", verbose=False))
bench("decompress", lambda: comp.decompress(f"{BENCH_DIR}/10.omnia", f"{BENCH_DIR}/10_out/", verbose=False))

# 3. Test with 50 slices
os.makedirs(f"{BENCH_DIR}/50", exist_ok=True)
for f in test_files[:50]:
    os.system(f"cp '{f}' {BENCH_DIR}/50/")

print(f"\n  50 slices:")
bench("compress", lambda: comp.compress(f"{BENCH_DIR}/50/", f"{BENCH_DIR}/50.omnia", verbose=False))
bench("decompress", lambda: comp.decompress(f"{BENCH_DIR}/50.omnia", f"{BENCH_DIR}/50_out/", verbose=False))

# 4. Summary
print(f"\n{'='*50}")
print(f"  RESULTS SUMMARY")
print(f"{'='*50}")
for r in RESULTS:
    ratio = r.get("ratio", 0)
    size_mb = r.get("compressed_size", 0) / 1024**2
    print(f"  {r['name']:30s} {r['time_s']:>6.2f}s  {ratio:>4.1f}x  {size_mb:>6.1f}MB")

# Save
report = {
    "sdk_version": "2.0.0",
    "test_date": time.strftime("%Y-%m-%d"),
    "num_slices_tested": [10, 50],
    "results": RESULTS,
}
with open(f"{BENCH_DIR}/benchmark_results.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Report saved: {BENCH_DIR}/benchmark_results.json")
