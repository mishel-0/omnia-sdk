# Verification Guide — for skeptics

This is the audit path for anyone who wants to independently confirm the
"20x data feeding" claim. Everything is designed to be checked, not trusted.

## What the claim is

`OmniaDataset(cache_mode="ram")` feeds whole-slide-image tiles to a training
loop ~20x faster per epoch (data path only) than reading the same tiles with
openslide from a .svs file, because openslide re-decodes JPEG-2000 from disk
every epoch while .omnia decodes once at init and serves zero-copy tensor
views.

## Step 1 — read the benchmark code

`omnia_sdk/benchmark.py` is ~250 lines. Verify:
- No hardcoded speedup. The only constants are the seed (42), tile size
  (256), batch (64), epochs (default 5), and tile count (default 5000).
- Both paths iterate the IDENTICAL fixed tile coordinate list (seeded), same
  batch size, same DataLoader settings, `num_workers=0`.
- The .svs path calls `slide.read_region(...)` inside `__getitem__` — the
  decode happens every epoch.
- The .omnia path preloads once (`preload_time_s` in the JSON) then serves
  `self._data[idx]` views.
- The first epoch is excluded from the mean (`warmup_dropped`).
- GPU utilization is sampled live from `nvidia-smi` during one epoch of each
  path.

## Step 2 — run it yourself

```bash
git clone https://github.com/mishel-0/omnia-sdk.git
cd omnia-sdk
pip install -e .
python -m omnia_sdk.benchmark
```

Or use the one-click Colab notebook (no install, no uploads):
https://colab.research.google.com/github/mishel-0/omnia-sdk/blob/main/colab/omnia_vs_svs_benchmark.ipynb

Expected: a printed summary and `benchmarks/benchmark_results.json`.

## Step 3 — check the receipt

The JSON contains:

| Field | What it proves |
|---|---|
| `machine.*` | the exact environment (OS, CPU, GPU, python, pinned package versions) |
| `slide.sha256` | the exact input file — hashes to the same value on any run |
| `measurement.svs_epoch_times_s` | per-epoch decode-bound times, full list |
| `measurement.omnia_epoch_times_s` | per-epoch preloaded times, full list |
| `measurement.data_speedup` | mean(svs) / mean(omnia) — the headline |
| `measurement.full_train` | end-to-end times with a real ResNet-18 loop — the caveat number |
| `measurement.regime` | data-bound / mixed / model-bound — which case this machine is in |
| `measurement.gpu_util_*` | live nvidia-smi samples: expect ~high for omnia, low for svs |
| `measurement.passed_20x` | objective pass/fail vs the 20x target |

If `full_train.speedup` is ~1-2x, the machine is model-bound — that's the
published caveat working as intended, not a broken benchmark.

## Expected results by machine

| Machine | Expected data speedup | Why |
|---|---|---|
| Colab T4 / L4 (GPU) | 15-25x | svs epoch is decode-bound (~100s), omnia is memory-bound (~5s) |
| Modern Mac/PC (CPU) | 8-12x | same ratio, faster per-tile decode |
| Weak/shared GPU | speedup still high data-only | end-to-end with training will be lower — see caveat below |

## The caveat you should verify too

The 20x is the DATA path. The JSON also contains a full-train measurement
(real ResNet-18 forward/backward) and a `regime` classification. On a
model-bound machine the end-to-end number will be ~1-2x — that is the
published caveat, not a bug. A claim that hides this is a red flag — this
one doesn't.

## Reproducibility contract

- Fixed seeds: 42 (torch, numpy, random, tile sampler).
- Identical tile set for both paths.
- Warmup epoch dropped from the mean.
- `num_workers=0` (workers are slower for preloaded data — testable).
- Public slide auto-downloaded (CMU openslide test-data mirror); SHA-256 in
  the JSON. Any slide works via `--svs`.
- Package versions printed in the JSON.
