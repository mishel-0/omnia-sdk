# Benchmark Methodology

## What is measured

Per-epoch **data feeding** time for two paths over an identical fixed tile set:

1. **.svs + openslide** — `read_region(level 0)` + RGB convert inside
   `__getitem__`, so every epoch pays the JPEG-2000 decode cost (realistic
   openslide usage).
2. **.omnia + OmniaDataset(cache_mode="ram")** — all tiles decoded once at
   init (timed separately as `preload_time_s`), then each epoch serves
   zero-copy views of the preloaded float32 tensor.

Both loaders: `DataLoader(batch_size=64, shuffle=True, num_workers=0,
pin_memory=True if GPU)`, batches moved to device with `non_blocking=True`.
No model runs — the ratio isolates the data path.

## Protocol

- **Seed 42** for torch, numpy, random, and the tile-coordinate sampler.
- **Identical tile list**: 5,000 (default) level-0 256x256 coordinates,
  sampled once, reused by both paths. Edge padding is handled by openslide
  (always returns 256x256) and by the container (zero-padded edge tiles).
- **Epochs**: 5 (default) per path. The first epoch is warmup and is dropped
  from the mean (`warmup_dropped=true`).
- **num_workers=0**: preloaded data has no I/O to parallelize; worker
  processes only add IPC overhead. Measured: nw=2 is 1.12x slower.
- **GPU utilization**: `nvidia-smi --query-gpu=utilization.gpu` sampled every
  ~0.4s during one epoch of each path. Only collected where a GPU exists — on
  a CPU-only machine these fields are empty, and the committed
  `benchmarks/benchmark_results.json` is such a run.

  The direction is not in doubt: `.svs` spends 97.6% of an epoch decoding
  rather than training, so its utilisation is low by construction. The
  magnitude is a property of your model and your GPU, and this repository does
  not yet publish a measured pair. Run `colab/omnia_vs_svs_benchmark.ipynb` on
  a GPU runtime to obtain one for your own hardware.

## Speedup definition

```
data_speedup = mean(svs_epoch_times[1:]) / mean(omnia_epoch_times[1:])
```

## The two regimes (read this)

The benchmark always measures BOTH the data path and a real full-train loop
(ResNet-18, small subset), because the honest speedup depends on which
regime your workload is in. Classification is a pure function of the
measured full-train speedup, with thresholds matching the README table:

- **Data-bound** — full-train speedup ≥ 10x. The GPU is starved waiting for
  openslide decode (tile classification, grading, detection; small/fast
  models). Data path 15-25x, end-to-end 10x+.
- **Mixed** — full-train speedup ~2-10x. Data and training both matter.
- **Model-bound** — full-train speedup ~1-2x. Training (forward/backward)
  dominates the epoch (foundation-model fine-tuning, huge ViTs). The data
  path is still fast but the end-to-end number is near break-even.

The JSON reports `measurement.regime` (data-bound / mixed / model-bound)
derived from the full-train speedup, plus `measurement.full_train` with the
raw times. Publish all three buckets — hiding the middle case is how
benchmarks get shredded in diligence.

## Known, published caveats

1. **Data path vs full-train.** `data_speedup` isolates the loader; the
   `full_train` block includes a real ResNet-18 forward/backward. Which one
   to quote depends on your workload's regime — the JSON tells you.
2. **Machine dependence.** Absolute times vary (CPU, GPU, disk, slide).
   The RATIO is stable in the 8-25x range across the machines tested.
3. **.svs is already compressed.** zstd on .svs bytes achieves only ~1.1x
   (the file is ~127x-compressed JP2K). The 5x zstd number is vs decoded raw
   pixels. Size and speed are different claims.

## Baseline measurements (committed in benchmarks/benchmark_results.json)

| Machine | Tiles | .svs s/epoch | .omnia s/epoch | Speedup |
|---|---|---|---|---|
| Colab T4, 20x | 6,426 | 103.82 | 5.20 | 20.0x |
| Mac CPU | 10,000 | 14.86 | 1.69 | 8.8x |

Regenerate with `python -m omnia_sdk.benchmark` — the JSON is rewritten from
the machine that runs it, not shipped as a stale claim.
