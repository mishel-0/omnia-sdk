# Which codec trains fastest?

`omnia recommend slide.svs` measures this on your own data in about a minute without a GPU.
The numbers below are from one run on a RunPod **A40** (ResNet-18, AMP, batch 128, shuffled
`DataLoader`, pod limited to 9 vCPU). Raw data: [`benchmarks/codec_results_a40.json`](../benchmarks/codec_results_a40.json).
Re-run it: `omnia benchmark codecs your.svs`.

Two public OpenSlide test slides, level-0 256 px tiles. GPU ceiling (no data loading) was 2560 img/s.

| Variant | CMU-1 size | vs raw | decode, 1 thread (tiles/s) | GPU kept busy, 4 workers | JP2K slide, 4 workers |
|---|---|---|---|---|---|
| raw `.npy` | 590 MB | 1.0x | 41,600 | 97% | 98% |
| zstd L1, 1 tile/block | 543 MB | 1.09x | 2,900 | 98% | 94% |
| zstd L3, 1 tile/block | 542 MB | 1.09x | 3,340 | 97% | 97% |
| zstd L3, 16 tiles/block | 542 MB | 1.09x | 243 | **33%** | 72% |
| zstd L9, 16 tiles/block | 543 MB | 1.09x | 244 | **33%** | 75% |
| JPEG q75 | 48 MB | 12.3x | 1,860 | 97% | 97% |
| JPEG q90 | 76 MB | 7.8x | 1,710 | 96% | 98% |
| JPEG q95 | 104 MB | 5.7x | 1,540 | 96% | 98% |
| openslide (decode the .svs each time) | 178 MB | n/a | 474 | 53% | **11%** |

## What the data says
- **Batching tiles into blocks hurts shuffled training and buys nothing here.** Sixteen tiles per block
  gave the same size as one (1.09x) and made a random read 6-14x slower. `omnia convert` now defaults to one
  tile per block for slides.
- **Higher zstd levels bought no space** on these tiles and cost 2-3x more to encode.
- **Lossless is barely smaller than raw** (1.09x): scanner-compressed H&E tiles are noisy. If you need
  lossless, expect disk close to raw size.
- **JPEG is 5.7-12x smaller and trained as fast as raw** with 4 workers. It is lossy and the source slides
  are already lossy, so this is a second generation of loss. Fidelity against the original tiles (ResNet-18
  feature cosine): q75 0.979 / 0.958, q90 0.989 / 0.986, q95 0.994 / 0.988 (CMU-1 / JP2K slide). PSNR 26-32 dB.
  **Not measured: the effect on a trained model's accuracy.** Check that on your task before switching.
- **Reading the slide directly can starve the GPU when CPU cores are scarce**: openslide kept 53% and 11% of
  the GPU busy on 4 workers, and recovered at 8 workers. With plenty of cores (or a slower model) every
  option keeps up, so the advantage is mostly about CPU cost and disk size, not a guaranteed epoch speed-up.

## Limits of this run
One GPU (A40), one model (ResNet-18), two slides, 9 vCPU, a single run per variant (no repeats). A faster GPU or
model raises the ceiling and widens the gaps. This measures the on-demand path; `OmniaDataset(cache_mode="ram")`
decodes once up front, which avoids per-epoch decode cost but needs the whole dataset in RAM as float32.
