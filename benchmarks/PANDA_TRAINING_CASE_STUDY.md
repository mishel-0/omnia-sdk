# Case study: training a real prostate-cancer grading model on omnia-sdk

Every number below is from an actual training run, not a synthetic benchmark.
This is what happened when omnia-sdk was used to train an attention-MIL model
on the [PANDA](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
prostate biopsy dataset — 9,128 real whole-slide images, on a single rented
RTX 4090.

## The problem this solved

The same pipeline was first run against the raw `.svs`-style data directly
via openslide, streamed from a network-mounted archive. That run was budgeted
at ~3 hours based on the training loop alone — the evaluation pass (test-time
augmentation over every validation slide, every epoch) was never separately
timed. The run took **11 hours and cost $8.15** before the account ran dry,
and it never finished a single epoch.

That failure is the reason omnia-sdk got used here at all: the openslide path
re-extracts and re-decodes each slide from network storage on every access,
so its cost is dominated by unpredictable I/O, not model compute. Converting
the dataset once to `.omnia` containers turns every subsequent epoch into
pure GPU-bound work.

## The conversion

9,128 slides, tissue-tile-selected (64 tiles/slide, 128px, Macenko-normalized
— the exact same tissue-detection code the training script itself uses, not
a separate reimplementation), converted to `.omnia` containers:

| | |
|---|---|
| Slides converted | 9,128 |
| Failures | **0** |
| Wall time | 95 minutes |
| Output size | 23,960 MB (1.20x compression vs. raw tile bytes) |
| Cost | ~$1.22 |

## The training speedup

Measured on the same machine, same model, same data, before and after
conversion:

| | openslide (`.svs`, streamed) | omnia-sdk (`.omnia`, pre-converted) |
|---|---|---|
| Per-step throughput | ~0.44-0.58 steps/sec | ~9-13 steps/sec |
| First full-dataset epoch | never finished (11h, $8.15, aborted) | **117 seconds** |
| Steady-state epoch (cache-warm) | ~100-160s (after openslide's LRU cache fills) | ~100-160s (consistent from epoch 1) |

The steady-state numbers converge once openslide's local-disk cache is warm
— the real win isn't raw per-epoch throughput at that point, it's that
omnia-sdk has **no cold-start penalty and no unpredictable I/O tail**. The
decompression cost is paid once, at conversion time, not re-paid every epoch
against network storage. That predictability is what a $10 training budget
actually needs — the original failure came from a wall-clock estimate that
couldn't account for I/O variance the tool didn't expose. See
[docs/VERIFICATION.md](../docs/VERIFICATION.md) for how to reproduce a
throughput measurement like this on your own hardware before committing to a
real budget.

## The actual model results

Two independent 5-fold attention-MIL models (EfficientNet-B0 and -B1
backbones), each trained end-to-end on the omnia-sdk data path:

| | Cross-validated QWK (9,128 held-out slides) |
|---|---|
| B0 alone | 0.8640 |
| B1 alone | 0.8642 |
| **2-model ensemble** | **0.8735** |

Best single fold (B0, fold 2, extended training): **0.8771**.

For reference, the actual [Kaggle PANDA 1st-place
solution](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)
scored 0.940 private leaderboard — using a 2-model x 5-fold ensemble plus an
iterative manual label-cleaning pass across ~45-60 GPU-hours. This run used a
fraction of that compute and reached a genuinely competitive single-digit
percentage below it, entirely within a $10 total budget for conversion,
training, and ensembling combined.

## Total cost

Conversion + calibration + 5-fold B0 + 5-fold B1 + ensembling: **under $10**,
on rented cloud GPU time, including the mistakes made along the way (a
wrong-network-volume pod, a killed and restarted run) — none of which
required starting over, because `.omnia` containers and checkpoints persist
independently of any one pod.
