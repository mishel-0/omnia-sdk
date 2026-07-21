# Benchmarks

This directory contains the reproducible benchmark suite for .omnia.

## Contents

| File | Description |
|------|-------------|
| `benchmark.json` | Full 100-epoch results with hardware, software, and per-epoch timing |
| `benchmark.py` | Reproduction script — run it yourself |
| `requirements.txt` | Python dependencies |
| `dataset.md` | How to obtain the LIDC-IDRI dataset |

## How to reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Obtain the LIDC-IDRI dataset (see dataset.md)
#    Expected: /path/to/lidc_raw/ with DICOM files

# 3. Convert to .omnia (requires omnia-sdk)
#    See: https://github.com/mishel-0/omnia-sdk

# 4. Run benchmark
python benchmark.py /path/to/lidc_raw/ /path/to/compressed/
```

## Reference results

Results will vary with hardware. Expected ranges:

| Metric | DICOM | .omnia |
|--------|-------|--------|
| Cold epoch 1 | 40-200 s | 20-70 s |
| Steady epoch | 18-45 s | 17-35 s |
| GPU utilization | 40-60% | 85-100% |
| Dataset load | 30-300 s | 0.5-5 s |

## Methodology

- Model: ResNet-18 (conv1 modified for 1-channel input, 2-class output)
- Batch size: 64
- Data loaders: 4 workers, prefetch=2, pin_memory, persistent_workers
- Epochs: 100 (first epoch reported separately as cold start)
- GPU monitoring: nvidia-smi polled every 1s during training
- CUDNN: benchmark mode enabled, deterministic disabled
