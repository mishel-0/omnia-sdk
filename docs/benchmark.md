# Benchmark Methodology

## Reproduction

```bash
# 1. Install dependencies
pip install -r benchmarks/requirements.txt

# 2. Obtain LIDC-IDRI dataset (see benchmarks/dataset.md)

# 3. Run benchmark
python benchmarks/benchmark.py /path/to/lidc_raw/ /path/to/compressed/
```

## Reference hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A4000 (16 GB) |
| CPU | 16 cores @ 2.5 GHz (Xeon) |
| RAM | 440 GB |
| Storage | Network filesystem |
| OS | Ubuntu 22.04, kernel 5.15 |

## Reference software

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| torchvision | 0.23.0+cu128 |
| numpy | 2.1.2 |
| pydicom | 3.0.2 |

## Model

ResNet-18 with conv1 modified for 1-channel input. 2-class output. ~11.2M parameters.

## Training configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Epochs | 100 |
| Workers | 4 |
| Prefetch factor | 2 |
| Pin memory | True |
| Persistent workers | True |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| CUDNN benchmark | True |

## Cache effects

The test system has 440 GB RAM. The 1.8 GB DICOM dataset fits entirely in OS page cache. This is the best possible case for DICOM. For production datasets (10 TB+) that exceed available RAM, DICOM remains in cold-start territory while .omnia maintains steady throughput.

## Expected variation

Results may vary ±15% depending on CPU model, storage hardware, OS cache state, and background processes.
