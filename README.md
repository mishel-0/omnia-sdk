# .omnia SDK

Fast, lossless medical image containers for ML training.

## Install

```bash
pip install https://github.com/mishel-0/omnia-sdk/releases/download/v1.1.0/omnia_sdk-1.1.0-py3-none-any.whl
```

## License

A valid license key is required. Contact misheladnan35@gmail.com to request one.

Once you receive your key:

```bash
mkdir -p ~/.omnia
echo "your-license-key-here" > ~/.omnia/license.key
```

## Usage

```python
from omnia_sdk import OmniaContainer

# Compress a study
stats = OmniaContainer.write("study.omnia", pixel_arrays)

# Read slices
with OmniaContainer("study.omnia") as study:
    slice_47 = study[47]

# PyTorch training (requires torch)
from omnia_sdk.dataset import OmniaDataset
```

```bash
# CLI
omnia compress ./raw/ ./compressed/
omnia verify ./study.omnia
```

## Benchmark

| Metric | Raw DICOM | .omnia |
|--------|-----------|--------|
| Cold epoch 1 | 215.8 s | 69.3 s |
| GPU utilization | 48% | 93% |
| Storage | 1,819 MB | 837 MB |

## Contact

misheladnan35@gmail.com
