# .omnia SDK

Fast, lossless medical image containers for ML training.

```bash
pip install https://github.com/mishel-0/omnia-sdk/releases/download/v1.0.1/omnia_sdk-1.0.1-py3-none-any.whl

# Convert DICOM to .omnia
omnia compress /path/to/raw_dicom/ /path/to/output/

# Train with PyTorch
from omnia_sdk import OmniaDataset
ds = OmniaDataset("/path/to/omnia/files/")
```

**Contact:** misheladnan35@gmail.com
**License:** Proprietary — All rights reserved.
