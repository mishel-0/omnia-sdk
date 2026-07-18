# Omnia SDK

**DICOM Compression SDK — 3x lossless, 277 files → 1, multi-language.**

```
pip install omnia-sdk
omnia compress ./ct_scan/ output.omnia
```

## Quick Start

```python
from omnia import OmniaCompressor

comp = OmniaCompressor()

# Compress 277 DICOM slices → 1 .omnia file (3x smaller)
result = comp.compress("/data/ct_scan/", "study.omnia")
print(f"{result['ratio']:.1f}x compression")

# Decompress back to original DICOMs
comp.decompress("study.omnia", "/data/restored/")
```

## CLI

```bash
# Compress
omnia compress ./ct_slices/ study.omnia

# Decompress
omnia decompress study.omia ./restored/

# Info
omnia info study.omnia

# Verify integrity
omnia verify study.omnia
```

## Language Support

| Language | Status | Install |
|----------|--------|---------|
| Python | ✅ Stable | `pip install omnia-sdk` |
| C++ | ✅ Stable | `#include <omnia/omnia.hpp>` |
| C# | ✅ Stable | `dotnet add package OmniaSDK` |
| Java | ✅ Stable | Maven: `com.omnia:omnia-sdk` |
| CLI | ✅ Stable | Binary download |

## Architecture

```
.omnia container:
  ┌─────────────────────────────────┐
  │ HEADER: Magic, version, sizes   │
  ├─────────────────────────────────┤
  │ METADATA: DICOM tags (JSON+ZSTD)│
  ├─────────────────────────────────┤
  │ PIXELS: JPEG 2000 lossless      │
  └─────────────────────────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Compression ratio | **3.0x** (verified on 538 slices) |
| Compress speed | **~30 slices/sec** |
| Decompress speed | **~60 slices/sec** |
| Lossless | **100%** (SHA256 verified) |
| Input | DICOM (.dcm), any size |
| Output | Single .omnia file |

## White-Label

Omnia SDK is available for white-label licensing to PACS vendors and cloud storage providers.

**Contact:** hello@omnia.ai

## License

MIT License — free for integration. Commercial licenses available.
