# SDK Reference

## Python SDK

### Study

```python
from omnia import Study

study = Study("path/to/study.omnia")
```

**Parameters:** `path` — path to a .omnia file.

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `num_slices` | int | Number of slices in the study |
| `shape` | tuple | Slice dimensions (height, width) |
| `dtype` | numpy.dtype | Pixel data type |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `study[i]` | numpy.ndarray | Get slice `i` (O(1)) |
| `study.volume()` | numpy.ndarray | Get all slices as a 3D array |
| `study.metadata(i)` | dict | Get metadata for slice `i` |

### Dataset

```python
from omnia_sdk.dataset import OmniaDataset

ds = OmniaDataset("/path/to/compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
```

Works as a drop-in `torch.utils.data.Dataset`. Each item returns `(tensor, label)`.

## C SDK

The C SDK is available for enterprise licensing. It exposes six functions:

```c
OmniaHandle* omnia_open(const char* path);
uint16_t*    omnia_get_slice(OmniaHandle* handle, int index);
uint16_t*    omnia_get_volume(OmniaHandle* handle);
const char*  omnia_get_tag(OmniaHandle* handle, int slice, const char* tag);
void         omnia_close(OmniaHandle* handle);
```

Contact `contact@omnia-sdk.com` for SDK access.
