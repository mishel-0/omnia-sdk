"""
PyTorch Datasets for .omnia containers.

Pre-decompresses all tiles at init for maximum training speed.
Worker-safe: uses shared memory via numpy memmap for large datasets,
or direct RAM for smaller datasets.

Usage:
    ds = OmniaDataset("slide.omnia", cache_mode="ram")  # fast, needs RAM
    ds = OmniaDataset("slide.omnia", cache_mode="mmap")  # disk-backed, for large
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    for images, labels in loader:
        ...
"""
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .container import OmniaContainer


class OmniaDataset(Dataset):
    """PyTorch Dataset over .omnia files with pre-decompression.

    At init, decompresses ALL tiles into a contiguous array.
    Training then becomes zero-copy array slicing — no I/O, no Zstd.

    cache_modes:
        "ram"  — decompress to RAM (fastest). Good for <4 GB datasets.
        "mmap" — decompress to temp .npy file, memory-map it.
                 Good for large slides (>4 GB). Backed by disk.
        "none" — per-tile on-demand decompression (original behavior).
                 Minimal memory, slowest random access.

    Args:
        omnia_path: Path to a .omnia file or directory containing .omnia files.
        cache_mode: "ram" (default), "mmap", or "none".
        normalize: Divisor for pixel values (default: 255.0).
    """

    def __init__(self, omnia_path: Union[str, Path],
                 cache_mode: str = "ram",
                 normalize: float = 255.0):
        self.omnia_path = Path(omnia_path)
        self.normalize = normalize
        self.cache_mode = cache_mode
        self._data: Optional[torch.Tensor] = None
        self._mmap: Optional[np.ndarray] = None
        self._mmap_path: Optional[Path] = None

        # Load metadata
        self.container = OmniaContainer(self.omnia_path)
        self.container.open()
        self.num_tiles = self.container.num_slices
        self.shape = self.container.shape  # (H, W, C)
        self.dtype = self.container.dtype
        # Cache the constant label so __getitem__ doesn't allocate per tile
        self._label = torch.tensor(0, dtype=torch.long)

        # Pre-decompress all tiles
        if cache_mode != "none":
            self._preload()
        else:
            self.container.close()

    def _preload(self):
        """Decompress all tiles into contiguous storage."""
        if self.container.pixel_codec == "jpeg":
            # JPEG tiles are stored as encoded bytes — decode via PIL
            self._preload_jpeg()
            return

        n = self.num_tiles
        h, w, c = self.shape

        if self.cache_mode == "mmap":
            # Temp file, memory-mapped
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
            self._mmap_path = Path(tmp.name)
            tmp.close()
            # Decompress one by one into the memmap
            arr = np.lib.format.open_memmap(
                str(self._mmap_path), mode="w+",
                dtype=self.dtype, shape=(n, h, w, c)
            )
        else:
            # RAM: allocate the final normalized CHW float32 tensor directly.
            # Filling it per-tile (instead of a uint8 copy + big transpose)
            # halves peak memory on large slides.
            data = np.empty((n, 3, h, w), dtype=np.float32)

        for i in range(n):
            tile = self.container.get_slice(i)
            if tile.ndim == 1:
                # Native JP2K tile — skip (can't preload without decoder)
                if self._mmap_path:
                    self._mmap_path.unlink(missing_ok=True)
                self.container.close()
                self.cache_mode = "none"
                return
            if self.cache_mode == "mmap":
                # Handle edge tiles: pad to standard shape if smaller
                if tile.shape[:2] != (h, w):
                    padded = np.zeros((h, w, c), dtype=self.dtype)
                    th, tw = tile.shape[:2]
                    padded[:th, :tw] = tile
                    arr[i] = padded
                else:
                    arr[i] = tile
            else:
                # Handle edge tiles: zero-pad to standard shape
                if tile.shape[:2] != (h, w):
                    data[i] = 0
                    th, tw = tile.shape[:2]
                    data[i, :, :th, :tw] = tile.transpose(2, 0, 1)
                else:
                    data[i] = tile.transpose(2, 0, 1)

        self.container.close()

        if self.cache_mode == "mmap":
            # Keep as numpy memmap, convert to tensor on the fly
            self._mmap = arr
            self._data = None
        else:
            # Pre-normalize once so __getitem__ returns zero-copy views
            data /= self.normalize  # in-place
            self._data = torch.from_numpy(data)

    def _preload_jpeg(self):
        """Decode JPEG tiles (raw bytes in container) into a normalized CHW tensor."""
        from PIL import Image
        import io

        n = self.num_tiles
        raw0 = self.container.get_slice(0)
        img = Image.open(io.BytesIO(raw0.tobytes())).convert("RGB")
        h, w = img.size[1], img.size[0]
        arr = np.empty((n, h, w, 3), dtype=np.uint8)
        arr[0] = np.asarray(img)
        for i in range(1, n):
            raw = self.container.get_slice(i)
            img = Image.open(io.BytesIO(raw.tobytes())).convert("RGB")
            arr[i] = np.asarray(img)
        self.container.close()

        arr_chw = np.ascontiguousarray(arr.transpose(0, 3, 1, 2), dtype=np.float32)
        arr_chw /= self.normalize
        self._data = torch.from_numpy(arr_chw)

    def __len__(self) -> int:
        return self.num_tiles

    def __getitem__(self, idx: int):
        if self._data is not None:
            # RAM mode: zero-copy view — DataLoader does the single batch copy.
            # No per-tile clone/division (pre-normalized at preload).
            return self._data[idx], self._label

        if self._mmap is not None:
            # Mmap mode: read tile from memmap, convert
            arr = self._mmap[idx]
            tensor = torch.from_numpy(arr.astype(np.float32))
            tensor = tensor.permute(2, 0, 1) / self.normalize
            return tensor, self._label

        # No cache mode (original per-tile decompression)
        arr = self.container.get_slice(idx)
        if self.container.pixel_codec == "jpeg":
            # Decode JPEG tile bytes back to RGB
            from PIL import Image
            import io
            arr = np.asarray(
                Image.open(io.BytesIO(arr.tobytes())).convert("RGB")
            )
        tensor = torch.from_numpy(arr.astype(np.float32))
        if tensor.dim() == 1:
            # Native JP2K tile — raw compressed bytes, no decoder here.
            # Return as-is instead of crashing on permute().
            return tensor, self._label
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor / self.normalize
        return tensor, self._label

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'container') and self.container._file:
            self.container.close()
        if self._mmap_path and self._mmap_path.exists():
            self._mmap_path.unlink(missing_ok=True)

    def __del__(self):
        self.close()
