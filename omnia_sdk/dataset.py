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
import csv
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

        # A directory of containers — a cohort. The docstring has always
        # promised this and the constructor never implemented it, so pointing
        # the dataset at the output of `convert("slides/", "out/")` raised
        # IsADirectoryError: you could convert a cohort and then not load it.
        #
        # Composed from one dataset per file rather than by teaching the
        # single-file path about many containers. Each part preloads exactly as
        # it does on its own, so the cache modes — and the speed they are
        # measured on — are untouched.
        self._data = None
        self._mmap = None
        self._mmap_path = None
        self._parts: Optional[list] = None
        self._offsets: list[int] = []
        if self.omnia_path.is_dir():
            files = sorted(self.omnia_path.glob("*.omnia"))
            if not files:
                raise FileNotFoundError(f"No .omnia files in {self.omnia_path}")
            self._parts = [OmniaDataset(f, cache_mode=cache_mode, normalize=normalize)
                           for f in files]
            total = 0
            for part in self._parts:
                self._offsets.append(total)
                total += len(part)
            self.num_tiles = total
            self.shape = self._parts[0].shape
            self.dtype = self._parts[0].dtype
            self.container = None
            self._data = None
            self._mmap = None
            self._mmap_path = None
            self._label = torch.tensor(0, dtype=torch.long)
            return
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
        # A cohort: find the container this index falls in and hand over.
        if self._parts is not None:
            if idx < 0:
                idx += self.num_tiles
            if not 0 <= idx < self.num_tiles:
                raise IndexError(f"index {idx} out of range for {self.num_tiles} tiles")
            import bisect
            part = bisect.bisect_right(self._offsets, idx) - 1
            return self._parts[part][idx - self._offsets[part]]

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
        """Clean up resources.

        Defensive about its own attributes on purpose: __del__ calls this, and
        __del__ runs on objects whose constructor raised part-way through.
        """
        if getattr(self, "_parts", None):
            for part in self._parts:
                part.close()
            return
        if getattr(self, 'container', None) is not None and self.container._file:
            self.container.close()
        mmap_path = getattr(self, "_mmap_path", None)
        if mmap_path and mmap_path.exists():
            mmap_path.unlink(missing_ok=True)

    def __del__(self):
        self.close()


class ManifestOmniaDataset(Dataset):
    """One item per SLIDE (not per tile), labeled from an external manifest.

    Plain `OmniaDataset` over a directory hardcodes every sample's label to
    `0` (see above) — it's an I/O-speed layer with no concept of a
    classification target. That's fine for benchmarking, but silently
    unusable for training: point it at a labeled cohort and it trains
    against a constant, which would burn a run without ever raising an
    error.

    This wraps a directory of `.omnia` containers (one per slide) plus a
    manifest CSV in the schema `slide_id,slide_path,isup_grade,
    dataset_source,scanner_label` (the format `build_manifest.py` in
    omnia-AI/panda-training/kaggle writes) and resolves each container's
    label by filename stem == slide_id. A container with no matching
    manifest row — or a strict-mode manifest row with no matching container
    — raises at construction time rather than falling back to a default
    label. A silent default here is exactly the "label=0 fallback" bug this
    class exists to rule out; it must fail loudly, at dataset-build time
    (seconds, on the machine building the manifest), not partway through a
    billed training run.

    Each item is a slide's full tile bag (matching what a MIL trainer like
    runpod_train_attn_mil.py's BagDataset expects: one bag == one slide) as
    `(tiles, isup_grade, scanner_label)`, where `tiles` has the shape/dtype
    that the underlying single-file `OmniaDataset` would return for
    `_data` — pre-normalized float, `(N, 3, H, W)`.

    Args:
        omnia_dir: directory of `.omnia` files, one per slide, named
            `<slide_id>.omnia`.
        manifest_csv: path to the manifest CSV.
        cache_mode: forwarded to each per-slide `OmniaDataset` ("ram" or
            "mmap" — "none" isn't supported here since a full bag needs all
            of a slide's tiles at once, not one tile at a time).
        strict: if True, raise when a manifest row has no matching `.omnia`
            file (catches a manifest built against a different/incomplete
            conversion run). Default True. This is the ONLY join direction
            checked here — the manifest is authoritative and drives what
            gets loaded; a `.omnia` file in `omnia_dir` that no row in THIS
            manifest mentions is not an error; the manifest may legitimately
            be a subset of everything the directory holds (a train/val/inner
            split all pointed at one shared container directory, each with
            its own subset manifest, is the normal case this is built for —
            not a bug to guard against). A directory-wide "every container
            has a label somewhere" check belongs in build_manifest.py's
            audit against the FULL manifest, not here against one split's.
    """

    def __init__(self, omnia_dir: Union[str, Path], manifest_csv: Union[str, Path],
                 cache_mode: str = "ram", normalize: float = 255.0, strict: bool = True):
        if cache_mode not in ("ram", "mmap"):
            raise ValueError(f"ManifestOmniaDataset needs a full per-slide bag at once; "
                              f"cache_mode must be 'ram' or 'mmap', got {cache_mode!r}")

        omnia_dir = Path(omnia_dir)
        if not omnia_dir.is_dir():
            raise FileNotFoundError(f"{omnia_dir} is not a directory")

        manifest_rows = []
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            required = {"slide_id", "isup_grade", "dataset_source", "scanner_label"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{manifest_csv} is missing required columns "
                                  f"(need {required}, got {reader.fieldnames})")
            for r in reader:
                manifest_rows.append(r)
        if not manifest_rows:
            raise ValueError(f"{manifest_csv} has zero rows")

        from collections import Counter
        id_counts = Counter(r["slide_id"] for r in manifest_rows)
        dup = sorted(sid for sid, n in id_counts.items() if n > 1)
        if dup:
            raise ValueError(f"{manifest_csv} has duplicate slide_id(s): {dup[:10]}")

        missing_container = [r["slide_id"] for r in manifest_rows
                              if not (omnia_dir / f"{r['slide_id']}.omnia").exists()]
        if missing_container:
            preview = ", ".join(missing_container[:10])
            msg = (f"{len(missing_container)} manifest row(s) in {manifest_csv} have no "
                   f"matching .omnia file in {omnia_dir} (first 10: {preview}). This usually "
                   f"means the conversion step didn't finish, or the manifest was built "
                   f"against a different dataset snapshot.")
            if strict:
                raise KeyError(msg + " Pass strict=False to skip these rows instead "
                                      "(not recommended without understanding why they're missing).")
            print(f"WARNING: {msg} Skipping them (strict=False).")

        self.slide_ids = []
        self._parts = []
        self._grades = []
        self._scanners = []
        self._sources = []
        bad_grade = []
        for r in manifest_rows:
            sid = r["slide_id"]
            f = omnia_dir / f"{sid}.omnia"
            if not f.exists():
                continue  # already reported above; only reachable when strict=False
            grade = int(r["isup_grade"])
            if not 0 <= grade <= 5:
                bad_grade.append((sid, grade))
                continue
            self.slide_ids.append(sid)
            self._parts.append(OmniaDataset(f, cache_mode=cache_mode, normalize=normalize))
            self._grades.append(grade)
            self._scanners.append(int(r["scanner_label"]))
            self._sources.append(r["dataset_source"])
        if bad_grade:
            raise ValueError(f"{len(bad_grade)} manifest row(s) have isup_grade outside [0,5]: "
                              f"{bad_grade[:10]}")

    def __len__(self) -> int:
        return len(self._parts)

    def __getitem__(self, idx: int):
        part = self._parts[idx]
        grade = self._grades[idx]
        scanner = self._scanners[idx]
        if part._data is not None:
            tiles = part._data
        elif part._mmap is not None:
            tiles = torch.from_numpy(part._mmap.astype(np.float32)).permute(0, 3, 1, 2) / part.normalize
        else:
            raise RuntimeError(
                f"{self.slide_ids[idx]}.omnia did not preload (cache_mode='none' fallback, "
                f"e.g. a native JP2K container this SDK can't decode) — ManifestOmniaDataset "
                f"needs a full in-memory/mmap bag per slide, not per-tile decompression."
            )
        return tiles, grade, scanner

    def close(self):
        for part in getattr(self, "_parts", []):
            part.close()

    def __del__(self):
        self.close()
