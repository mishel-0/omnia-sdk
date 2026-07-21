"""
PyTorch Datasets for .omnia Zstd containers.
Worker-safe: each worker creates its own containers on __init__.
Supports pin_memory, prefetch, persistent_workers, multi-GPU.
"""
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom

from container import OmniaContainer


class OmniaDataset(Dataset):
    """PyTorch Dataset over .omnia files.
    
    Worker-safe: each DataLoader worker creates its own OmniaContainer
    instances, so file handles are never shared across processes.
    
    Usage:
        ds = OmniaDataset("/path/to/omnia/files/")
        loader = DataLoader(
            ds, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )
        for images, labels in loader:
            ...
    """

    def __init__(self, omnia_dir: Union[str, Path], normalize: float = 4095.0):
        self.omnia_dir = Path(omnia_dir)
        self.normalize = normalize
        self.metas: list[dict] = []  # per-file metadata for worker init
        self.index: list[tuple[int, int]] = []  # (file_idx, slice_idx)
        self._containers: list[OmniaContainer] | None = None

        omnia_files = sorted(self.omnia_dir.glob("*.omnia"))
        if not omnia_files:
            raise FileNotFoundError(f"No .omnia files found in {omnia_dir}")

        # Store just the metadata — workers will open their own handles
        for fi, omnia_path in enumerate(omnia_files):
            c = OmniaContainer(omnia_path)
            c.open()
            self.metas.append({
                "path": str(omnia_path),
                "num_slices": c.num_slices,
                "dtype": c.dtype,
                "shape": c.shape,
            })
            for si in range(c.num_slices):
                self.index.append((fi, si))
            c.close()  # Workers will re-open

        # Pre-allocate containers on main process too (for single-worker use)
        self._containers = None

    def _lazy_init(self):
        """Lazy open of containers (called once per worker process)."""
        if self._containers is not None:
            return
        self._containers = []
        for m in self.metas:
            c = OmniaContainer(m["path"])
            c.open()
            self._containers.append(c)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        self._lazy_init()
        fi, si = self.index[idx]
        arr = self._containers[fi].get_slice(si)
        tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0) / self.normalize
        return tensor, torch.tensor(0, dtype=torch.long)

    def close_all(self):
        if self._containers:
            for c in self._containers:
                c.close()
            self._containers = None

    def __del__(self):
        self.close_all()


class DicomDataset(Dataset):
    """PyTorch Dataset over raw DICOM files. For benchmarking only."""

    def __init__(self, dicom_dir: Union[str, Path], normalize: float = 4095.0):
        self.normalize = normalize
        self.files: list[Path] = []

        study_dirs = [d for d in Path(dicom_dir).iterdir() if d.is_dir()]
        for sd in study_dirs:
            for f in sorted(sd.rglob("*.dcm")):
                try:
                    ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
                    if getattr(ds, "Modality", "") == "CT":
                        self.files.append(f)
                except Exception:
                    pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ds = pydicom.dcmread(str(self.files[idx]), force=True)
        arr = ds.pixel_array.astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0) / self.normalize
        return tensor, torch.tensor(0, dtype=torch.long)
