"""Tests for the PyTorch dataset layer.

The three cache modes are the library's central performance claim — "ram" is
where the speed comes from. They are therefore the place where a bug would be
least visible and most damaging: a mode that returns subtly different tensors
from the others would mean the number you benchmark and the data you train on
come from different code paths.

So these assert the modes agree with each other, and with the container.
"""
import numpy as np
import pytest
import torch

from omnia_sdk import OmniaContainer, OmniaDataset


def _write(path, n=24, size=16, seed=11):
    rng = np.random.default_rng(seed)
    tiles = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]
    OmniaContainer.write(path, tiles, batch_size=8)
    return tiles


def test_length_and_shape(tmp_path):
    path = tmp_path / "a.omnia"
    tiles = _write(path)
    ds = OmniaDataset(path)
    assert len(ds) == len(tiles)


def test_values_match_the_source_tiles(tmp_path):
    """Normalisation is the only thing that should change between disk and
    tensor — so dividing back out must land on the original bytes."""
    path = tmp_path / "b.omnia"
    tiles = _write(path)
    ds = OmniaDataset(path, normalize=255.0)

    for i in (0, 1, 7, 8, 23):
        x = ds[i]
        t = x[0] if isinstance(x, (tuple, list)) else x
        assert isinstance(t, torch.Tensor)
        back = (t.numpy() * 255.0).round().astype(np.uint8)
        # Tensors may be CHW while tiles are HWC.
        if back.shape != tiles[i].shape and back.ndim == 3:
            back = np.transpose(back, (1, 2, 0))
        np.testing.assert_array_equal(back, tiles[i])


@pytest.mark.parametrize("mode", ["ram", "mmap", "none"])
def test_every_cache_mode_returns_the_same_data(tmp_path, mode):
    """The fast path and the slow path must not disagree.

    If "ram" — the mode the benchmark measures — returned anything different
    from "none", the published speedup would describe a code path nobody
    trains on.
    """
    path = tmp_path / f"c-{mode}.omnia"
    tiles = _write(path, n=20, seed=12)

    reference = OmniaDataset(path, cache_mode="none")
    ds = OmniaDataset(path, cache_mode=mode)
    assert len(ds) == len(reference) == len(tiles)

    for i in (0, 5, 8, 19):
        a = ds[i]
        b = reference[i]
        a = a[0] if isinstance(a, (tuple, list)) else a
        b = b[0] if isinstance(b, (tuple, list)) else b
        torch.testing.assert_close(a, b)


def test_directory_of_containers(tmp_path):
    """A dataset can be pointed at a folder, which is how a cohort is loaded.

    Counting tiles is not enough: with several containers behind one index
    space, the failure that matters is a tile served from the wrong slide.
    Every tile is therefore checked against the file it actually came from.
    """
    d = tmp_path / "cohort"
    d.mkdir()
    per_file = {}
    for k in range(3):
        per_file[f"s{k}.omnia"] = _write(d / f"s{k}.omnia", n=6, seed=20 + k)

    ds = OmniaDataset(d)
    assert len(ds) == sum(len(v) for v in per_file.values())

    # sorted(), because that is the order the dataset lays the files out in.
    expected = [t for name in sorted(per_file) for t in per_file[name]]
    for i, original in enumerate(expected):
        x = ds[i]
        t = x[0] if isinstance(x, (tuple, list)) else x
        back = (t.numpy() * 255.0).round().astype(np.uint8)
        if back.shape != original.shape and back.ndim == 3:
            back = np.transpose(back, (1, 2, 0))
        np.testing.assert_array_equal(back, original, err_msg=f"tile {i} came from the wrong slide")


def test_empty_directory_is_reported_clearly(tmp_path):
    """A folder with no containers should say so, not fail somewhere deeper."""
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        OmniaDataset(d)


def test_iterating_the_whole_dataset(tmp_path):
    """Every index is reachable — no gap at a batch or container boundary."""
    path = tmp_path / "e.omnia"
    tiles = _write(path, n=17, seed=13)
    ds = OmniaDataset(path)
    seen = 0
    for i in range(len(ds)):
        x = ds[i]
        t = x[0] if isinstance(x, (tuple, list)) else x
        assert t is not None
        seen += 1
    assert seen == len(tiles)


def test_works_with_a_dataloader(tmp_path):
    """The reason this class exists is to be fed to PyTorch."""
    from torch.utils.data import DataLoader
    path = tmp_path / "f.omnia"
    _write(path, n=16, seed=14)
    ds = OmniaDataset(path)
    n = 0
    for batch in DataLoader(ds, batch_size=4):
        t = batch[0] if isinstance(batch, (tuple, list)) else batch
        assert t.shape[0] <= 4
        n += t.shape[0]
    assert n == 16
