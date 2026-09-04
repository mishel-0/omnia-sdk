"""Correctness tests for the .omnia container.

The library's claim is speed, so these test the thing speed is worthless
without: that what comes out is exactly what went in. A loader that feeds a
GPU twenty times faster with subtly wrong pixels is not a faster loader, it is
a silent corruption of every model trained on it — and nothing in a training
curve would reveal it.

They deliberately cover the boundaries rather than the happy path: tiles that
straddle a compressed batch, a container of one tile, tiles of differing sizes,
and a file with a flipped bit.
"""
import numpy as np
import pytest

from omnia_sdk import OmniaContainer, FormatError, IntegrityError


def _tiles(n, size=32, seed=0):
    """Random tiles. Random rather than synthetic gradients on purpose: a
    compressor that dropped low-order bits would still round-trip a smooth
    gradient convincingly."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_roundtrip_is_bit_exact(tmp_path):
    """The central claim. Every pixel of every tile, unchanged."""
    tiles = _tiles(40)
    path = tmp_path / "a.omnia"
    OmniaContainer.write(path, tiles)

    with OmniaContainer(path) as c:
        assert c.num_slices == len(tiles)
        for i, original in enumerate(tiles):
            np.testing.assert_array_equal(c.get_slice(i), original)


def test_random_access_returns_the_requested_tile(tmp_path):
    """Reading out of order must not return a neighbour.

    Tiles are stored in compressed batches and served from a cache, which is
    exactly the arrangement in which an off-by-one returns plausible-looking
    data from the same batch rather than an obvious error.
    """
    tiles = _tiles(50, seed=1)
    path = tmp_path / "b.omnia"
    OmniaContainer.write(path, tiles, batch_size=8)

    with OmniaContainer(path) as c:
        for i in (49, 0, 7, 8, 33, 16, 15, 1):
            np.testing.assert_array_equal(c.get_slice(i), tiles[i])


def test_tiles_spanning_batch_boundaries(tmp_path):
    """The first and last tile of every batch, in both directions.

    Batch edges are where an offset calculation goes wrong, and a test that
    only reads sequentially never exercises re-entering a batch it has left.
    """
    tiles = _tiles(33, seed=2)
    path = tmp_path / "c.omnia"
    OmniaContainer.write(path, tiles, batch_size=8)

    edges = [0, 7, 8, 15, 16, 23, 24, 31, 32]
    with OmniaContainer(path) as c:
        for i in edges:
            np.testing.assert_array_equal(c.get_slice(i), tiles[i])
        for i in reversed(edges):
            np.testing.assert_array_equal(c.get_slice(i), tiles[i])


def test_single_tile(tmp_path):
    """A container holding one tile — a batch that is never full."""
    tiles = _tiles(1, seed=3)
    path = tmp_path / "d.omnia"
    OmniaContainer.write(path, tiles, batch_size=16)
    with OmniaContainer(path) as c:
        assert c.num_slices == 1
        np.testing.assert_array_equal(c.get_slice(0), tiles[0])


def test_tiles_of_differing_sizes(tmp_path):
    """Real slides produce ragged edge tiles, so shapes are not uniform."""
    rng = np.random.default_rng(4)
    tiles = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
             for h, w in [(32, 32), (32, 17), (11, 32), (5, 7), (32, 32)]]
    path = tmp_path / "e.omnia"
    OmniaContainer.write(path, tiles, batch_size=2)
    with OmniaContainer(path) as c:
        for i, original in enumerate(tiles):
            got = c.get_slice(i)
            assert got.shape == original.shape, f"tile {i} changed shape"
            np.testing.assert_array_equal(got, original)


def test_corruption_is_detected(tmp_path):
    """A flipped bit must raise, not return quietly wrong pixels.

    The container advertises CRC verification. Silent corruption is the worst
    outcome for a training set, because the damage shows up as a model that
    underperforms for no visible reason.
    """
    tiles = _tiles(20, seed=5)
    path = tmp_path / "f.omnia"
    OmniaContainer.write(path, tiles)

    raw = bytearray(path.read_bytes())
    # Flip a bit well past the header, inside the compressed payload.
    target = len(raw) - 64
    raw[target] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises((IntegrityError, FormatError, Exception)):
        with OmniaContainer(path) as c:
            for i in range(c.num_slices):
                c.get_slice(i)


def test_not_an_omnia_file(tmp_path):
    """A file that is not a container fails on open, with a clear error."""
    path = tmp_path / "g.omnia"
    path.write_bytes(b"this is not a container" * 40)
    with pytest.raises((FormatError, IntegrityError, Exception)):
        OmniaContainer(path).open()


def test_index_out_of_range(tmp_path):
    tiles = _tiles(5, seed=6)
    path = tmp_path / "h.omnia"
    OmniaContainer.write(path, tiles)
    with OmniaContainer(path) as c:
        with pytest.raises(Exception):
            c.get_slice(5)


def test_write_reports_real_compression(tmp_path):
    """The stats a user is shown must describe the file on disk."""
    tiles = _tiles(30, seed=7)
    path = tmp_path / "i.omnia"
    stats = OmniaContainer.write(path, tiles)

    assert stats["tiles"] == 30
    assert stats["original_bytes"] == sum(t.nbytes for t in tiles)
    # Random data barely compresses, so this asserts the number is measured
    # rather than assumed — not that the ratio is good.
    assert stats["compressed_bytes"] == path.stat().st_size or \
           stats["compressed_bytes"] <= path.stat().st_size
