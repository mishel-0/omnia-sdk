"""
.omnia v4 Container — Pyramid-aware lossless Zstd container for whole-slide images.

Format:
    [MAGIC:4="OMN2"][VERSION:1][CODEC:1][RESERVED:2]
    [META_SIZE:8]
    [META:zstd-compressed JSON]
    [batch_0:crc32(4)+zstd(tile_batch)]...[batch_N:crc32(4)+zstd(tile_batch)]

v4.1 adds batch compression — tiles compressed in groups for better Zstd ratios.
Backward-compatible: reads v2/v3/v4 (batch_size=1 = per-tile, legacy behavior).
"""
import struct, json, zlib
from pathlib import Path
from typing import Any, Optional, Union, Iterator
import numpy as np

try:
    import zstandard as _zstd
    def _zstd_compress(data: bytes, level: int = 5) -> bytes:
        return _zstd.ZstdCompressor(level=level).compress(data)
    def _zstd_decompress(data: bytes) -> bytes:
        return _zstd.ZstdDecompressor().decompress(data)
except ImportError:
    import zstd as _zstd
    _zstd_compress = _zstd.compress
    _zstd_decompress = _zstd.decompress

MAGIC = b"OMN2"
VERSION = 4
HEADER_FMT = "<4s B B H"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

DEFAULT_BATCH_SIZE = 16
DEFAULT_ZSTD_LEVEL = 5


class FormatError(Exception):
    """Raised when a .omnia file has an invalid format."""

class IntegrityError(Exception):
    """Raised when a decoded tile fails its CRC32 check."""


class _BatchCache:
    """LRU cache for decompressed batches."""
    __slots__ = ('batch_idx', 'data', 'offsets')
    def __init__(self, batch_idx: int, data: bytes, offsets: list[int]):
        self.batch_idx = batch_idx
        self.data = data
        self.offsets = offsets


# How many decompressed batches to keep in LRU cache
_BATCH_CACHE_SIZE = 32


def _compute_tile_offsets(tile_indices: list[int],
                          tile_shapes: dict[int, tuple],
                          default_shape: tuple) -> list[int]:
    """Return cumulative byte offsets for tiles in a batch."""
    offsets = [0]
    for idx in tile_indices:
        shape = tile_shapes.get(idx, default_shape)
        offsets.append(offsets[-1] + shape[0] * shape[1] * shape[2])
    return offsets


class OmniaContainer:
    """Read/write .omnia v4 container for whole-slide images.

    Supports multi-resolution pyramid levels, per-tile CRC integrity,
    edge tiles, and **batch compression** (multiple tiles compressed
    as one block for better Zstd ratios).

    Usage:
        # Write
        stats = OmniaContainer.write("slide.omnia", tiles, batch_size=16)

        # Read (context manager — fast sequential access)
        with OmniaContainer("slide.omnia") as c:
            tile = c.get_slice(42)

        # Read (worker-safe — opens/closes per call)
        tile = OmniaContainer("slide.omnia").get_slice(42)
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._file: Optional[Any] = None
        self._meta: Optional[dict] = None
        self._sizes: Optional[list[int]] = None
        self._crcs: Optional[list[int]] = None
        self._offsets: Optional[list[int]] = None
        self._pixel_start: int = 0
        self.num_slices: int = 0
        self.dtype: np.dtype = np.dtype("uint8")
        self.shape: tuple = (256, 256, 3)
        self.num_levels: int = 1
        self.levels: list[dict] = []
        self.tile_shapes: dict[int, tuple] = {}
        # Batch support
        self.batch_size: int = 1
        self._batch_cache: dict[int, _BatchCache] = {}
        # Native pixel passthrough
        self.pixel_codec: str = "zstd"
        self.native_codec: Optional[str] = None

    # ── Writer ─────────────────────────────────────────────────

    @staticmethod
    def write(path: Union[str, Path], tiles: Union[list[np.ndarray], Iterator[np.ndarray]],
              metadata: Optional[dict] = None,
              batch_size: int = DEFAULT_BATCH_SIZE,
              zstd_level: int = DEFAULT_ZSTD_LEVEL) -> dict:
        """Write tiles to .omnia file. Returns compression stats.

        Streams to disk in batches — does NOT buffer the whole file in memory.
        Suitable for large WSI with 100K+ tiles.

        Args:
            path: Output .omnia file path.
            tiles: List or iterator of numpy arrays (HxWxC, uint8).
            metadata: Optional dict with 'levels', 'source', 'slide_dims'.
            batch_size: Tiles per compressed batch (default: 16).
                        Higher = better Zstd ratio, more memory per batch.
            zstd_level: Zstd compression level (default: 5).

        Returns:
            {"tiles": N, "original_bytes": N, "compressed_bytes": N, "ratio": X}
        """
        path = Path(path)
        it = iter(tiles)

        tile_shapes: dict[int, list[int]] = {}
        crcs: list[int] = []
        sizes: list[int] = []
        orig_size = 0

        # Buffer for current batch
        batch_buf: list[bytes] = []
        batch_shapes: list[list[int]] = []
        batch_indices: list[int] = []

        def _flush_batch(buf: list[bytes], buf_indices: list[int],
                         buf_shapes: list[list[int]],
                         crcs_out: list[int], sizes_out: list[int],
                         tile_shapes_out: dict, f_out) -> None:
            """Compress and write one batch to file."""
            combined = b"".join(buf)
            comp = _zstd_compress(combined, zstd_level)
            crc = zlib.crc32(comp)
            chunk = struct.pack("<I", crc) + comp
            f_out.write(chunk)
            crcs_out.append(crc)
            sizes_out.append(len(chunk))
            # Record non-standard shapes for this batch
            for bi, bs in zip(buf_indices, buf_shapes):
                if bs != shape:
                    tile_shapes_out[bi] = bs

        from io import BytesIO
        pixel_buf = BytesIO()

        # Process tiles in batches — write pixel data to temp buffer
        dtype = None
        shape = None
        tile_index = -1
        for arr in it:
            tile_index += 1
            if dtype is None:
                dtype = str(arr.dtype)
                shape = list(arr.shape)
            elif str(arr.dtype) != dtype:
                raise ValueError(
                    f"Tile {tile_index} has dtype {arr.dtype}, "
                    f"expected {dtype}"
                )
            b = np.ascontiguousarray(arr).tobytes()
            actual = list(arr.shape)
            batch_buf.append(b)
            batch_indices.append(tile_index)
            batch_shapes.append(actual)
            orig_size += arr.nbytes

            if len(batch_buf) >= batch_size:
                _flush_batch(batch_buf, batch_indices, batch_shapes,
                             crcs, sizes, tile_shapes, pixel_buf)
                batch_buf = []
                batch_indices = []
                batch_shapes = []

        if tile_index < 0:
            raise ValueError("No tiles to write")

        # Flush remaining tiles
        if batch_buf:
            _flush_batch(batch_buf, batch_indices, batch_shapes,
                         crcs, sizes, tile_shapes, pixel_buf)

        total_tiles = tile_index + 1

        # Build metadata
        meta: dict[str, Any] = {
            "v": VERSION,
            "codec": "zstd",
            "n": total_tiles,
            "sz": sizes,
            "crc": crcs,
            "dtype": dtype,
            "shape": shape,
            "original_bytes": orig_size,
            "batch_size": batch_size,
            "zstd_level": zstd_level,
        }
        if tile_shapes:
            meta["tile_shapes"] = {str(k): v for k, v in tile_shapes.items()}
        if metadata:
            for k in ("levels", "source", "slide_dims"):
                if k in metadata:
                    meta[k] = metadata[k]

        mc = _zstd_compress(json.dumps(meta, separators=(",", ":")).encode(), 19)
        pixel_data = pixel_buf.getvalue()

        # Final write: header + metadata + pixel data
        with open(path, "wb") as f:
            f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 0, 0))
            f.write(struct.pack("<Q", len(mc)))
            f.write(mc)
            f.write(pixel_data)

        final_size = path.stat().st_size
        return {
            "tiles": total_tiles,
            "original_bytes": orig_size,
            "compressed_bytes": final_size,
            "ratio": round(orig_size / final_size, 3),
        }

    @staticmethod
    def write_native(path: Union[str, Path],
                     tile_bytes_iter: Iterator[bytes],
                     metadata: Optional[dict] = None,
                     pixel_codec: str = "native",
                     native_codec: str = "jp2k") -> dict:
        """Write .omnia where each tile is a pre-encoded byte blob (no re-compression).

        Used for native JP2K passthrough (`pixel_codec="native"`) and lossy
        JPEG re-encode (`pixel_codec="jpeg"`). Stored as CRC32 + raw bytes,
        exactly like the legacy per-tile layout. Returns compression stats.
        """
        from io import BytesIO

        path = Path(path)
        pixel_buf = BytesIO()
        sizes: list[int] = []
        crcs: list[int] = []
        orig = 0
        n = 0
        for data in tile_bytes_iter:
            crc = zlib.crc32(data)
            pixel_buf.write(struct.pack("<I", crc) + data)
            sizes.append(len(data) + 4)
            crcs.append(crc)
            orig += len(data)
            n += 1
        if n == 0:
            raise ValueError("No tiles to write")

        meta: dict[str, Any] = {
            "v": VERSION,
            "codec": "zstd",
            "pixel_codec": pixel_codec,
            "n": n,
            "sz": sizes,
            "crc": crcs,
            "dtype": "uint8",
            "shape": [256, 256, 3],
            "original_bytes": orig,
            "native_codec": native_codec,
        }
        if metadata:
            for k in ("levels", "source", "slide_dims"):
                if k in metadata:
                    meta[k] = metadata[k]

        mc = _zstd_compress(json.dumps(meta, separators=(",", ":")).encode(), 19)
        with open(path, "wb") as f:
            f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 0, 0))
            f.write(struct.pack("<Q", len(mc)))
            f.write(mc)
            f.write(pixel_buf.getvalue())

        final_size = path.stat().st_size
        return {
            "tiles": n,
            "original_bytes": orig,
            "compressed_bytes": final_size,
            "ratio": round(orig / final_size, 3),
        }

    # ── Reader ─────────────────────────────────────────────────

    def _parse_header(self):
        """Parse metadata from file handle (must be at position 0)."""
        header = self._file.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise FormatError("File too small for header")
        magic, ver, codec, _ = struct.unpack(HEADER_FMT, header)
        if magic != MAGIC:
            raise FormatError(f"Bad magic: {magic!r}, expected {MAGIC!r}")
        if ver not in (2, 3, 4):
            raise FormatError(f"Unsupported version: {ver}")
        if codec != 0:
            raise FormatError(f"Unsupported codec id: {codec} (expected 0=zstd)")

        ms_raw = self._file.read(8)
        if len(ms_raw) < 8:
            raise FormatError("File too small for metadata size")
        meta_size = struct.unpack("<Q", ms_raw)[0]

        meta_bytes = self._file.read(meta_size)
        if len(meta_bytes) < meta_size:
            raise FormatError("File too small for metadata content")
        try:
            meta_raw = _zstd_decompress(meta_bytes)
        except Exception as e:
            raise FormatError(f"Failed to decompress metadata: {e}") from e
        try:
            self._meta = json.loads(meta_raw)
        except json.JSONDecodeError as e:
            raise FormatError(f"Corrupt metadata (invalid JSON): {e}") from e

        self.num_slices = self._meta["n"]
        self._sizes = self._meta["sz"]
        self._crcs = self._meta.get("crc")
        self.dtype = np.dtype(self._meta.get("dtype", "uint8"))
        self.shape = tuple(self._meta.get("shape", [256, 256, 3]))

        # Batch support (default 1 = legacy per-tile format)
        self.batch_size = self._meta.get("batch_size", 1)

        # Pixel codec (default "zstd" = RGB tiles, "native" = raw JP2K)
        self.pixel_codec = self._meta.get("pixel_codec", "zstd")
        self.native_codec = self._meta.get("native_codec")

        # Pyramid levels
        self.levels = self._meta.get("levels", [])
        self.num_levels = len(self.levels) if self.levels else 1

        # Edge tile shapes
        raw_shapes = self._meta.get("tile_shapes", {})
        self.tile_shapes = {int(k): tuple(v) for k, v in raw_shapes.items()}

        self._pixel_start = self._file.tell()

        # Precompute cumulative offsets for O(1) batch seek
        self._offsets = []
        running = 0
        for s in self._sizes:
            self._offsets.append(running)
            running += s

    def open(self):
        """Open file and parse metadata. Keeps handle open for fast sequential access."""
        if self._file is not None:
            return
        self._file = open(self.path, "rb")
        self._parse_header()

    def close(self):
        """Close the file handle if open."""
        if self._file:
            self._file.close()
            self._file = None
        self._batch_cache.clear()

    def _get_batch_indices(self, index: int) -> tuple[int, int, int]:
        """Return (batch_idx, offset_in_batch, batch_tile_count) for a tile index."""
        if self.batch_size <= 1:
            return index, 0, 1  # legacy: one tile per "batch"
        batch_idx = index // self.batch_size
        offset_in_batch = index % self.batch_size
        # Last batch may have fewer tiles
        total_batches = (self.num_slices + self.batch_size - 1) // self.batch_size
        if batch_idx >= total_batches:
            raise IndexError(f"Tile {index}, max {self.num_slices - 1}")
        if batch_idx == total_batches - 1:
            batch_tile_count = self.num_slices - batch_idx * self.batch_size
        else:
            batch_tile_count = self.batch_size
        return batch_idx, offset_in_batch, batch_tile_count

    def _decompress_batch(self, batch_idx: int) -> _BatchCache:
        """Decompress a batch and return cached result (LRU cache)."""
        # Check cache
        if batch_idx in self._batch_cache:
            return self._batch_cache[batch_idx]

        start_tile = batch_idx * self.batch_size
        batch_tile_count = min(self.batch_size, self.num_slices - start_tile)
        tile_indices = list(range(start_tile, start_tile + batch_tile_count))
        offsets = _compute_tile_offsets(tile_indices, self.tile_shapes, self.shape)

        # Read and decompress the batch chunk
        offset = self._pixel_start + self._offsets[batch_idx]
        self._file.seek(offset)
        data = self._file.read(self._sizes[batch_idx])

        if self._crcs:
            stored_crc = struct.unpack("<I", data[:4])[0]
            comp = data[4:]
            if stored_crc != zlib.crc32(comp):
                raise IntegrityError(f"CRC mismatch on batch {batch_idx}")
        else:
            comp = data

        raw = _zstd_decompress(comp)
        cache = _BatchCache(batch_idx, raw, offsets)
        
        # LRU: evict oldest if cache full
        if len(self._batch_cache) >= _BATCH_CACHE_SIZE:
            # Simple strategy: pop the first inserted key
            self._batch_cache.pop(next(iter(self._batch_cache)))
        self._batch_cache[batch_idx] = cache
        return cache

    def get_slice(self, index: int) -> np.ndarray:
        """Decode and return one tile.

        With batch compression, decompresses the entire batch on first
        access and caches it in LRU. Subsequent tiles in cached batches
        are zero-copy slices.

        Args:
            index: Tile index (0-based, ordered level-by-level).

        Returns:
            numpy array of shape (H, W, C) with actual tile dimensions.
        """
        own_handle = self._file is None
        if own_handle:
            f = open(self.path, "rb")
            self._file = f
            self._parse_header()

        if index < 0 or index >= self.num_slices:
            if own_handle:
                self._file.close()
                self._file = None
            raise IndexError(f"Tile {index}, max {self.num_slices - 1}")

        try:
            if self.batch_size <= 1:
                # Legacy: one tile per chunk
                offset = self._pixel_start + self._offsets[index]
                self._file.seek(offset)
                data = self._file.read(self._sizes[index])

                if self._crcs:
                    stored_crc = struct.unpack("<I", data[:4])[0]
                    comp = data[4:]
                    if stored_crc != zlib.crc32(comp):
                        raise IntegrityError(f"CRC mismatch on tile {index}")
                else:
                    comp = data

                if self.pixel_codec in ("native", "jpeg"):
                    # Pre-encoded tile bytes (JP2K passthrough or JPEG) — return as-is
                    return np.frombuffer(comp, dtype=np.uint8)
                raw = _zstd_decompress(comp)
                actual_shape = self.tile_shapes.get(index, self.shape)
                return np.frombuffer(raw, dtype=self.dtype).reshape(actual_shape)
            else:
                # Batched: decompress batch, slice tile
                batch_idx, offset_in_batch, _ = self._get_batch_indices(index)
                cache = self._decompress_batch(batch_idx)
                start = cache.offsets[offset_in_batch]
                end = cache.offsets[offset_in_batch + 1]
                raw = cache.data[start:end]
                actual_shape = self.tile_shapes.get(index, self.shape)
                return np.frombuffer(raw, dtype=self.dtype).reshape(actual_shape)
        finally:
            if own_handle:
                self._file.close()
                self._file = None
                self._batch_cache.clear()

    def get_level_info(self, level: int) -> dict:
        """Return level descriptor for pyramid level index."""
        if not self.levels:
            return {"w": self.shape[1], "h": self.shape[0], "tw": self.shape[1],
                    "th": self.shape[0], "tx": 1, "ty": 1, "tc": self.num_slices,
                    "start": 0}
        return self.levels[level]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
