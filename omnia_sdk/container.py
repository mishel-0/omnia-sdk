"""
.omnia Training Edition — Zstd container format
One file replaces 277+ DICOM files. Lossless Zstd compression.
Optimized for training: fast per-slice random access, worker-safe.
"""
import struct, json, zlib
from pathlib import Path
from typing import Optional, Union
import numpy as np

# ── Zstd import: prefer 'zstandard' (pip install zstandard), fallback 'zstd' ──
try:
    import zstandard as _zstd
    def _zstd_compress(b: bytes, level: int = 3) -> bytes:
        return _zstd.ZstdCompressor(level=level).compress(b)
    def _zstd_decompress(b: bytes) -> bytes:
        return _zstd.ZstdDecompressor().decompress(b)
except ImportError:
    import zstd as _zstd  # legacy fallback
    _zstd_compress = _zstd.compress
    _zstd_decompress = _zstd.decompress


MAGIC = b"OMN2"
VERSION = 3  # Training edition (Zstd codec)
HEADER_FMT = "<4s B B H"  # magic(4) + version(1) + codec(1) + reserved(2)
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8 bytes


class FormatError(Exception):
    """Raised when a .omnia file has an invalid format."""


class IntegrityError(Exception):
    """Raised when a decoded slice fails its CRC32 check."""


class OmniaContainer:
    """Read/write .omnia Zstd container for training.
    
    Worker-safe: each get_slice() opens the file if needed, seeks, reads,
    decompresses, and does NOT keep a persistent cross-call handle.
    For single-worker use, use the context manager for efficiency.
    
    Format:
        [MAGIC:4="OMN2"][VERSION:1][CODEC:1][RESERVED:2]
        [META_SIZE:8]
        [META:zstd-compressed JSON]
        [chunk_0:crc32(4)+zstd(pixels)]...[chunk_N:crc32(4)+zstd(pixels)]
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._file: Optional[open] = None
        self._meta: Optional[dict] = None
        self._sizes: Optional[list[int]] = None
        self._crcs: Optional[list[int]] = None
        self._offsets: Optional[list[int]] = None
        self._pixel_start: int = 0
        self.num_slices: int = 0
        self.dtype: np.dtype = np.dtype("int16")
        self.shape: tuple = (512, 512)

    # ── Writer ─────────────────────────────────────────────────

    @staticmethod
    def write(path: Union[str, Path], slices: list[np.ndarray],
              metadata: Optional[list[dict]] = None) -> dict:
        """Write slices to .omnia file. Returns stats dict.
        
        Each chunk stored as [CRC32:4][zstd(pixel_data)].
        """
        path = Path(path)
        orig_size = sum(s.nbytes for s in slices)
        dtype = str(slices[0].dtype)
        shape = list(slices[0].shape)

        chunks: list[bytes] = []
        sizes: list[int] = []
        crcs: list[int] = []
        for arr in slices:
            comp = _zstd_compress(np.ascontiguousarray(arr).tobytes(), 3)
            crc = zlib.crc32(comp)
            chunk = struct.pack("<I", crc) + comp
            chunks.append(chunk)
            sizes.append(len(chunk))
            crcs.append(crc)

        pixel_data = b"".join(chunks)

        meta = {
            "v": VERSION,
            "codec": "zstd",
            "n": len(slices),
            "sz": sizes,
            "crc": crcs,
            "dtype": dtype,
            "shape": shape,
            "original_bytes": orig_size,
        }
        if metadata:
            meta["s"] = metadata

        mc = _zstd_compress(json.dumps(meta, separators=(",", ":")).encode(), 19)

        with open(path, "wb") as f:
            f.write(struct.pack(HEADER_FMT, MAGIC, VERSION, 0, 0))
            f.write(struct.pack("<Q", len(mc)))
            f.write(mc)
            f.write(pixel_data)

        final_size = path.stat().st_size
        return {
            "slices": len(slices),
            "original_bytes": orig_size,
            "compressed_bytes": final_size,
            "ratio": round(orig_size / final_size, 3),
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
        if ver not in (2, 3):
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
        self._meta = json.loads(_zstd_decompress(meta_bytes))

        self.num_slices = self._meta["n"]
        self._sizes = self._meta["sz"]
        self._crcs = self._meta.get("crc")
        self.dtype = np.dtype(self._meta.get("dtype", "int16"))
        self.shape = tuple(self._meta.get("shape", [512, 512]))
        self._pixel_start = self._file.tell()

        # Precompute cumulative offsets for O(1) seek
        self._offsets = []
        running = 0
        for s in self._sizes:
            self._offsets.append(running)
            running += s

    def open(self):
        """Parse metadata. Keeps file handle open for fast sequential access.
        
        For worker-safe usage (num_workers>0), use get_slice() without
        calling open() — it will open/close per call.
        """
        if self._file is not None:
            return
        self._file = open(self.path, "rb")
        self._parse_header()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def get_slice(self, index: int) -> np.ndarray:
        """Decode and return one slice.
        
        Worker-safe: if no handle is open, opens the file, reads, and closes.
        If a handle is open (via open() or context manager), reuses it.
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
            raise IndexError(f"Slice {index}, max {self.num_slices - 1}")

        try:
            offset = self._pixel_start + self._offsets[index]
            self._file.seek(offset)
            data = self._file.read(self._sizes[index])

            # CRC check (if stored)
            if self._crcs:
                stored_crc, comp = struct.unpack("<I", data[:4]), data[4:]
                if stored_crc[0] != zlib.crc32(comp):
                    raise IntegrityError(
                        f"CRC mismatch on slice {index}: "
                        f"stored=0x{stored_crc[0]:08x}, "
                        f"computed=0x{zlib.crc32(comp):08x}"
                    )
            else:
                comp = data

            raw = _zstd_decompress(comp)
            return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)
        finally:
            if own_handle:
                self._file.close()
                self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
