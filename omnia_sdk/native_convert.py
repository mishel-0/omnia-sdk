#!/usr/bin/env python3
"""
Native .svs to .omnia converter — stores raw JP2K tile bytes directly.

No decompress/recompress cycle. Output size ≈ .svs file size.
For training, use `svs_to_omnia.py` (RGB Zstd format).
"""
import sys, time
from pathlib import Path
from typing import Optional


def svs_to_omnia_native(svs_path: Path, omnia_path: Path,
                        verbose: bool = True,
                        min_level: int = 0) -> dict:
    """Convert .svs to .omnia with native JP2K passthrough.

    Reads JP2K-compressed tiles directly from the .svs TIFF structure
    and stores them losslessly in the .omnia container.
    Output size = header + metadata + raw tile data (≈ .svs file size).

    min_level: first pyramid level to keep (0 = 40x, 1 = 20x, ...).
    The .svs is already ~127x compressed, so shrinking it further means
    dropping high-res levels: min_level=1 (20x and below) gives ~8x
    smaller than the .svs, losslessly — plenty for Gleason grading.

    Returns:
        {"tiles": N, "svs_size_mb": N, "omnia_size_mb": N, "levels": [...]}
    """
    import tifffile

    svs_size = svs_path.stat().st_size

    with tifffile.TiffFile(str(svs_path)) as tif:
        # Find pyramid pages (JP2K-compressed, subfiletype=0)
        pages = []
        for page in tif.pages:
            if page.compression == 33003:  # JP2K
                pages.append(page)

        if not pages:
            raise ValueError("No JP2K-compressed pages found in .svs")

        if min_level < 0 or min_level >= len(pages):
            raise ValueError(f"min_level must be 0..{len(pages) - 1} (slide has {len(pages)} pages)")
        pages = pages[min_level:]

        if verbose:
            print(f"  Slide: {svs_path.name}")
            print(f"  Pages: {len(pages)} (keeping {min_level}..{len(pages) - 1})")

        all_tile_data = []
        levels_meta = []
        global_idx = 0

        for pi, page in enumerate(pages):
            w = page.imagewidth
            h = page.imagelength
            tw = page.tilewidth
            th = page.tilelength
            tx = (w + tw - 1) // tw
            ty = (h + th - 1) // th
            tc = tx * ty

            tile_offsets = list(page.tags[324].value)
            tile_sizes = list(page.tags[325].value)

            if verbose:
                print(f"  L{pi}: {w}x{h}  {tx}x{ty} tiles ({tw}x{th})")

            fh = page.parent.filehandle
            lt0 = time.perf_counter()
            level_data = []
            for tix, (offset, size) in enumerate(zip(tile_offsets, tile_sizes)):
                fh.seek(offset)
                raw = fh.read(size)
                level_data.append(raw)

            all_tile_data.extend(level_data)
            if verbose:
                print(f"    {tc} tiles in {time.perf_counter()-lt0:.1f}s")

            levels_meta.append({
                "w": w, "h": h, "tw": tw, "th": th,
                "tx": tx, "ty": ty, "tc": tc, "start": global_idx,
            })
            global_idx += tc

    total_tiles = len(all_tile_data)
    raw_bytes = sum(len(t) for t in all_tile_data)

    if verbose:
        print(f"  Total tiles: {total_tiles}")
        print(f"  Raw tile data: {raw_bytes/1e6:.1f} MB")
        print(f"  Writing .omnia...", end=" ", flush=True)

    from .container import OmniaContainer

    stats = OmniaContainer.write_native(
        omnia_path, iter(all_tile_data),
        metadata={
            "levels": levels_meta,
            "source": svs_path.name,
            "slide_dims": [pages[0].imagewidth, pages[0].imagelength],
        },
        pixel_codec="native",
        native_codec="jp2k",
    )

    omnia_size = omnia_path.stat().st_size

    if verbose:
        print(f"done")
        print(f"  .svs: {svs_size/1e6:.1f} MB → .omnia: {omnia_size/1e6:.1f} MB")
        print(f"  Ratio vs svs: {svs_size/omnia_size:.2f}x")

    return {
        "tiles": total_tiles,
        "svs_bytes": svs_size,
        "omnia_bytes": omnia_size,
        "ratio_vs_svs": round(svs_size / omnia_size, 3),
        "levels": levels_meta,
        "source": svs_path.name,
    }
