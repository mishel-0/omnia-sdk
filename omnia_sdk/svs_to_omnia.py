#!/usr/bin/env python3
"""
Convert .svs whole-slide images to .omnia v4 containers.

Extracts all tiles across all pyramid levels, compresses each
losslessly with Zstd (batched), and streams them to the .omnia container.

Usage:
    omnia svs-convert input.svs [output.omnia]
"""
import sys, time
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from .container import OmniaContainer, DEFAULT_BATCH_SIZE, DEFAULT_ZSTD_LEVEL


def svs_to_omnia(svs_path: Path, omnia_path: Path,
                 verbose: bool = True,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 zstd_level: int = DEFAULT_ZSTD_LEVEL,
                 codec: str = "zstd",
                 quality: int = 85,
                 min_level: int = 0) -> dict:
    """Convert .svs to .omnia v4. Returns stats dict.

    codec:
      "zstd" — lossless. Decodes tiles to RGB and re-compresses with Zstd
               (training-speed format; output is typically LARGER than a
               JP2K .svs — the speed claim is about throughput, not size).
      "jpeg" — lossy. Re-encodes tiles as JPEG (quality 0-100).

    min_level:
      First pyramid level to keep (0 = 40x, 1 = 20x, 2 = 10x, 3 = 5x).
      TCGA .svs files are ~127x compressed already, so the ONLY way to
      actually shrink them is to drop high-resolution levels. Gleason
      grading at 20x doesn't need level 0 (40x, 87% of the file):
      `min_level=1` makes the container ~8x SMALLER than the .svs,
      losslessly.

    Streams tiles to disk in batches — no OOM on large slides.
    """
    from openslide import OpenSlide
    from PIL import Image
    import io

    if codec not in ("zstd", "jpeg"):
        raise ValueError(f"Unknown codec: {codec!r} (expected 'zstd' or 'jpeg')")

    slide = OpenSlide(str(svs_path))
    level_count = slide.level_count
    if min_level < 0 or min_level >= level_count:
        raise ValueError(f"min_level must be 0..{level_count - 1} (slide has {level_count} levels)")

    if verbose:
        print(f"  Slide: {svs_path.name}")
        print(f"  Levels: {level_count} (keeping {min_level}..{level_count - 1})")

    # Build level metadata from openslide properties
    levels_meta = []
    global_idx = 0
    for level in range(min_level, level_count):
        w, h = slide.level_dimensions[level]
        tw = int(slide.properties.get(f"openslide.level[{level}].tile-width", 256))
        th = int(slide.properties.get(f"openslide.level[{level}].tile-height", 256))
        tx = (w + tw - 1) // tw
        ty = (h + th - 1) // th
        tc = tx * ty

        levels_meta.append({
            "w": w, "h": h, "tw": tw, "th": th,
            "tx": tx, "ty": ty, "tc": tc, "start": global_idx,
        })
        global_idx += tc

    slide_dims = slide.dimensions

    # Generator: yields tiles one at a time from openslide
    def tile_generator() -> Iterator[np.ndarray]:
        for level in range(min_level, level_count):
            w, h = slide.level_dimensions[level]
            tw = int(slide.properties.get(f"openslide.level[{level}].tile-width", 256))
            th = int(slide.properties.get(f"openslide.level[{level}].tile-height", 256))
            tx = (w + tw - 1) // tw
            ty = (h + th - 1) // th
            downsample = slide.level_downsamples[level]

            if verbose:
                print(f"  L{level}: {w}x{h}  {tx}x{ty} tiles ({tw}x{th})")

            lt0 = time.perf_counter()
            count = 0
            for tile_y in range(ty):
                for tile_x in range(tx):
                    loc_x_l0 = int(tile_x * tw * downsample)
                    loc_y_l0 = int(tile_y * th * downsample)
                    tw_a = min(tw, w - tile_x * tw)
                    th_a = min(th, h - tile_y * th)
                    pil = slide.read_region((loc_x_l0, loc_y_l0), level, (tw_a, th_a))
                    yield np.array(pil.convert("RGB"), dtype=np.uint8)
                    count += 1

            if verbose:
                print(f"    {count} tiles in {time.perf_counter()-lt0:.1f}s")

    meta = {
        "source": svs_path.name,
        "slide_dims": list(slide_dims),
        "levels": levels_meta,
    }

    if verbose:
        print(f"  Writing .omnia (codec={codec}, batch_size={batch_size}, "
              f"zstd_level={zstd_level}, quality={quality})...", end=" ", flush=True)

    wt0 = time.perf_counter()

    if codec == "jpeg":
        # Lossy path: re-encode every tile as JPEG (padded to 256x256 so
        # edge tiles stay uniform for the dataset preload).
        def jpeg_bytes():
            for arr in tile_generator():
                img = Image.fromarray(arr)
                if img.size != (256, 256):
                    canvas = Image.new("RGB", (256, 256), (0, 0, 0))
                    canvas.paste(img, (0, 0))
                    img = canvas
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                yield buf.getvalue()

        stats = OmniaContainer.write_native(omnia_path, jpeg_bytes(),
                                            metadata=meta,
                                            pixel_codec="jpeg",
                                            native_codec="jpeg")
    else:
        stats = OmniaContainer.write(omnia_path, tile_generator(),
                                      metadata=meta,
                                      batch_size=batch_size,
                                      zstd_level=zstd_level)

    stats["write_time_s"] = round(time.perf_counter() - wt0, 2)
    stats["source"] = svs_path.name
    stats["total_tiles"] = stats["tiles"]
    stats["svs_size_mb"] = round(svs_path.stat().st_size / 1e6, 1)
    stats["svs_bytes"] = svs_path.stat().st_size
    stats["ratio_vs_svs"] = round(svs_path.stat().st_size / stats["compressed_bytes"], 3)

    slide.close()

    if verbose:
        mb_comp = stats["compressed_bytes"] / 1e6
        print(f"done ({stats['write_time_s']}s)")
        print(f"  .svs: {stats['svs_size_mb']:.1f} MB → "
              f".omnia: {mb_comp:.1f} MB")
        if codec == "jpeg":
            print(f"  Compression: {stats['ratio_vs_svs']}x vs .svs "
                  f"(JPEG q{quality}, lossy)")
        else:
            mb_raw = stats["original_bytes"] / 1e6
            print(f"  Compression: {stats['ratio_vs_svs']}x vs .svs | "
                  f"{stats['ratio']}x vs raw pixels ({mb_raw:.0f} MB)")
        if stats["ratio_vs_svs"] < 1:
            print(f"  WARNING: output is {1/stats['ratio_vs_svs']:.1f}x LARGER "
                  f"than the source .svs (source is already compressed). "
                  f"Try '--codec jpeg' for 2x smaller, or 'svs-native' for "
                  f"size-neutral conversion.")

    return stats
