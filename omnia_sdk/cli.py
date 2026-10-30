#!/usr/bin/env python3
"""
CLI commands for .omnia SDK.
"""
import argparse, sys, logging
from pathlib import Path

from .container import OmniaContainer, IntegrityError, FormatError

logger = logging.getLogger(__name__)


def svs_convert(svs_path: str, omnia_path: str | None = None,
               batch_size: int = 16, zstd_level: int = 5,
               codec: str = "zstd", quality: int = 85,
               min_level: int = 0) -> bool:
    """CLI svs-convert command."""
    from .svs_to_omnia import svs_to_omnia
    sp = Path(svs_path)
    if not sp.exists():
        logger.error("File not found: %s", sp)
        return False
    op = Path(omnia_path) if omnia_path else sp.with_suffix(".omnia")
    stats = svs_to_omnia(sp, op, batch_size=batch_size, zstd_level=zstd_level,
                         codec=codec, quality=quality, min_level=min_level)
    if codec == "jpeg":
        logger.info("Converted %s: %.1f MB → %.1f MB (%.2fx vs .svs, JPEG q%d lossy)",
                    sp.name, stats["svs_size_mb"],
                    stats["compressed_bytes"] / 1e6,
                    stats["ratio_vs_svs"], quality)
    else:
        logger.info("Converted %s: %.1f MB → %.1f MB (%.2fx vs .svs, %.2fx vs raw)",
                    sp.name, stats["svs_size_mb"],
                    stats["compressed_bytes"] / 1e6,
                    stats["ratio_vs_svs"], stats["ratio"])
    if stats["ratio_vs_svs"] < 1:
        logger.warning(
            "Output is %.1fx LARGER than the source .svs — the source is already "
            "compressed and Zstd-on-RGB cannot beat it. Use '--codec jpeg' for "
            "2x smaller (lossy) or 'svs-native' for size-neutral storage.",
            1 / stats["ratio_vs_svs"])
    return True


def verify(omnia_path: str) -> bool:
    """CLI verify command — checks all CRCs."""
    op = Path(omnia_path)
    if not op.exists():
        logger.error("File not found: %s", op)
        return False

    try:
        with OmniaContainer(op) as c:
            errors = 0
            for i in range(c.num_slices):
                try:
                    c.get_slice(i)
                except IntegrityError as e:
                    logger.error("Tile %d: %s", i, e)
                    errors += 1
            if errors == 0:
                logger.info("Verification passed: %d tiles, 0 errors", c.num_slices)
                return True
            logger.error("Verification failed: %d errors", errors)
            return False
    except (FormatError, IntegrityError) as e:
        logger.error("Verification failed: %s", e)
        return False


def info(omnia_path: str) -> bool:
    """Display .omnia file metadata."""
    op = Path(omnia_path)
    if not op.exists():
        logger.error("File not found: %s", op)
        return False
    try:
        with OmniaContainer(op) as c:
            print(f"  File:         {c.path.name}")
            print(f"  Size:         {c.path.stat().st_size / 1e6:.1f} MB")
            print(f"  Version:      {c._meta.get('v', '?')}")
            print(f"  Codec:        {c._meta.get('codec', '?')}")
            print(f"  Tiles:        {c.num_slices}")
            print(f"  Tile shape:   {c.shape}")
            print(f"  Dtype:        {c.dtype}")
            print(f"  Levels:       {c.num_levels}")
            # `start` is the level's first tile index in the flat array, not
            # its pyramid level. Printing it labelled level 1 as "L4209",
            # which reads as a corrupt file rather than the second level.
            for i, lvl in enumerate(c.levels):
                print(f"    L{i}: {lvl['w']}x{lvl['h']}  "
                      f"{lvl['tx']}x{lvl['ty']} tiles ({lvl['tw']}x{lvl['th']})"
                      f"  [{lvl['tc']} tiles @ index {lvl['start']}]")
            raw = c._meta.get("original_bytes", 0)
            comp = c.path.stat().st_size
            if raw:
                print(f"  Ratio:        {raw/comp:.2f}x (raw: {raw/1e6:.1f} MB)")
            if "batch_size" in (c._meta or {}):
                print(f"  Batch size:   {c._meta['batch_size']} tiles/chunk")
                print(f"  Zstd level:   {c._meta.get('zstd_level', '?')}")
                print(f"  Batches:      {len(c._meta.get('sz', []))}")
            if c.pixel_codec in ("native", "jpeg"):
                print(f"  Pixel codec:  {c.pixel_codec} ({c.native_codec or '?'})")
            if "source" in (c._meta or {}):
                print(f"  Source:       {c._meta['source']}")
        return True
    except Exception as e:
        logger.error("Failed to read .omnia file: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description=".omnia SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    svs_p = subparsers.add_parser("svs-convert", help="Convert .svs to .omnia (Zstd RGB or JPEG)")
    svs_p.add_argument("svs_file", help="Input .svs file")
    svs_p.add_argument("omnia_file", nargs="?", help="Output .omnia file")
    svs_p.add_argument("--batch-size", type=int, default=16,
                       help="Tiles per compressed batch (default: 16, higher = better ratio)")
    svs_p.add_argument("--zstd-level", type=int, default=5,
                       help="Zstd compression level (default: 5, higher = smaller but slower)")
    svs_p.add_argument("--codec", choices=["zstd", "jpeg"], default="zstd",
                       help="Tile codec (default: zstd lossless; jpeg = lossy)")
    svs_p.add_argument("--quality", type=int, default=85, metavar="0-100",
                       help="JPEG quality for --codec jpeg (default: 85, lower = smaller)")
    svs_p.add_argument("--min-level", type=int, default=0, metavar="N",
                       help="First pyramid level to keep (0=40x, 1=20x, 2=10x, 3=5x). "
                            "min-level 1 drops the 40x level and makes the file "
                            "~8x smaller than .svs — the only real way to "
                            "compress already-JP2K slides")

    svs_native = subparsers.add_parser("svs-native",
        help="Convert .svs to .omnia (native JP2K passthrough — same size as .svs)")
    svs_native.add_argument("svs_file", help="Input .svs file")
    svs_native.add_argument("omnia_file", nargs="?", help="Output .omnia file")
    svs_native.add_argument("--min-level", type=int, default=0, metavar="N",
                            help="First pyramid level to keep (0=40x, 1=20x, ...). "
                                 "min-level 1 = ~8x smaller than .svs, lossless")

    conv_p = subparsers.add_parser("convert",
        help="Convert any supported medical image (slide, DICOM, NIfTI) to .omnia")
    conv_p.add_argument("input", help="File or folder: .svs/.ndpi/.mrxs/.tif, DICOM folder, .nii/.nii.gz")
    conv_p.add_argument("omnia_file", nargs="?", help="Output .omnia file")
    conv_p.add_argument("--min-level", type=int, default=None, metavar="N",
                        help="Whole-slide only: first pyramid level to keep (default 1)")
    conv_p.add_argument("--batch-size", type=int, default=1,
                        help="Tiles per compressed block (default 1: measured fastest for shuffled training; "
                             "16 saved no space on H&E tiles and cut GPU use to 33-58%% in the A40 benchmark)")
    conv_p.add_argument("--zstd-level", type=int, default=5)

    conv_p.add_argument("--codec", choices=["zstd", "jpeg"], default=None,
                        help="Whole-slide only: zstd (lossless, default) or jpeg (lossy, ~5-12x smaller)")
    conv_p.add_argument("--quality", type=int, default=90, metavar="0-100",
                        help="JPEG quality for --codec jpeg (default 90)")

    bench_p = subparsers.add_parser("benchmark",
        help="Measure which codec/format trains fastest on your data")
    bench_sub = bench_p.add_subparsers(dest="bench_kind", required=True)
    bc = bench_sub.add_parser("codecs", help="zstd vs JPEG vs raw vs openslide on your slides",
                              add_help=False)
    bc.add_argument("rest", nargs=argparse.REMAINDER)
    bf = bench_sub.add_parser("formats", help=".omnia vs each format's usual reader (NIfTI, DICOM, slides)",
                              add_help=False)
    bf.add_argument("rest", nargs=argparse.REMAINDER)

    rec_p = subparsers.add_parser("recommend",
        help="Quick (no GPU) check of which codec suits a slide: shortcut for 'benchmark codecs --quick'")
    rec_p.add_argument("slides", nargs="+")
    rec_p.add_argument("--tiles", type=int, default=2000,
                       help="Tiles to sample (default 2000; fewer than ~1000 hides the cost of batched blocks)")

    verify_p = subparsers.add_parser("verify", help="Verify .omnia integrity")
    verify_p.add_argument("omnia_file", help=".omnia file")

    info_p = subparsers.add_parser("info", help="Show .omnia file metadata")
    info_p.add_argument("omnia_file", help=".omnia file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "svs-convert":
        sys.exit(0 if svs_convert(args.svs_file, args.omnia_file,
                                  batch_size=args.batch_size,
                                  zstd_level=args.zstd_level,
                                  codec=args.codec,
                                  quality=args.quality,
                                  min_level=args.min_level) else 1)
    elif args.command == "convert":
        from .converters import convert_file, detect_format, UnsupportedFormat
        try:
            fmt = detect_format(args.input)
            kw = {"batch_size": args.batch_size, "zstd_level": args.zstd_level}
            if fmt in ("whole-slide", "dicom-wsi"):
                kw["min_level"] = 1 if args.min_level is None else args.min_level
            if fmt == "whole-slide" and args.codec:
                kw["codec"] = args.codec
                kw["quality"] = args.quality
            stats = convert_file(args.input, args.omnia_file, fmt=fmt, **kw)
        except (UnsupportedFormat, FileNotFoundError, ImportError) as e:
            logging.error("%s", e)
            sys.exit(1)
        logging.info("Converted %s (%s): %.1f MB -> %.1f MB (%.2fx vs source)",
                     stats["source"], fmt, stats["source_bytes"] / 1e6,
                     stats["compressed_bytes"] / 1e6, stats["ratio_vs_source"] if "ratio_vs_source" in stats else stats.get("ratio_vs_svs", 0))
        sys.exit(0)
    elif args.command == "benchmark":
        if args.bench_kind == "codecs":
            from .codec_bench import main as run
        else:
            from .format_bench import main as run
        run(args.rest)
        sys.exit(0)
    elif args.command == "recommend":
        from .codec_bench import main as run
        run([*args.slides, "--quick", "--tiles", str(args.tiles)])
        sys.exit(0)
    elif args.command == "svs-native":
        from .native_convert import svs_to_omnia_native
        from pathlib import Path
        sp = Path(args.svs_file)
        if not sp.exists():
            logging.error("File not found: %s", sp)
            sys.exit(1)
        op = Path(args.omnia_file) if args.omnia_file else sp.with_suffix(".omnia")
        svs_to_omnia_native(sp, op, min_level=args.min_level)
        sys.exit(0)
    elif args.command == "verify":
        sys.exit(0 if verify(args.omnia_file) else 1)
    elif args.command == "info":
        sys.exit(0 if info(args.omnia_file) else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
