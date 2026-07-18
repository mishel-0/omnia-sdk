#!/usr/bin/env python3
"""
Omnia CLI — omnia compress/decompress/info/verify
"""
import sys
import argparse
import json
from pathlib import Path

from . import OmniaCompressor, __version__, verify as _verify


def main():
    parser = argparse.ArgumentParser(
        prog="omnia",
        description="Omnia DICOM Compression — 277 files → 1, 3x lossless",
    )
    parser.add_argument("--version", action="version", version=f"omnia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    c = sub.add_parser("compress", help="Compress DICOM slices to .omnia")
    c.add_argument("input", help="Directory containing .dcm files")
    c.add_argument("output", help="Output .omnia file path")
    c.add_argument("-v", "--verbose", action="store_true")

    # decompress
    d = sub.add_parser("decompress", aliases=["decomp", "extract"],
                        help="Decompress .omnia to DICOM slices")
    d.add_argument("input", help=".omnia file path")
    d.add_argument("output", help="Output directory for restored DICOMs")
    d.add_argument("-v", "--verbose", action="store_true")

    # info
    i = sub.add_parser("info", help="Show .omnia file metadata")
    i.add_argument("input", help=".omnia file path")
    i.add_argument("-j", "--json", action="store_true", help="Output as JSON")

    # verify
    v = sub.add_parser("verify", help="Verify .omnia file integrity")
    v.add_argument("input", help=".omnia file path")

    args = parser.parse_args()

    comp = OmniaCompressor()

    if args.command == "compress":
        result = comp.compress(args.input, args.output, verbose=args.verbose)
        print(f"✅ {result['original_size']/1024**2:.1f} MB → "
              f"{result['compressed_size']/1024**2:.1f} MB "
              f"({result['ratio']:.2f}x)")
        if result["ratio"] < 1.5:
            print("⚠️  Low compression ratio. Input may not be DICOM.")

    elif args.command in ("decompress", "decomp", "extract"):
        result = comp.decompress(args.input, args.output, verbose=args.verbose)
        print(f"✅ {result['num_slices']} slices restored to {result['output_dir']}")

    elif args.command == "info":
        slices = comp.info(args.input)
        if args.json:
            print(json.dumps([{
                "rows": s.rows, "cols": s.cols,
                "patient_id": s.patient_id,
            } for s in slices], indent=2))
        else:
            print(f"  .omnia file: {args.input}")
            print(f"  Slices:      {len(slices)}")
            print(f"  Resolution:  {slices[0].rows}x{slices[0].cols}" if slices else "  (empty)")
            if slices:
                print(f"  Patient:     {slices[0].patient_id[:40]}")
                print(f"  Study UID:   {slices[0].study_uid[:40]}")

    elif args.command == "verify":
        ok = _verify(args.input)
        print(f"{'✅ Valid' if ok else '❌ Corrupt'} .omnia file")


if __name__ == "__main__":
    main()
