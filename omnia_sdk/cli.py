#!/usr/bin/env python3
"""CLI for .omnia — compress, info, verify."""
import sys, argparse, json, struct, zstd
from pathlib import Path
from .container import OmniaContainer
from .convert import convert_study


def cmd_compress(args):
    result = convert_study(Path(args.input), Path(args.output))
    if result["slices"] > 0:
        print(f"✅ {result['study']}: {result['slices']} slices, "
              f"{result['original_bytes']/1e6:.1f} MB → {result['compressed_bytes']/1e6:.1f} MB, "
              f"{result['ratio']:.2f}x lossless")
    else:
        print(f"⏭️  {result.get('study', '?' )}: {result.get('error', 'no CT slices')}")


def cmd_info(args):
    c = OmniaContainer(args.input)
    c.open()
    print(f"File:     {args.input}")
    print(f"Slices:   {c.num_slices}")
    print(f"Shape:    {c.shape}")
    print(f"Dtype:    {c.dtype}")
    print(f"CRC:      {'yes' if c._crcs else 'no'}")
    c.close()


def cmd_verify(args):
    c = OmniaContainer(args.input)
    c.open()
    errors = 0
    for i in range(c.num_slices):
        try:
            c.get_slice(i)
        except Exception as e:
            print(f"❌ Slice {i}: {e}")
            errors += 1
    c.close()
    if errors == 0:
        print(f"✅ {args.input}: {c.num_slices} slices verified, 0 errors")
    else:
        print(f"❌ {errors}/{c.num_slices} slices failed")


def main():
    parser = argparse.ArgumentParser(prog="omnia", description=".omnia — Medical Image Container")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compress", help="Convert DICOM study to .omnia")
    p.add_argument("input", help="DICOM study directory")
    p.add_argument("output", help="Output .omnia file or directory")

    p = sub.add_parser("info", help="Show .omnia file info")
    p.add_argument("input", help=".omnia file")

    p = sub.add_parser("verify", help="Verify all slices in .omnia file")
    p.add_argument("input", help=".omnia file")

    args = parser.parse_args()
    if args.command == "compress":
        cmd_compress(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
