"""The command line: conversion works end to end and the codec rules pick what the data says."""
import sys

import numpy as np
import pytest

from omnia_sdk import cli, OmniaContainer
from omnia_sdk.codec_bench import recommend

nib = pytest.importorskip("nibabel")


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["omnia", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def test_convert_then_verify_nifti(tmp_path, monkeypatch):
    src = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(np.arange(8 * 8 * 4, dtype=np.int16).reshape(8, 8, 4), np.eye(4)), str(src))
    out = tmp_path / "v.omnia"
    assert _run(monkeypatch, "convert", str(src), str(out)) == 0
    assert _run(monkeypatch, "verify", str(out)) == 0
    with OmniaContainer(out) as c:
        assert c.num_slices == 4


def test_convert_rejects_unknown_input(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("nope")
    assert _run(monkeypatch, "convert", str(f)) == 1


ROWS = [
    {"variant": "raw", "kind": "lossless", "size_mb": 100, "ratio_vs_raw": 1.0, "decode_1t_tiles_s": 40000},
    {"variant": "zstd_L1_b1", "kind": "lossless", "size_mb": 92, "ratio_vs_raw": 1.09, "decode_1t_tiles_s": 2900},
    {"variant": "zstd_L3_b16", "kind": "lossless", "size_mb": 91.9, "ratio_vs_raw": 1.09, "decode_1t_tiles_s": 240},
    {"variant": "jpeg_q75", "kind": "lossy", "size_mb": 8, "ratio_vs_raw": 12, "decode_1t_tiles_s": 1800, "feature_cosine": 0.979},
    {"variant": "jpeg_q90", "kind": "lossy", "size_mb": 13, "ratio_vs_raw": 7.8, "decode_1t_tiles_s": 1700, "feature_cosine": 0.989},
    {"variant": "jpeg_q95", "kind": "lossy", "size_mb": 17, "ratio_vs_raw": 5.7, "decode_1t_tiles_s": 1500, "feature_cosine": 0.994},
]


def test_recommendation_avoids_slow_batched_blocks_and_picks_fidelity_among_compact():
    rec = recommend(ROWS)
    assert rec["lossless"] == "zstd_L1_b1"      # the b16 variant is 12x slower to read shuffled
    assert rec["compact"] == "jpeg_q95"         # highest fidelity that is still >= 5x smaller


def test_convert_a_folder_of_slides(tmp_path, monkeypatch):
    pytest.importorskip("openslide")
    tifffile = pytest.importorskip("tifffile")
    src = tmp_path / "slides"
    src.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        tifffile.imwrite(str(src / f"{name}.tif"), rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
                         tile=(256, 256), photometric="rgb")
    out = tmp_path / "out"
    assert _run(monkeypatch, "convert", str(src), str(out), "--min-level", "0") == 0
    assert sorted(p.name for p in out.glob("*.omnia")) == ["a.omnia", "b.omnia"]
