"""`omnia doctor`: show which formats this machine can convert, and fix what is missing.

    omnia doctor          report
    omnia doctor --fix    pip-install the missing readers into this same environment
"""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys

# (feature, import name, pip requirement, what it enables)
CHECKS = [
    ("Whole-slide images", "openslide", ["openslide-python", "openslide-bin", "Pillow"], "SVS, NDPI, MRXS, SCN, BIF, TIFF"),
    ("DICOM series", "pydicom", ["pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], "CT, MR and other DICOM"),
    ("DICOM whole-slide", "wsidicom", ["wsidicom"], "VL whole-slide DICOM"),
    ("NIfTI", "nibabel", ["nibabel"], ".nii and .nii.gz"),
    ("PyTorch dataset", "torch", ["torch"], "OmniaDataset for training (optional)"),
]


def _probe(module: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        if module == "openslide":  # the Python package can import while the C library is missing
            mod.OpenSlide  # noqa: B018
            version = getattr(mod, "__library_version__", getattr(mod, "__version__", "?"))
        else:
            version = getattr(mod, "__version__", "?")
        return True, str(version)
    except Exception as e:  # ImportError, or OSError from a missing native library
        return False, f"{type(e).__name__}: {e}".splitlines()[0][:90]


def report() -> list[dict]:
    rows = []
    for feature, module, reqs, enables in CHECKS:
        ok, detail = _probe(module)
        rows.append({"feature": feature, "module": module, "ok": ok, "detail": detail,
                     "requirements": reqs, "enables": enables})
    return rows


def main(fix: bool = False) -> int:
    from . import __version__
    print(f"omnia-sdk {__version__}  ·  Python {platform.python_version()}  ·  {platform.system()} {platform.machine()}")
    print(f"environment: {sys.prefix}\n")
    rows = report()
    for r in rows:
        mark = "OK " if r["ok"] else "-- "
        print(f"  {mark} {r['feature']:<20} {r['enables']:<34} {r['detail'] if r['ok'] else 'not installed'}")
    missing = [r for r in rows if not r["ok"] and r["module"] != "torch"]
    if not missing:
        print("\nEvery format reader is installed. Try:  omnia convert <file or folder>")
        return 0
    reqs = sorted({q for r in missing for q in r["requirements"]})
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *reqs]
    if not fix:
        print("\nTo add the missing readers:  omnia doctor --fix")
        print("  (runs: " + " ".join(["python", "-m", "pip", "install", *reqs]) + ")")
        return 1
    print("\nInstalling: " + " ".join(reqs))
    code = subprocess.call(cmd)
    if code != 0:
        print("\npip failed. The output above says why; on a managed Python try the one-line installer from the README.")
        return code
    importlib.invalidate_caches()
    still = [r["feature"] for r in report() if not r["ok"] and r["module"] != "torch"]
    if still:
        print("\nStill missing after install: " + ", ".join(still) + ". Restart the terminal and run `omnia doctor` again.")
        return 1
    print("\nDone. Every format reader is installed.")
    return 0
