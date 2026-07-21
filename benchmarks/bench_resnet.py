#!/usr/bin/env python3
"""
Training benchmark: DICOM vs .omnia Zstd on real raw LIDC data.
ResNet-18, optimized DataLoader, GPU benchmark mode.
"""
import sys, os, time, json, threading, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from container import OmniaContainer
from dataset import DicomDataset, OmniaDataset

# ── Tuning ──
BATCH = 64
EPOCHS = 5
NUM_WORKERS = 4
PREFETCH = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
RESULTS = Path(__file__).resolve().parent.parent / "results" / "clean_benchmark.json"


def make_model():
    m = models.resnet18(weights=None, num_classes=2)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m = m.cuda()
    return m, torch.optim.Adam(m.parameters(), lr=0.001), nn.CrossEntropyLoss()


gpu_vals = []
mon_active = False


def gpu_monitor():
    global gpu_vals
    while mon_active:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            gpu_vals.append(float(r.stdout.strip()))
        except Exception:
            pass
        time.sleep(0.3)


def train_epoch(model, loader, crit, opt):
    global gpu_vals, mon_active
    gpu_vals = []
    mon_active = True
    t = threading.Thread(target=gpu_monitor, daemon=True)
    t.start()

    model.train()
    t0 = time.perf_counter()
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        opt.zero_grad()
        loss = crit(model(images), labels)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - t0

    mon_active = False
    t.join(timeout=2)
    gu = np.mean(gpu_vals) if gpu_vals else 0
    return round(elapsed, 2), round(gu, 1)


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/lidc_raw"
    omnia_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/omnia_training_data"

    # Enable cudnn benchmark for optimal conv selection
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("=" * 65)
    print("OMNIA TRAINING BENCHMARK — OPTIMIZED")
    print(f"  Model=ResNet-18  Batch={BATCH}  Workers={NUM_WORKERS}")
    print(f"  prefetch={PREFETCH}  pin_memory={PIN_MEMORY}")
    print("=" * 65)

    # Storage
    dcm_bytes = sum(f.stat().st_size for sd in Path(raw_dir).iterdir()
                    if sd.is_dir() for f in sd.rglob("*.dcm") if f.suffix.lower() == ".dcm")
    omnia_bytes = sum(f.stat().st_size for f in Path(omnia_dir).glob("*.omnia"))
    print(f"\n  Storage: DICOM={dcm_bytes/1e6:.0f} MB  .omnia={omnia_bytes/1e6:.0f} MB  "
          f"({dcm_bytes/omnia_bytes:.2f}x)")

    # Load datasets
    print("\n1. Loading datasets...")
    t0 = time.perf_counter()
    ds_dcm = DicomDataset(raw_dir)
    dl_dcm = DataLoader(ds_dcm, batch_size=BATCH, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                         persistent_workers=PERSISTENT_WORKERS,
                         prefetch_factor=PREFETCH)
    print(f"   DICOM: {len(ds_dcm)} slices ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    ds_om = OmniaDataset(omnia_dir)
    dl_om = DataLoader(ds_om, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                        persistent_workers=PERSISTENT_WORKERS,
                        prefetch_factor=PREFETCH)
    print(f"   .omnia: {len(ds_om)} slices ({time.perf_counter()-t0:.1f}s)")

    # ── Benchmark DICOM ──
    print(f"\n--- DICOM (ResNet-18, {EPOCHS} epochs, {NUM_WORKERS} workers) ---")
    dcm = []
    for ep in range(EPOCHS):
        m, o, c = make_model()
        t, gu = train_epoch(m, dl_dcm, c, o)
        dcm.append({"epoch": ep + 1, "time": t, "gpu": gu})
        tag = "cold" if ep == 0 else "hot"
        print(f"  Epoch {ep+1}/{EPOCHS} ({tag}): {t}s  GPU:{gu}%")

    # ── Benchmark .omnia ──
    print(f"\n--- .omnia Zstd (ResNet-18, {EPOCHS} epochs, {NUM_WORKERS} workers) ---")
    om = []
    for ep in range(EPOCHS):
        m, o, c = make_model()
        t, gu = train_epoch(m, dl_om, c, o)
        om.append({"epoch": ep + 1, "time": t, "gpu": gu})
        tag = "cold" if ep == 0 else "hot"
        print(f"  Epoch {ep+1}/{EPOCHS} ({tag}): {t}s  GPU:{gu}%")

    # ── Results ──
    dt = [r["time"] for r in dcm]
    ot = [r["time"] for r in om]
    dg = [r["gpu"] for r in dcm]
    og = [r["gpu"] for r in om]

    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    print(f"{'='*65}")
    print(f"\n{'Method':<35s} {'E1':>8s} {'E2':>8s} {'E3':>8s} {'E4':>8s} {'E5':>8s} {'Avg':>8s}")
    print(f"  {'─'*75}")
    print(f"  {'DICOM':<35s}", *[f"{t:>6.2f}s" for t in dt], f"{np.mean(dt):>6.2f}s")
    print(f"  {'.omnia Zstd':<35s}", *[f"{t:>6.2f}s" for t in ot], f"{np.mean(ot):>6.2f}s")

    d_avg = np.mean(dt)
    o_avg = np.mean(ot)
    warm_d = np.mean(dt[2:]) if len(dt) > 2 else d_avg
    warm_o = np.mean(ot[2:]) if len(ot) > 2 else o_avg

    print(f"\n  Avg speedup:       {d_avg/o_avg:.2f}x")
    print(f"  Warm speedup (3-5):{warm_d/warm_o:.2f}x")
    print(f"  GPU util DICOM:    {np.mean(dg):.1f}%")
    print(f"  GPU util .omnia:   {np.mean(og):.1f}%")
    print(f"  Storage:           {dcm_bytes/omnia_bytes:.2f}x")

    results = {
        "model": "ResNet-18",
        "batch": BATCH,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "cudnn_benchmark": True,
        "epochs": EPOCHS,
        "slices": len(ds_dcm),
        "storage": {
            "dicom_mb": round(dcm_bytes / 1e6, 1),
            "omnia_mb": round(omnia_bytes / 1e6, 1),
            "ratio": round(dcm_bytes / omnia_bytes, 2),
        },
        "dicom": {"epoch_times": dt, "gpu_utils": dg, "avg": round(np.mean(dt), 2)},
        "omnia": {"epoch_times": ot, "gpu_utils": og, "avg": round(np.mean(ot), 2)},
        "speedup_avg": round(d_avg / o_avg, 2),
        "speedup_warm": round(warm_d / warm_o, 2),
    }

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {RESULTS}")

    ds_om.close_all()


if __name__ == "__main__":
    main()
