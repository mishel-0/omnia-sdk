#!/usr/bin/env python3
"""
Reproducibility benchmark: DICOM vs .omnia
Requires: omnia-sdk, pydicom, torch, torchvision, numpy

Usage:
    python benchmark.py /path/to/lidc_raw/ /path/to/compressed/

Output:
    - Prints per-epoch timing and GPU utilization
    - Saves full results to benchmark_results.json

Expected results (RTX A4000, 3,387 slices, ResNet-18, 100 epochs):
    DICOM steady epoch:  ~18 s/epoch (varies with hardware)
    .omnia steady epoch: ~18 s/epoch (varies with hardware)
    Cold epoch 1 DICOM:  ~40 s (filesystem cache empty)
    Cold epoch 1 .omnia: ~22 s
"""
import sys, time, json, threading, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from omnia_sdk.dataset import DicomDataset, OmniaDataset

# Reproducible settings matching the reference benchmark
BATCH = 64
EPOCHS = 100
NUM_WORKERS = 4


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
        time.sleep(1.0)


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
        crit(model(images), labels).backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    mon_active = False
    t.join(timeout=3)
    gu = np.mean(gpu_vals) if gpu_vals else 0
    return round(elapsed, 2), round(gu, 1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py /path/to/lidc_raw/ /path/to/compressed/")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    omnia_dir = Path(sys.argv[2])

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print(f"DICOM data: {raw_dir}")
    print(f".omnia data: {omnia_dir}")
    print(f"Model: ResNet-18, Batch: {BATCH}, Workers: {NUM_WORKERS}, Epochs: {EPOCHS}")

    # Load
    ds_dcm = DicomDataset(raw_dir)
    dl_dcm = DataLoader(ds_dcm, batch_size=BATCH, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    ds_om = OmniaDataset(omnia_dir)
    dl_om = DataLoader(ds_om, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    print(f"Slices: {len(ds_dcm)}")

    # Benchmark DICOM
    print("\n--- DICOM ---")
    dcm_results = []
    for ep in range(EPOCHS):
        m, o, c = make_model()
        t, gu = train_epoch(m, dl_dcm, c, o)
        dcm_results.append({"epoch": ep + 1, "time": t, "gpu": gu})
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{EPOCHS}: {t}s  GPU:{gu}%")

    # Benchmark .omnia
    print("\n--- .omnia ---")
    omnia_results = []
    for ep in range(EPOCHS):
        m, o, c = make_model()
        t, gu = train_epoch(m, dl_om, c, o)
        omnia_results.append({"epoch": ep + 1, "time": t, "gpu": gu})
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{EPOCHS}: {t}s  GPU:{gu}%")

    # Save
    output = {
        "config": {
            "batch": BATCH, "epochs": EPOCHS, "workers": NUM_WORKERS,
            "slices": len(ds_dcm),
        },
        "dicom": {
            "epoch_times": [r["time"] for r in dcm_results],
            "gpu_util": [r["gpu"] for r in dcm_results],
            "total_time_s": sum(r["time"] for r in dcm_results),
        },
        "omnia": {
            "epoch_times": [r["time"] for r in omnia_results],
            "gpu_util": [r["gpu"] for r in omnia_results],
            "total_time_s": sum(r["time"] for r in omnia_results),
        },
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")

    ds_om.close_all()


if __name__ == "__main__":
    main()
