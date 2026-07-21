#!/usr/bin/env python3
"""
Training benchmark: DICOM vs .omnia Zstd.
TinyCNN for I/O-bound comparison. Optimized DataLoader.
"""
import sys, time, json, threading, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import DicomDataset, OmniaDataset

BATCH = 32
EPOCHS = 3
NUM_WORKERS = 4


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def fresh_model():
    m = TinyCNN().cuda()
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
        crit(model(images), labels).backward()
        opt.step()
    elapsed = time.perf_counter() - t0
    mon_active = False
    t.join(timeout=2)
    gu = np.mean(gpu_vals) if gpu_vals else 0
    return round(elapsed, 2), round(gu, 1)


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/lidc_raw"
    omnia_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/omnia_training_data"

    torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print("OMNIA TRAINING BENCHMARK — TinyCNN")
    print(f"  Batch={BATCH}, Workers={NUM_WORKERS}")
    print("=" * 65)

    dcm_bytes = sum(f.stat().st_size for sd in Path(raw_dir).iterdir()
                    if sd.is_dir() for f in sd.rglob("*.dcm") if f.suffix.lower() == ".dcm")
    omnia_bytes = sum(f.stat().st_size for f in Path(omnia_dir).glob("*.omnia"))

    t0 = time.perf_counter()
    ds_dcm = DicomDataset(raw_dir)
    dl_dcm = DataLoader(ds_dcm, batch_size=BATCH, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    print(f"\n  DICOM: {len(ds_dcm)} slices ({time.perf_counter()-t0:.1f}s)")
    print(f"  .omnia: loading...")
    t0 = time.perf_counter()
    ds_om = OmniaDataset(omnia_dir)
    dl_om = DataLoader(ds_om, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
    print(f"  .omnia: {len(ds_om)} slices ({time.perf_counter()-t0:.1f}s)")
    print(f"  Storage: DICOM={dcm_bytes/1e6:.0f} MB  .omnia={omnia_bytes/1e6:.0f} MB  "
          f"({dcm_bytes/omnia_bytes:.2f}x)")

    print(f"\n--- DICOM ---")
    dcm = []
    for ep in range(EPOCHS):
        m, o, c = fresh_model()
        t, gu = train_epoch(m, dl_dcm, c, o)
        dcm.append({"epoch": ep + 1, "time": t, "gpu": gu})
        tag = "cold" if ep == 0 else "hot"
        print(f"  Epoch {ep+1}/{EPOCHS} ({tag}): {t}s  GPU:{gu}%")

    print(f"\n--- .omnia Zstd ---")
    om = []
    for ep in range(EPOCHS):
        m, o, c = fresh_model()
        t, gu = train_epoch(m, dl_om, c, o)
        om.append({"epoch": ep + 1, "time": t, "gpu": gu})
        tag = "cold" if ep == 0 else "hot"
        print(f"  Epoch {ep+1}/{EPOCHS} ({tag}): {t}s  GPU:{gu}%")

    dt = [r["time"] for r in dcm]
    ot = [r["time"] for r in om]
    dg = [r["gpu"] for r in dcm]
    og = [r["gpu"] for r in om]

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"  DICOM avg:     {np.mean(dt):.2f}s  GPU:{np.mean(dg):.1f}%")
    print(f"  .omnia avg:    {np.mean(ot):.2f}s  GPU:{np.mean(og):.1f}%")
    print(f"  Speedup:       {np.mean(dt)/np.mean(ot):.2f}x")
    print(f"  Storage ratio: {dcm_bytes/omnia_bytes:.2f}x")

    ds_om.close_all()


if __name__ == "__main__":
    main()
