#!/usr/bin/env python3
"""
100-epoch benchmark: DICOM vs .omnia Zstd, ResNet-18 on real LIDC.
Reports aggregate: total time, steady-state avg, projected savings.
"""
import sys, time, json, threading, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omnia_sdk"))
from dataset import DicomDataset, OmniaDataset

BATCH = 64
EPOCHS = 100
NUM_WORKERS = 4
RESULTS = Path(__file__).resolve().parent.parent / "results" / "epoch100_benchmark.json"


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


def run_benchmark(name, loader, epochs):
    """Run epochs epochs, return list of {epoch, time, gpu}."""
    results = []
    log_interval = max(1, epochs // 10)
    for ep in range(epochs):
        m, o, c = make_model()
        t, gu = train_epoch(m, loader, c, o)
        results.append({"epoch": ep + 1, "time": t, "gpu": gu})
        if (ep + 1) % log_interval == 0 or ep == 0 or ep == epochs - 1:
            tag = "cold" if ep == 0 else "hot"
            print(f"  {name} Epoch {ep+1}/{epochs} ({tag}): {t}s  GPU:{gu}%")
    return results


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/lidc_raw"
    omnia_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/omnia_training_data"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("=" * 65)
    print("OMNIA — 100-EPOCH BENCHMARK")
    print(f"  Model=ResNet-18  Batch={BATCH}  Workers={NUM_WORKERS}")
    print("=" * 65)

    dcm_bytes = sum(f.stat().st_size for sd in Path(raw_dir).iterdir()
                    if sd.is_dir() for f in sd.rglob("*.dcm") if f.suffix.lower() == ".dcm")
    om_bytes = sum(f.stat().st_size for f in Path(omnia_dir).glob("*.omnia"))
    print(f"\n  Storage: DICOM={dcm_bytes/1e6:.0f} MB  .omnia={om_bytes/1e6:.0f} MB  "
          f"({dcm_bytes/om_bytes:.2f}x)")

    print("\n1. Loading datasets...")
    t0 = time.perf_counter()
    ds_dcm = DicomDataset(raw_dir)
    dl_dcm = DataLoader(ds_dcm, batch_size=BATCH, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)
    print(f"   DICOM: {len(ds_dcm)} slices ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    ds_om = OmniaDataset(omnia_dir)
    dl_om = DataLoader(ds_om, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)
    print(f"   .omnia: {len(ds_om)} slices ({time.perf_counter()-t0:.1f}s)")

    total_start = time.perf_counter()

    print(f"\n--- DICOM ({EPOCHS} epochs) ---")
    dcm = run_benchmark("DICOM", dl_dcm, EPOCHS)
    dcm_total = time.perf_counter() - total_start
    print(f"\n  DICOM done: {dcm_total/60:.1f} min")

    om_start = time.perf_counter()
    print(f"\n--- .omnia Zstd ({EPOCHS} epochs) ---")
    om = run_benchmark(".omnia", dl_om, EPOCHS)
    om_total = time.perf_counter() - om_start
    print(f"\n  .omnia done: {om_total/60:.1f} min")

    grand_total = time.perf_counter() - total_start + (time.perf_counter() - total_start - om_start + om_start)
    # just compute total wall time
    grand_total = time.perf_counter() - total_start

    # Analysis
    dt = np.array([r["time"] for r in dcm])
    ot = np.array([r["time"] for r in om])
    dg = np.array([r["gpu"] for r in dcm])
    og = np.array([r["gpu"] for r in om])

    # Steady state: last 50 epochs
    d_steady = np.mean(dt[-50:])
    o_steady = np.mean(ot[-50:])
    d_gpu_steady = np.mean(dg[-50:])
    o_gpu_steady = np.mean(og[-50:])

    # Cold + warmup vs steady
    d_total = np.sum(dt)
    o_total = np.sum(ot)

    print("\n" + "=" * 65)
    print("100-EPOCH RESULTS")
    print("=" * 65)
    print(f"\n{'':>30s} {'DICOM':>12s}  {'.omnia':>12s}  {'SAVING':>10s}")
    print(f"  {'─'*65}")
    print(f"  {'Total time (100 epochs):':<30s} {d_total:>7.1f}s  {o_total:>7.1f}s  {d_total-o_total:>7.1f}s")
    print(f"  {'Total time (min):':<30s} {d_total/60:>6.1f}m  {o_total/60:>6.1f}m  {(d_total-o_total)/60:>5.1f}m")
    print(f"  {'Steady epoch (last 50):':<30s} {d_steady:>6.2f}s  {o_steady:>6.2f}s  {d_steady/o_steady:>6.2f}x")
    print(f"  {'GPU util (steady):':<30s} {d_gpu_steady:>5.1f}%  {o_gpu_steady:>5.1f}%")
    print(f"  {'Storage ratio:':<30s} {dcm_bytes/om_bytes:>6.2f}x")
    print(f"  {'Cold epoch 1:':<30s} {dt[0]:>7.2f}s  {ot[0]:>7.2f}s  {dt[0]/ot[0]:>5.1f}x")
    print(f"  {'Epoch 2-5 (warmup):':<30s} {np.mean(dt[1:5]):>6.2f}s  {np.mean(ot[1:5]):>6.2f}s  {np.mean(dt[1:5])/np.mean(ot[1:5]):>5.1f}x")

    # Projection
    print(f"\n\n  PROJECTION TO 50,000 STUDIES (300 epochs):")
    scale = 50000 * 300 / (len(ds_dcm) * EPOCHS)
    d_proj = d_total * scale
    o_proj = o_total * scale
    print(f"  DICOM:  {d_proj/3600:.1f} hours ({d_proj/86400:.1f} days)")
    print(f"  .omnia: {o_proj/3600:.1f} hours ({o_proj/86400:.1f} days)")
    print(f"  SAVE:   {(d_proj-o_proj)/3600:.1f} hours ({(d_proj-o_proj)/86400:.1f} days)")

    results = {
        "config": {
            "model": "ResNet-18", "batch": BATCH, "epochs": EPOCHS,
            "num_workers": NUM_WORKERS, "slices": len(ds_dcm),
            "data": "Real uncompressed LIDC (15 patients)",
        },
        "storage": {
            "dicom_mb": round(dcm_bytes / 1e6, 1),
            "omnia_mb": round(om_bytes / 1e6, 1),
            "ratio": round(dcm_bytes / om_bytes, 2),
        },
        "dicom": {
            "total_time_s": round(float(d_total), 1),
            "steady_epoch_s": round(float(d_steady), 2),
            "gpu_steady_pct": round(float(d_gpu_steady), 1),
            "epoch_times": [round(float(t), 2) for t in dt],
            "gpu_util": [round(float(g), 1) for g in dg],
        },
        "omnia": {
            "total_time_s": round(float(o_total), 1),
            "steady_epoch_s": round(float(o_steady), 2),
            "gpu_steady_pct": round(float(o_gpu_steady), 1),
            "epoch_times": [round(float(t), 2) for t in ot],
            "gpu_util": [round(float(g), 1) for g in og],
        },
        "speedup": {
            "steady_state": round(float(d_steady / o_steady), 2),
            "total_100epoch": round(float(d_total / o_total), 2),
            "cold_epoch1": round(float(dt[0] / ot[0]), 2),
        },
    }

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\n  Detailed results: {RESULTS}")

    ds_om.close_all()


if __name__ == "__main__":
    main()
