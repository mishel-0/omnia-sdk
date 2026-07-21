#!/usr/bin/env python3
"""
Reproducibility benchmark: DICOM vs .omnia (v1.01 FIXED)
Requires: omnia-sdk, pydicom, torch, torchvision, numpy

Usage:
    python benchmark.py /path/to/lidc_raw/ /path/to/compressed/ [--epochs 100] [--batch 64] [--workers 4]

Fixes from v1.0:
    - Model created ONCE, not per epoch (fixes memory leak + training bug)
    - Thread-safe GPU monitoring with queue.Queue
    - CLI arguments for all parameters
    - Input validation for directories and CUDA
    - Graceful nvidia-smi fallback
    - Proper cleanup and resource management
    - Reproducibility seed
    - Structured logging
"""
import argparse
import sys
import time
import json
import threading
import subprocess
import queue
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from omnia_sdk.dataset import DicomDataset, OmniaDataset

# Default settings matching the reference benchmark
DEFAULT_BATCH = 64
DEFAULT_EPOCHS = 100
DEFAULT_WORKERS = 4
DEFAULT_PREFETCH = 2

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DICOM vs .omnia training performance"
    )
    parser.add_argument("raw_dir", type=Path, help="Path to raw DICOM directory")
    parser.add_argument("omnia_dir", type=Path, help="Path to .omnia directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default: {DEFAULT_BATCH})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"DataLoader workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH,
                        help=f"Prefetch factor (default: {DEFAULT_PREFETCH})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-gpu-monitor", action="store_true",
                        help="Disable GPU monitoring (for systems without nvidia-smi)")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"),
                        help="Output JSON file path")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate all input paths and environment."""
    errors = []

    if not args.raw_dir.exists():
        errors.append(f"Raw DICOM directory does not exist: {args.raw_dir}")
    if not args.omnia_dir.exists():
        errors.append(f".omnia directory does not exist: {args.omnia_dir}")

    if not torch.cuda.is_available():
        errors.append("CUDA is not available. This benchmark requires a GPU.")
    else:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.1f GB",
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    logger.info("Raw DICOM: %s", args.raw_dir)
    logger.info(".omnia: %s", args.omnia_dir)


def make_model(num_classes: int = 2) -> nn.Module:
    """Create ResNet-18 model adapted for single-channel CT input."""
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m = m.cuda()
    return m


def make_optimizer(model: nn.Module, lr: float = 0.001) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def make_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


class GPUMonitor:
    """Thread-safe GPU utilization monitor using queue.Queue."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._queue: queue.Queue[float] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _monitor(self) -> None:
        """Background thread: poll nvidia-smi."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, check=True
                )
                util = float(result.stdout.strip())
                self._queue.put(util)
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        """Stop monitoring and return collected values."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        values = []
        while not self._queue.empty():
            try:
                values.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return values


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_gpu_monitor: bool = True
) -> Tuple[float, float]:
    """Train one epoch and return (elapsed_time, avg_gpu_utilization)."""

    monitor = GPUMonitor() if use_gpu_monitor else None
    if monitor:
        monitor.start()

    model.train()
    t0 = time.perf_counter()

    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0

    gpu_values = monitor.stop() if monitor else []
    avg_gpu = np.mean(gpu_values) if gpu_values else 0.0

    return round(elapsed, 2), round(float(avg_gpu), 1)


def benchmark_dataset(
    dataset_name: str,
    dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    epochs: int,
    use_gpu_monitor: bool
) -> dict:
    """Run full benchmark on a single dataset."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    logger.info("--- %s ---", dataset_name)
    logger.info("Samples: %d, Batches per epoch: %d", len(dataset), len(loader))

    # Create model ONCE — not per epoch!
    model = make_model()
    optimizer = make_optimizer(model)
    criterion = make_criterion()

    results = []
    for ep in range(epochs):
        t, gu = train_epoch(model, loader, criterion, optimizer, use_gpu_monitor)
        results.append({"epoch": ep + 1, "time": t, "gpu": gu})

        if (ep + 1) % 10 == 0 or ep == 0:
            tag = "cold" if ep == 0 else "hot"
            logger.info("  Epoch %d/%d (%s): %.2fs GPU:%.1f%%", ep+1, epochs, tag, t, gu)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "epoch_times": [r["time"] for r in results],
        "gpu_util": [r["gpu"] for r in results],
        "total_time_s": sum(r["time"] for r in results),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    validate_inputs(args)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info("=" * 60)
    logger.info("BENCHMARK CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Model: ResNet-18 (1-channel input)")
    logger.info("Batch: %d, Workers: %d, Prefetch: %d", args.batch, args.workers, args.prefetch)
    logger.info("Epochs: %d, Seed: %d", args.epochs, args.seed)
    logger.info("GPU Monitor: %s", "disabled" if args.no_gpu_monitor else "enabled")

    # Load datasets
    ds_dcm = DicomDataset(args.raw_dir)
    ds_om = OmniaDataset(args.omnia_dir)

    if len(ds_dcm) != len(ds_om):
        logger.warning("Dataset sizes differ! DICOM=%d, .omnia=%d", len(ds_dcm), len(ds_om))

    # Benchmark
    dcm_results = benchmark_dataset(
        "DICOM", ds_dcm, args.batch, args.workers, args.prefetch,
        args.epochs, not args.no_gpu_monitor
    )

    om_results = benchmark_dataset(
        ".omnia", ds_om, args.batch, args.workers, args.prefetch,
        args.epochs, not args.no_gpu_monitor
    )

    # Analysis
    dcm_arr = np.array(dcm_results["epoch_times"])
    om_arr = np.array(om_results["epoch_times"])

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("DICOM:  mean=%.2fs, std=%.2fs, total=%.1fs",
                dcm_arr.mean(), dcm_arr.std(), dcm_results["total_time_s"])
    logger.info(".omnia: mean=%.2fs, std=%.2fs, total=%.1fs",
                om_arr.mean(), om_arr.std(), om_results["total_time_s"])
    logger.info("Cold start speedup: %.2fx", dcm_arr[0] / om_arr[0])
    if args.epochs >= 50:
        steady_dcm = dcm_arr[50:].mean()
        steady_om = om_arr[50:].mean()
        logger.info("Steady state speedup: %.2fx", steady_dcm / steady_om)
    logger.info("Overall speedup: %.2fx", dcm_results["total_time_s"] / om_results["total_time_s"])

    # Save
    output = {
        "config": {
            "batch": args.batch,
            "epochs": args.epochs,
            "workers": args.workers,
            "prefetch": args.prefetch,
            "seed": args.seed,
            "slices": len(ds_dcm),
        },
        "dicom": dcm_results,
        "omnia": om_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Cleanup
    if hasattr(ds_om, 'close_all'):
        ds_om.close_all()


if __name__ == "__main__":
    main()
