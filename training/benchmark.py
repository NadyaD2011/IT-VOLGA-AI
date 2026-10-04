#!/usr/bin/env python3
"""Замер скорости решения (требование 100 мс/кадр, раздел 3 п. 2).

    python training/benchmark.py --images путь/к/картинкам --iters 3

Пишет avg/p50/p95 времени на кадр по стадиям (детекция/всё) и сводку по железу —
эти числа вставляются в пояснительную записку.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--config", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args(argv)
    from solution.cli import load_config
    from solution.pipeline import PlateReader

    cfg = load_config(args.config)
    reader = PlateReader(cfg)
    files = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".png")))
    if not files:
        print("нет изображений"); return 1

    for f in files[:args.warmup]:
        reader.read_image(f)
    ts = []
    for rep in range(args.iters):
        for f in files:
            t0 = time.perf_counter()
            reader.read_image(f)
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    import platform
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
        torchv = torch.__version__
    except Exception:
        gpu, torchv = "n/a", "n/a"
    print(f"Кадров: {len(files)}, прогонов: {args.iters}, замеров: {len(ts)}")
    print(f"avg={ts.mean():.1f} мс, p50={np.percentile(ts,50):.1f}, "
          f"p95={np.percentile(ts,95):.1f}, max={ts.max():.1f}")
    print(f"Процессор: {platform.processor() or platform.machine()}; torch: {torchv}; device: {gpu}")
    print("Вставьте эти цифры в docs/пояснительная_записка.md, раздел «Скорость».")
    return 0


if __name__ == "__main__":
    sys.exit(main())
