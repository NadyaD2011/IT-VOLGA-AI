#!/usr/bin/env python3
"""Прогон решения по каталогу + оценка организаторским evaluate.py (удобно для
самопроверки на отладочном наборе и smoke-тестах).

    python tests/run_infer_eval.py --images DIR --gt meta_or_labels.csv \
        [--det /tmp/models/detector.pt --crnn /tmp/models/crnn.pt --imgsz 320]

GT-файл: format «image;plate_num;plate_type;is_vehicle». Если у GT 10 колонок
(формат meta.csv датасета) — columns игнорируются, берутся первые 4.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--det", default=None)
    ap.add_argument("--crnn", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = {"models": {"imgsz": args.imgsz, "conf": 0.3, "device": "auto"},
           "recognition": {"device": "auto"}, "post": {"conf_out_min": 0.1}}
    if args.det:
        cfg["models"].update({"detector_weights": args.det, "backend": "ultralytics"})
    else:
        cfg["models"]["backend"] = "contour"
    if args.crnn:
        cfg["recognition"]["weights"] = args.crnn
    else:
        cfg["recognition"]["weights"] = None

    from solution.pipeline import PlateReader
    reader = PlateReader(cfg)
    out = args.out or os.path.join(tempfile.mkdtemp(prefix="grz_pred_"), "pred.csv")
    rows, st = reader.process_dir(args.images, out_csv=out)
    print(f"rows={len(rows)} avg={st.avg_ms:.1f} мс -> {out}")

    # нормализуем gt к 4 колонкам
    gt4 = out + ".gt4.csv"
    with open(args.gt, encoding="utf-8-sig") as f, \
         open(gt4, "w", encoding="utf-8", newline="") as o:
        rd = csv.reader(f, delimiter=";")
        wr = csv.writer(o, delimiter=";", lineterminator="\n")
        for i, r in enumerate(rd):
            if i == 0:
                wr.writerow(["image", "plate_num", "plate_type", "is_vehicle"])
            else:
                wr.writerow(r[:4])
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "organizers", "evaluate.py"),
                        "--gt", gt4, "--pred", out, "--images", args.images])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
