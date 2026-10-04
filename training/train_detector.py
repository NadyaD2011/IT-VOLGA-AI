#!/usr/bin/env python3
"""Обучение детектора знаков (YOLOv8n через Ultralytics) на датасете из dataset/.

Что происходит:
  1. из dataset/meta.csv берутся изображения и классы (type1=0, type1a=1,
     type1b=2, other=3); из dataset/labels/*.txt — геометрия (берутся первые
     5 колонок «class cx cy w h», хвост с quad детектору не нужен);
  2. собирается рабочий каталог .runs/det/dataset (копии/симлинки train/val);
  3. стартует обучение YOLO (по умолчанию yolov8n.pt — 3 МБ весов, ~10 мс/кадр
     на GTX 1050 Ti при imgsz 640);
  4. лучший чекпоинт копируется в solution/models/detector.pt.

Запуск (из корня проекта):
    python training/train_detector.py --epochs 120 --imgsz 640 --device cuda --batch 16
    python training/train_detector.py --base solution/models/detector.pt   # дообучение
Зависимости: pip install ultralytics (сам поставит torch; при первом запуске
скачивается yolov8n.pt — дальше работа офлайн).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def split_by_image(image: str, val_frac: float) -> str:
    h = int(hashlib.md5(image.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < int(val_frac * 1000) else "train"


def build_workdir(dataset: str, work: str, val_frac: float, copy: bool = True) -> str:
    import csv
    images: dict[str, dict] = {}
    with open(os.path.join(dataset, "meta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            images.setdefault(r["image"], {"cls": set()})
    if not images:
        sys.exit("meta.csv пуст — сначала: python tools/dataset.py meta")
    # классы берём из labels/*.txt (там уже нормированные 5+8)
    for img_rel in images:
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        lab = os.path.join(dataset, "labels", stem + ".txt")
        rows = []
        if os.path.isfile(lab):
            with open(lab, encoding="utf-8") as f:
                for ln in f:
                    toks = ln.split()
                    if len(toks) >= 5:
                        rows.append(" ".join(toks[:5]))
        images[img_rel]["label"] = rows

    for split in ("train", "val"):
        os.makedirs(os.path.join(work, "images", split), exist_ok=True)
        os.makedirs(os.path.join(work, "labels", split), exist_ok=True)
    n_tr = n_vl = 0
    for img_rel, info in images.items():
        src = os.path.join(dataset, img_rel.replace("/", os.sep))
        if not os.path.isfile(src):
            continue
        split = split_by_image(img_rel, val_frac)
        stem = os.path.splitext(os.path.basename(img_rel))[0]
        dst_i = os.path.join(work, "images", split, stem + os.path.splitext(src)[1])
        dst_l = os.path.join(work, "labels", split, stem + ".txt")
        if copy:
            if not os.path.isfile(dst_i):
                shutil.copy2(src, dst_i)
        else:
            if os.path.islink(dst_i):
                os.unlink(dst_i)
            elif os.path.isfile(dst_i):
                os.remove(dst_i)
            os.symlink(os.path.abspath(src), dst_i)
        with open(dst_l, "w", encoding="utf-8") as f:
            f.write("\n".join(info["label"]) + ("\n" if info["label"] else ""))
        n_tr += split == "train"
        n_vl += split == "val"
    yaml_path = os.path.join(work, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(work)}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: type1\n  1: type1a\n  2: type1b\n  3: other\n")
    print(f"Рабочий каталог: {work} (train={n_tr}, val={n_vl})")
    if n_tr < 20:
        print("ВНИМАНИЕ: слишком мало обучающих кадров — соберите датасет "
              "(см. README, разделы «Синтетика» и «Реальные данные»).", file=sys.stderr)
    return yaml_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--base", default="yolov8n.pt",
                    help="базовая модель: yolov8n.pt (нужен интернет 1 раз) или путь к .pt")
    ap.add_argument("--work", default=os.path.join(ROOT, ".runs", "det"))
    ap.add_argument("--out", default=os.path.join(ROOT, "solution", "models", "detector.pt"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1")
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--resume-training", action="store_true")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers (0 — экономнее по RAM)")
    args = ap.parse_args(argv)

    work = os.path.abspath(args.work)
    data = build_workdir(args.dataset, work, args.val_frac)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Нужен пакет ultralytics: pip install ultralytics")
    import torch
    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(args.base)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=args.seed,
        project=work,
        name="grz",
        exist_ok=True,
        workers=args.workers,
        patience=40,
        cos_lr=True,
        # аугментации: умеренные, т.к. синтетика уже аугментирована генератором
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45,
        degrees=6.0, translate=0.12, scale=0.45, shear=2.5,
        perspective=0.0008, flipud=0.0, fliplr=0.0,
        mosaic=0.6, mixup=0.05, copy_paste=0.0,
        resume=args.resume_training,
    )
    best = os.path.join(work, "grz", "weights", "best.pt")
    if os.path.isfile(best):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shutil.copy2(best, args.out)
        print(f"Детектор сохранён: {args.out}")
    else:
        print(f"Не найден {best} — проверьте лог обучения")
    # валидация качества типов
    if os.path.isfile(args.out):
        try:
            m = YOLO(args.out)
            metrics = m.val(data=data, imgsz=args.imgsz, device=device, plots=False)
            print("mAP50-95:", getattr(metrics, "map", None))
        except Exception as e:
            print(f"валидация пропущена: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
