#!/usr/bin/env python3
"""Импорт внешних изображений (свои фото / открытые датасеты / фотобанки) в датасет.

Лицензия каждого источника ОБЯЗАТЕЛЬНО фиксируется (--source, --license) —
валидатор предупреждает на NC/ND/SA; под CC BY 4.0 совместимы только CC0,
CC BY, public domain и собственные материалы.

Формат входной разметки — CSV (разделитель ;) с заголовком:

    image;plate_num;plate_type;bbox[;conditions[;is_vehicle]]

где image — имя файла в --images, bbox = x,y,w,h (пиксели). quad при импорте
синтезируется по bbox (валидный порядок «ЛВ -> по часовой»); если у источника
есть quad, добавьте колонку quad (8 чисел) — она будет подхвачена.

Пример:
    python tools/import_public.py --images ~/raw/taxi --labels taxi.csv \
        --source "own dashcam 2026-05" --license own_photo --blur-faces
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from solution.plate_format import repair, CLASS_OF_TYPE  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="каталог исходных изображений")
    ap.add_argument("--labels", required=True, help="csv-разметка (см. докстринг)")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--source", required=True, help="источник (own_photo / URL / название)")
    ap.add_argument("--license", required=True, help="лицензия (own_photo, CC0, CC-BY-4.0...)")
    ap.add_argument("--prefix", default="real_pub", help="префикс имён файлов")
    ap.add_argument("--default-conditions", default="day")
    ap.add_argument("--blur-faces", action="store_true", help="размыть лица при импорте")
    args = ap.parse_args(argv)

    real_dir = os.path.join(args.dataset, "images", "real")
    lab_dir = os.path.join(args.dataset, "labels")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    try:
        from blur_faces import blur_faces_inplace
    except Exception:
        blur_faces_inplace = None

    with open(args.labels, encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f, delimiter=";"))

    seen_md5 = {}
    out_rows = []
    for r in rows_in:
        src = os.path.join(args.images, os.path.basename(r["image"].strip()))
        if not os.path.isfile(src):
            print(f"  ПРОПУСК: нет файла {src}")
            continue
        img = cv2.imread(src)
        if img is None:
            print(f"  ПРОПУСК: не читается {src}")
            continue
        h, w = img.shape[:2]
        if args.blur_faces and blur_faces_inplace:
            img = blur_faces_inplace(img)
        data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        import hashlib
        md5 = hashlib.md5(data).hexdigest()
        if md5 in seen_md5:
            print(f"  ПРОПУСК: дубликат {r['image']} == {seen_md5[md5]}")
            continue

        ptype = r.get("plate_type", "other").strip() or "other"
        plate, ok = repair(r.get("plate_num", ""), ptype)
        bbox = [int(float(v)) for v in r["bbox"].split(",")[:4]]
        if r.get("quad"):
            q = [int(float(v)) for v in r["quad"].split(",")[:8]]
            quad = np.float32([(q[i], q[i + 1]) for i in range(0, 8, 2)])
        else:
            x, y, bw, bh = bbox
            quad = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
        from solution.rectify import order_quad_clockwise_tl
        quad = np.clip(order_quad_clockwise_tl(quad), 0, [w - 1, h - 1]).astype(int)

        stem = f"{args.prefix}_{len(seen_md5):05d}"
        dst = os.path.join(real_dir, stem + ".jpg")
        with open(dst, "wb") as f:
            f.write(data)
        seen_md5[md5] = r["image"]

        x, y, bw, bh = bbox
        labels = [(f"{CLASS_OF_TYPE.get(ptype, 3)} "
                   f"{(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                   f"{bw / w:.6f} {bh / h:.6f} " +
                   " ".join(f"{v:.6f}" for v in (quad / np.array([w, h])).reshape(-1)))]
        with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")

        out_rows.append({
            "image": f"images/real/{stem}.jpg",
            "plate_num": plate,
            "plate_type": ptype,
            "bbox": f"{x},{y},{bw},{bh}",
            "quad": ",".join(str(int(v)) for v in quad.reshape(-1)),
            "is_vehicle": int(r.get("is_vehicle", "1") or 1),
            "is_synthetic": 0,
            "source": args.source,
            "license": args.license,
            "conditions": (r.get("conditions") or args.default_conditions).strip(),
        })
        if not ok:
            print(f"  ВНИМАНИЕ: «{r.get('plate_num')}» не лег в маску {ptype} -> "
                  f"«{plate}» — проверьте")

    with open(os.path.join(args.dataset, "real_rows.jsonl"), "a", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Импортировано: {len(out_rows)} знаков; непрочитанных: {len(rows_in) - len(out_rows)}")
    print("Далее: python tools/dataset.py meta && python tools/dataset.py validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
