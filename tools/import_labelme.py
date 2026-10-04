#!/usr/bin/env python3
"""Конвертер аннотаций LabelMe / X-AnyLabeling (прямоугольник или 4-угольник)
в формат датасета. Удобен, когда разметка велась в полноценном GUI
(например, X-AnyLabeling: рамка + полигон), а не в tools/label_assist.py.

Запуск:
    python tools/import_labelme.py --dir path/to/labelme_project \
        --dataset dataset --source own_photo --license "own_photo (CC BY 4.0)"

Ожидается: для каждого img.jpg файл img.json с shapes[]. Настройка маппинга
метки -> тип знака: --map "plate=type1,square=type1a,taxi=type1b,other=other"
По умолчанию label==type... принимается как есть; иначе other.
Текст знака читается из shape['text'] (норма: 'A123BC777'; нечитаемое — '#').
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution.plate_format import repair, CLASS_OF_TYPE  # noqa: E402
from solution.rectify import order_quad_clockwise_tl  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


def parse_map(spec: str) -> dict:
    m = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        m[k.strip().lower()] = v.strip()
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="каталог с img.jpg+img.json (LabelMe)")
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--source", default="own_photo")
    ap.add_argument("--license", default="own_photo (CC BY 4.0)")
    ap.add_argument("--map", default="", help="label->type, напр. plate1=type1,taxi=type1b")
    ap.add_argument("--conditions", default="day,angle")
    ap.add_argument("--prefix", default="real_lm")
    ap.add_argument("--blur-faces", action="store_true")
    args = ap.parse_args(argv)

    mapping = parse_map(args.map)
    real_dir = os.path.join(args.dataset, "images", "real")
    lab_dir = os.path.join(args.dataset, "labels")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    jsons = sorted(f for f in os.listdir(args.dir) if f.endswith(".json"))
    out_rows, n = [], 0
    for jf in jsons:
        jpath = os.path.join(args.dir, jf)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        imgname = data.get("imagePath", jf[:-5] + ".jpg")
        src = os.path.join(args.dir, os.path.basename(imgname))
        if not os.path.isfile(src) or cv2 is None:
            print(f"  ПРОПУСК: нет изображения {imgname}")
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        if args.blur_faces:
            sys.path.insert(0, HERE)
            from blur_faces import blur_faces_inplace
            img = blur_faces_inplace(img)
        shapes = [s for s in data.get("shapes", [])
                  if s.get("shape_type") in ("rectangle", "polygon") and len(s.get("points", [])) in (2, 4)]
        if not shapes:
            continue
        stem = f"{args.prefix}_{n:05d}"
        cv2.imwrite(os.path.join(real_dir, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels = []
        for s in shapes:
            pts = np.float32(s["points"])
            if len(pts) == 2:
                x0, y0 = pts[0]
                x1, y1 = pts[1]
                quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            else:
                quad = pts[:4]
            quad = np.clip(order_quad_clockwise_tl(quad), 0, [W - 1, H - 1]).astype(int)
            x1, y1 = quad[:, 0].min(), quad[:, 1].min()
            x2, y2 = quad[:, 0].max(), quad[:, 1].max()
            bx, by = int(x1), int(y1)
            bw, bh = int(x2 - x1), int(y2 - y1)
            label_raw = (s.get("label") or "").lower()
            ptype = mapping.get(label_raw) or (label_raw if label_raw in CLASS_OF_TYPE else "other")
            text = (s.get("text") or s.get("description") or "").strip()
            plate, ok = repair(text, ptype)
            if ptype == "other" and not plate:
                plate = ""
            labels.append(f"{CLASS_OF_TYPE.get(ptype, 3)} "
                          f"{(bx + bw / 2) / W:.6f} {(by + bh / 2) / H:.6f} "
                          f"{bw / W:.6f} {bh / H:.6f} " +
                          " ".join(f"{v:.6f}" for v in (quad / np.array([W, H])).reshape(-1)))
            out_rows.append({
                "image": f"images/real/{stem}.jpg",
                "plate_num": plate, "plate_type": ptype,
                "bbox": f"{bx},{by},{bw},{bh}",
                "quad": ",".join(str(int(v)) for v in quad.reshape(-1)),
                "is_vehicle": 1, "is_synthetic": 0,
                "source": args.source, "license": args.license,
                "conditions": args.conditions,
            })
            if not ok:
                print(f"  ВНИМАНИЕ {imgname}: текст «{text}» не лег в маску {ptype}")
        with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")
        n += 1

    with open(os.path.join(args.dataset, "real_rows.jsonl"), "a", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Импортировано кадров: {n}, знаков: {len(out_rows)}")
    print("Далее: python tools/dataset.py meta && python tools/dataset.py validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
