#!/usr/bin/env python3
"""Сборка обучающей выборки для CRNN из dataset/.

Для каждого знака (type1/1a/1b) кадр выпрямляется по quad, режется по той же
геометрии, что и инференс (solution/crops.py), и каждая часть сохраняется со
своей подписью:

    type1/1b:  body -> «A123BC» (или «AB123»),  region -> «777»
    type1a:    top -> «A123»,  series -> «BC»,  region -> «125»

Подписи с «#» в позиции — вырезаются из строки-цели? Нет: позиции-«#» помечаются
символом пропуска и удаляются из цели (CTC обучается только на читаемых);
если читаемых символов в части < 2 — часть не используется.

Для устойчивости к шуму разметки добавляется ±5% джиттер рамок.

Выход: .runs/rec/{train,val}/<part>/<stem>_i.png + manifest.tsv ("rel\ttext").

Запуск: python training/build_rec_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution.crops import GEOM, PLAN_ORDER  # noqa: E402
from solution.plate_format import BODY_LEN  # noqa: E402
from solution.rectify import rectify, order_quad_clockwise_tl  # noqa: E402


def split(image: str, val_frac=0.05) -> str:
    h = int(hashlib.md5(image.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < int(val_frac * 1000) else "train"


def part_targets(plate: str, ptype: str, mode: str):
    """{part_name: текст_части} из полного номера."""
    body = BODY_LEN.get(ptype, 6)
    if mode == "double":
        return {"top": plate[:4], "series": plate[4:6], "region": plate[6:]}
    return {"body": plate[:body], "region": plate[body:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, ".runs", "rec"))
    ap.add_argument("--img-h", type=int, default=52, help="высота выпрямления знака")
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--no-quad", action="store_true", help="кроить по bbox (если quad нет/плохой)")
    args = ap.parse_args(argv)

    ds = args.dataset
    rows = []
    with open(os.path.join(ds, "meta.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    if not rows:
        sys.exit("meta.csv пуст — сначала: python tools/dataset.py meta")

    out_root = os.path.abspath(args.out)
    manifest = []
    n_img = 0
    for r in rows:
        if r["plate_type"] not in ("type1", "type1a", "type1b"):
            continue
        img_path = os.path.join(ds, r["image"].replace("/", os.sep))
        img = cv2.imread(img_path)
        if img is None:
            continue
        plate = r["plate_num"]
        if sum(1 for c in plate if c != "#") < 4:
            continue  # почти нечитаемый — обучать нечему
        H, W = img.shape[:2]
        if r["quad"] and not args.no_quad:
            q = [float(v) for v in r["quad"].split(",")]
            quad = order_quad_clockwise_tl(np.float32([(q[i], q[i + 1]) for i in range(0, 8, 2)]))
            x1, y1 = quad[:, 0].min(), quad[:, 1].min()
            x2, y2 = quad[:, 0].max(), quad[:, 1].max()
        else:
            x, y, w, h = map(float, r["bbox"].split(","))
            quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            x1, y1, x2, y2 = x, y, x + w, y + h
        # одно- или двухстрочная раскладка — по аспекту (как в пайплайне)
        ar = (x2 - x1) / max(1e-6, (y2 - y1))
        mode = "double" if (r["plate_type"] == "type1a" or ar < 2.3) else "single"
        crop, (ow, oh) = rectify(img, quad, out_h=args.img_h)
        targets = part_targets(plate, r["plate_type"], mode)
        stem = os.path.splitext(os.path.basename(r["image"]))[0]
        for name in PLAN_ORDER[mode]:
            txt = (targets.get(name) or "").replace("#", "")
            if len(txt) < 1:
                continue
            x0, X1, y0, Y1 = GEOM[mode][name]["box"]
            box = (int(x0 * ow), int(y0 * oh), int(X1 * ow), int(Y1 * oh))
            for j in range(2):  # исходный + джиттер (устойчивость к шуму разметки)
                bx = list(box)
                if j:
                    dx = int(np.random.uniform(-args.jitter, args.jitter) * ow)
                    dy = int(np.random.uniform(-args.jitter, args.jitter) * oh)
                    bx = [max(0, bx[0] + dx), max(0, bx[1] + dy),
                          min(ow, bx[2] + dx), min(oh, bx[3] + dy)]
                piece = crop[bx[1]:bx[3], bx[0]:bx[2]]
                if piece.size == 0 or min(piece.shape[:2]) < 6:
                    continue
                sp = split(f"{r['image']}:{name}:{j}")
                d = os.path.join(out_root, sp, name)
                os.makedirs(d, exist_ok=True)
                fn = f"{stem}_{j}.png"
                cv2.imwrite(os.path.join(d, fn), piece[..., :3] if piece.ndim == 3 else piece)
                gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) if piece.ndim == 3 else piece
                cv2.imwrite(os.path.join(d, fn), gray)
                manifest.append((f"{sp}/{name}/{fn}", txt))
        n_img += 1
        if n_img % 2000 == 0:
            print(f"... обработано знаков: {n_img}")

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest.tsv"), "w", encoding="utf-8") as f:
        for rel, t in manifest:
            f.write(f"{rel}\t{t}\n")
    print(f"Знаков: {n_img}, образцов-строк: {len(manifest)} -> {out_root}/manifest.tsv")
    print("Далее: python training/train_rec.py --data .runs/rec --epochs 40")
    return 0


if __name__ == "__main__":
    sys.exit(main())
