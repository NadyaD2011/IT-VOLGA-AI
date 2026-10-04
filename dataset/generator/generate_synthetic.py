#!/usr/bin/env python3
"""Генератор синтетического датасета нестандартных ГРЗ (Волга-IT 2026).

Создаёт в датасете:
    images/synthetic/syn_%06d.jpg   — кадр с наложенным знаком
    labels/syn_%06d.txt             — YOLO-метка «class cx cy w h x1 y1..x4 y4» (нормиров.)
    synth_rows.jsonl                — «строки» для meta.csv (одна на знак)

Набор полностью воспроизводим: заданный --seed (numpy default_rng) детерминирует
выбор фонов, текстов, ракурсов и аугментаций. Повторный запуск с тем же seed и
тем же --count даёт побайтово идентичные номера знаков и разметку.

Зависимости: numpy, Pillow (см. requirements.txt рядом). Без интернета.

Пример:
    python generate_synthetic.py --count 6000 --seed 20260420
    python generate_synthetic.py --dataset ../../dataset --count 12000

После генерации пересоберите meta.csv:  python ../../tools/dataset.py meta
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backgrounds as bgs  # noqa: E402
from plate_render import render_plate, LETTERS, DIGITS  # noqa: E402

DEFAULT_DATASET = os.path.dirname(HERE)  # dataset/

# Коды регионов: частотные + обычные (взвешенно). Как на дорогах РФ, включая
# «новые» трёхзначные 125/225/252/323/550/750/797/...
FREQUENT = ["77", "99", "777", "799", "177", "199", "50", "150", "550", "750",
            "23", "93", "123", "163", "234", "323", "78", "98", "178", "278",
            "16", "116", "216", "25", "125", "225", "525", "74", "174", "274",
            "61", "161", "54", "154", "34", "134", "42", "142", "45", "145",
            "47", "52", "152", "56", "62", "72", "86", "164", "166", "18",
            "261", "96", "99", "748", "816", "502", "503"]
REGULAR = [f"{r:02d}" for r in range(1, 90)]

BG_SIZES = [(800, 600), (960, 540), (1024, 768), (1280, 720), (896, 504), (720, 540)]


def make_text(rng, ptype: str) -> str:
    L, D = LETTERS, DIGITS

    def letters(n):
        return "".join(L[rng.randint(0, len(L) - 1)] for _ in range(n))

    def digits(n):
        return "".join(D[rng.randint(0, 9)] for _ in range(n))

    region = rng.choice(FREQUENT) if rng.random() < 0.55 else rng.choice(REGULAR)
    if ptype == "type1b":
        return letters(2) + digits(3) + region
    if ptype == "other":
        return letters(2) + digits(3) + letters(2)          # «европейский» формат
    return letters(1) + digits(3) + letters(2) + region


# --------------------------------------------------------------------------- #
#  Гомография и вклейка
# --------------------------------------------------------------------------- #
def find_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """H: src(4x2) -> dst(4x2), h[2,2]=1."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h, *_ = np.linalg.lstsq(np.float64(A), np.float64(b), rcond=None)
    H = np.eye(3, dtype=np.float64)
    H[0, 0], H[0, 1], H[0, 2] = h[0], h[1], h[2]
    H[1, 0], H[1, 1], H[1, 2] = h[3], h[4], h[5]
    H[2, 0], H[2, 1] = h[6], h[7]
    return H


def warp_paste(bg: np.ndarray, plate: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Вклейка plate (H,P,3) в bg по произвольному выпуклому кваду (4x2, float)."""
    Hh, Ww = bg.shape[:2]
    h, w = plate.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(Ww - 1, x1), min(Hh - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return bg
    src = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = find_homography(src, quad.astype(np.float64))
    Hinv = np.linalg.inv(H)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    ones = np.ones_like(xs, dtype=np.float64)
    P = np.stack([xs, ys, ones], -1) @ Hinv.T
    u = P[..., 0] / P[..., 2]
    v = P[..., 1] / P[..., 2]
    mask = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    if mask.sum() < 0.4 * max(1, (x1 - x0) * (y1 - y0)):
        return bg
    uf = np.clip(u, 0, w - 1.001)
    vf = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(uf).astype(int), np.floor(vf).astype(int)
    fu, fv = uf - u0, vf - v0
    img = bg.copy()
    region = img[y0:y1 + 1, x0:x1 + 1]
    plate_f = plate.astype(np.float32)
    for c in range(3):
        pa = plate_f[v0, u0, c]
        pb = plate_f[v0, u0 + 1, c]
        pc = plate_f[v0 + 1, u0, c]
        pd = plate_f[v0 + 1, u0 + 1, c]
        val = (pa * (1 - fu) * (1 - fv) + pb * fu * (1 - fv) +
               pc * (1 - fu) * fv + pd * fu * fv)
        ch = region[..., c]
        ch[mask] = val[mask]
        region[..., c] = ch
    img[y0:y1 + 1, x0:x1 + 1] = region
    # лёгкая тень под верхним краем (реалистичнее ниша)
    shadow = quad + np.float64([2.5, 2.5])
    return img


# --------------------------------------------------------------------------- #
#  Аугментации
# --------------------------------------------------------------------------- #
def augment(img: np.ndarray, rng, strong=True) -> list[str]:
    conds = []
    pil = Image.fromarray(img)
    night = rng.random() < 0.20
    if night:
        # ночь: фон темнеет, знак частично освещён фарами (поэтому и 0.5, и
        # последующая подсветка «светлых» пикселей)
        f = np.float32(pil)
        lit = f.max(-1, keepdims=True) > 150
        f = f * np.where(lit, rng.uniform(0.80, 0.95), rng.uniform(0.30, 0.5))
        f[..., 2] *= 1.10
        f += rng.normal(0, 8.0, f.shape)
        pil = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
        conds.append("night")
    else:
        conds.append("day")
        if rng.random() < 0.35:
            pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(1.05, 1.35))
    if rng.random() < 0.30:
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.7)))
    if strong and rng.random() < 0.18:  # боковое свечение/блик (аддитивно!)
        w, h = pil.size
        gl = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(gl)
        y0 = rng.uniform(0, h * 0.7)
        d.polygon([(0, y0), (w, y0 + h * 0.2), (w, y0 + h * 0.45), (0, y0 + h * 0.28)],
                  fill=255)
        gl = gl.filter(ImageFilter.GaussianBlur(h * 0.10))
        a = rng.uniform(0.10, 0.22)
        add = np.float32(gl)[..., None] * a
        pil = Image.fromarray(np.clip(np.float32(pil) + add * np.float32((1.0, 1.0, 0.95)),
                                      0, 255).astype(np.uint8))
        conds.append("glare")
    if strong and rng.random() < 0.20:  # дождь: штрихи + мыло
        w, h = pil.size
        ov = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(ov)
        for _ in range(rng.randint(40, 140)):
            x, y = rng.random() * w, rng.random() * h
            ln = rng.uniform(8, 26)
            d.line([x, y, x + ln * 0.3, y + ln], fill=255, width=1)
        arr = np.clip(np.float32(pil) + np.float32(ov)[..., None] * rng.uniform(0.12, 0.28),
                      0, 255)
        pil = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
        conds.append("rain")
    if strong and rng.random() < 0.10:  # снег: точки
        w, h = pil.size
        d = ImageDraw.Draw(pil)
        for _ in range(rng.randint(60, 220)):
            x, y = rng.random() * w, rng.random() * h
            r = rng.uniform(0.8, 2.6)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(245, 245, 248))
        conds.append("snow")
    if strong and rng.random() < 0.30:  # грязь: разводы (blend только под пятнами)
        w, h = pil.size
        ov = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(ov)
        for _ in range(rng.randint(4, 14)):
            x, y = rng.random() * w, rng.random() * h
            rw, rh = rng.uniform(0.03, 0.2) * w, rng.uniform(0.02, 0.1) * h
            d.ellipse([x - rw, y - rh, x + rw, y + rh], fill=255)
        ov = ov.filter(ImageFilter.GaussianBlur(max(2, w // 90)))
        alpha = (np.float32(ov)[..., None] / 255.0) * rng.uniform(0.15, 0.45)
        brown = np.float32((92, 78, 58))[None, None, :]
        arr = np.float32(pil) * (1 - alpha) + brown * alpha
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        conds.append("dirt")
    if strong and rng.random() < 0.16:  # смаз движения (гориз. усреднение)
        arr = np.float32(pil)
        k = rng.randint(2, 6)
        acc = np.zeros_like(arr)
        for i in range(k + 1):
            acc += np.roll(arr, -i, axis=1)
        pil = Image.fromarray((acc / (k + 1)).clip(0, 255).astype(np.uint8))
        conds.append("motion_blur")
    if rng.random() < 0.4:
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.85, 1.18))
    return pil, conds


def order_quad(quad):
    """Левый верхний -> по часовой (совпадает с проверкой валидатора)."""
    pts = np.asarray(quad, float).reshape(4, 2)
    first = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -first, axis=0)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area < 0:
        pts = np.concatenate([pts[:1], pts[:0:-1]], axis=0)
    return pts


# --------------------------------------------------------------------------- #
#  Один образец
# --------------------------------------------------------------------------- #
def pick_weighted(rng, mix: dict):
    tot = sum(mix.values())
    r = rng.random() * tot
    acc = 0.0
    for k, v in mix.items():
        acc += v
        if r <= acc:
            return k
    return next(iter(mix))


def gen_one(idx: int, seed: int, args, mix, mode_mix, out_dirs):
    rng = bgs.Rng(seed * 1_000_003 + idx)
    ptype = pick_weighted(rng, mix)
    mode = pick_weighted(rng, mode_mix)
    if ptype == "other" and mode != "billboard" and rng.random() < 0.5:
        mode = "car"
    bw, bh = BG_SIZES[rng.randint(0, len(BG_SIZES) - 1)]
    if mode == "car":
        bg, (cx, cy) = bgs.car_rear(bw, bh, rng)
        is_vehicle = 1
    elif mode == "billboard":
        bg, (cx, cy) = bgs.billboard(bw, bh, rng, screen=rng.random() < 0.4)
        is_vehicle = 0
    else:
        bg, (cx, cy) = (bgs.asphalt(bw, bh, rng) if rng.random() < 0.6
                        else bgs.grass_bokeh(bw, bh, rng))
        is_vehicle = 0

    text = make_text(rng, ptype)
    other_style = (rng.choice(("eu", "mil", "transit", "moto"))
                   if ptype == "other" else None)
    plate = render_plate(text, ptype, scale=2, other_style=other_style).convert("RGB")
    pw, ph = plate.size
    target_w = int(bw * rng.uniform(0.15, 0.52))
    target_w = int(np.clip(target_w, 40, int(bw * 0.85)))
    target_h = int(round(target_w * ph / pw))
    if target_h < 10:
        target_h = 10
    plate = plate.resize((target_w, target_h), Image.LANCZOS)

    jit = rng.uniform(0.02, 0.11)
    cx = int(np.clip(cx * bw, target_w * 0.6 + 8, bw - target_w * 0.6 - 8))
    cy = int(np.clip(cy * bh, target_h * 1.1 + 6, bh - target_h * 1.1 - 6))
    x0, y0 = cx - target_w / 2, cy - target_h / 2
    quad = np.float64([[x0, y0], [x0 + target_w, y0],
                       [x0 + target_w, y0 + target_h], [x0, y0 + target_h]])
    quad += rng.normal(0, jit, (4, 2)) * np.float64([target_w, target_h])
    # уводим в перспективу: верхний край выше/ниже нижнего, боковые сходятся
    persp = rng.uniform(0.0, 0.18)
    quad[0, 1] -= target_h * persp
    quad[1, 1] -= target_h * persp * rng.uniform(0.6, 1.4)
    quad[2, 1] += target_h * persp * rng.uniform(0.0, 0.7)
    quad[3, 1] += target_h * persp * rng.uniform(0.0, 0.7)
    quad[:, 0] = np.clip(quad[:, 0], 3, bw - 4)
    quad[:, 1] = np.clip(quad[:, 1], 3, bh - 4)
    quad = order_quad(quad)

    img = warp_paste(bg, np.asarray(plate), quad)
    pil, conds = augment(img, rng, strong=args.strong)
    if persp > 0.06 or jit > 0.08:
        conds.append("angle")

    name = f"syn_{idx:06d}.jpg"
    out_dir, lab_dir = out_dirs
    q = rng.randint(args.jpeg_min, args.jpeg_max)
    pil.save(os.path.join(out_dir, name), quality=q, optimize=False)

    W, H = pil.size
    # финализируем quad в пикселях: clip -> округление -> повторное упорядочивание
    # (гарантирует «первая точка = левый верхний, обход по часовой» даже после округления)
    quad = order_quad(np.clip(quad, 0, [W - 1, H - 1]).round())
    qx, qy = quad[:, 0], quad[:, 1]
    bx, by = math.floor(qx.min()), math.floor(qy.min())
    bw2, bh2 = math.ceil(qx.max()) - bx, math.ceil(qy.max()) - by
    cls = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}[ptype]
    cxn, cyn = (bx + bw2 / 2) / W, (by + bh2 / 2) / H
    quadc = quad.astype(int)
    quadn = quadc / np.array([W, H], float)
    label = f"{cls} {cxn:.6f} {cyn:.6f} {bw2 / W:.6f} {bh2 / H:.6f} " + \
            " ".join(f"{v:.6f}" for v in quadn.reshape(-1))
    with open(os.path.join(lab_dir, os.path.splitext(name)[0] + ".txt"), "w",
              encoding="utf-8") as f:
        f.write(label + "\n")

    row = {
        "image": f"images/synthetic/{name}",
        "plate_num": text if ptype != "other" else text[:12],
        "plate_type": ptype,
        "bbox": f"{bx},{by},{bw2},{bh2}",
        "quad": ",".join(str(int(v)) for v in quadc.reshape(-1)),
        "is_vehicle": is_vehicle,
        "is_synthetic": 1,
        "source": "generated:script",
        "license": "generated (CC0)",
        "conditions": ",".join(conds),
    }
    return row


def parse_mix(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = float(v) if v else 1.0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="корень датасета")
    ap.add_argument("--count", type=int, default=6000,
                    help="сколько сгенерировать (всего будет id 0..count-1)")
    ap.add_argument("--seed", type=int, default=20260420,
                    help="фиксирующий random seed (воспроизводимость набора)")
    ap.add_argument("--mix", default="type1:29,type1a:30,type1b:29,other:12",
                    help="доли типов, напр. type1:35,type1a:30,type1b:30,other:5")
    ap.add_argument("--mode-mix", default="car:72,flat:18,billboard:10")
    ap.add_argument("--jpeg-min", type=int, default=52)
    ap.add_argument("--jpeg-max", type=int, default=92)
    ap.add_argument("--no-strong-aug", dest="strong", action="store_false",
                    help="отключить тяжёлые аугментации (быстрее, проще)")
    ap.add_argument("--resume", action="store_true",
                    help="пропускать уже существующие файлы (по умолчанию включён)")
    ap.add_argument("--progress", type=int, default=500)
    args = ap.parse_args(argv)

    ds = os.path.abspath(args.dataset)
    out_dir = os.path.join(ds, "images", "synthetic")
    lab_dir = os.path.join(ds, "labels")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    rows_path = os.path.join(ds, "synth_rows.jsonl")

    mix = parse_mix(args.mix)
    mode_mix = parse_mix(args.mode_mix)

    done = 0
    existing = set()
    if os.path.isfile(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    existing.add(os.path.basename(r["image"]))
                except Exception:
                    pass
    fh = open(rows_path, "a", encoding="utf-8")
    for i in range(args.count):
        name = f"syn_{i:06d}.jpg"
        if name in existing and os.path.isfile(os.path.join(out_dir, name)):
            continue
        row = gen_one(i, args.seed, args, mix, mode_mix, (out_dir, lab_dir))
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done += 1
        if args.progress and done % args.progress == 0:
            print(f"... {done} сгенерировано (id до {i})")
            fh.flush()
    fh.close()
    print(f"Готово: {done} новых из {args.count} (seed={args.seed}). "
          f"Метки в labels/, строки в synth_rows.jsonl. "
          f"Пересоберите meta.csv: python tools/dataset.py meta")


if __name__ == "__main__":
    main()
