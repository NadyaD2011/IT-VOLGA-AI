#!/usr/bin/env python3
"""Размытие лиц на фото/видео-кадрах датасета (требование п. 6.3.3).

OpenCV Haar-cascade (офлайн, входит в opencv-python). Использование:

  python tools/blur_faces.py --dir dataset/images/real [--inplace] [--backup backup/]
  python tools/blur_faces.py --dir raw_photos/            # создаёт raw_photos_blurred/

Можно зациклить на видеокадр: сначала выдернуть кадры ffmpeg'ом, потом прогнать
этот скрипт. Лица на знаке/под ним — редкость, но правило организаторов
строгое: лучше перестраховаться.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

_CASCADE = None


def get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    return _CASCADE


def blur_faces_inplace(img: np.ndarray, min_size: int = 24, strength: float = 0.5) -> np.ndarray:
    """Возвращает копию кадра с размытыми лицами (и upperbody — для профилей)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = get_cascade().detectMultiScale(gray, 1.15, 5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return img
    out = img.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in faces:
        x0, y0 = max(0, x - int(0.25 * w)), max(0, y - int(0.25 * h))
        x1, y1 = min(W, x + int(1.35 * w)), min(H, y + int(1.6 * h))
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        k = int(max(3, min(roi.shape[:2]) // 3))
        k |= 1
        roi_b = cv2.GaussianBlur(roi, (k, k), 0)
        out[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - strength, roi_b, strength, 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="каталог изображений")
    ap.add_argument("--out", default=None, help="куда писать (по умолчанию <dir>_blurred)")
    ap.add_argument("--inplace", action="store_true", help="перезаписывать на месте")
    ap.add_argument("--backup", default=None, help="сделать копию originals перед --inplace")
    args = ap.parse_args(argv)

    out_dir = args.dir if args.inplace else (args.out or args.dir.rstrip("/") + "_blurred")
    os.makedirs(out_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png")
    n = hit = 0
    for name in sorted(os.listdir(args.dir)):
        if not name.lower().endswith(exts):
            continue
        p = os.path.join(args.dir, name)
        img = cv2.imread(p)
        if img is None:
            continue
        blurred = blur_faces_inplace(img)
        if blurred.tobytes() != img.tobytes():
            hit += 1
        if args.inplace and args.backup:
            os.makedirs(args.backup, exist_ok=True)
            shutil.copy2(p, os.path.join(args.backup, name))
        cv2.imwrite(os.path.join(out_dir, name), blurred,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"Обработано {n}, кадров с найденными лицами: {hit}. Результат: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
