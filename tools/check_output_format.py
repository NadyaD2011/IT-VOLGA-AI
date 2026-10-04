#!/usr/bin/env python3
"""Проверка CSV решения к формату раздела 5 и совместимости с evaluate.py.

    python tools/check_output_format.py result.csv
    python tools/check_output_format.py result.csv --gt debug/debug_labels.csv --images debug_set/

Без --gt: только формат. С --gt: дополнительно прогоняет эталонный evaluate.py
(метрики и балл жюри в упрощённом виде).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution.plate_format import MASKS, PLATE_TYPES  # noqa: E402

HEADER = ["image", "plate_num", "plate_type", "confidence"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--images", default=None)
    args = ap.parse_args(argv)

    problems = []
    with open(args.csv_path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print("ERROR: файл не UTF-8"); return 1
    lines = text.splitlines()
    if not lines:
        print("ERROR: пустой файл"); return 1
    head = [h.strip() for h in lines[0].split(";")]
    if head != HEADER:
        problems.append(f"заголовок {head} != {HEADER}")
    known_images = None
    if args.images and os.path.isdir(args.images):
        known_images = {f for f in os.listdir(args.images)
                        if f.lower().endswith((".jpg", ".png"))}
    n = 0
    for i, ln in enumerate(lines[1:], start=2):
        if not ln.strip():
            continue
        parts = ln.split(";")
        if len(parts) != 4:
            problems.append(f"строка {i}: {len(parts)} полей вместо 4"); continue
        img, plate, ptype, conf = [p.strip() for p in parts]
        n += 1
        if os.path.basename(img) != img:
            problems.append(f"{i}: image должен быть без пути")
        if known_images is not None and img not in known_images:
            problems.append(f"{i}: изображение «{img}» не из тестового каталога")
        if ptype not in PLATE_TYPES:
            problems.append(f"{i}: plate_type «{ptype}» не из {PLATE_TYPES}")
        elif not MASKS[ptype].fullmatch(plate):
            problems.append(f"{i}: plate_num «{plate}» не по маске {ptype}")
        try:
            c = float(conf)
            if not 0.0 <= c <= 1.0:
                problems.append(f"{i}: confidence {c} вне [0,1]")
        except ValueError:
            problems.append(f"{i}: confidence «{conf}» не число")

    print(f"Строк: {n}. Проблем формата: {len(problems)}")
    for p in problems[:40]:
        print("  " + p)
    if len(problems) > 40:
        print(f"  ... ещё {len(problems) - 40}")

    if args.gt:
        cmd = [sys.executable, os.path.join(HERE, "organizers", "evaluate.py"),
               "--gt", args.gt, "--pred", args.csv_path]
        if args.images:
            cmd += ["--images", args.images]
        subprocess.call(cmd, cwd=ROOT)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
