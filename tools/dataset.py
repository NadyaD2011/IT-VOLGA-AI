#!/usr/bin/env python3
"""Менеджер датасета (структура раздела 6.2 задания).

Подкоманды:

  gen-synth  запуск генератора синтетики (пробрасывает аргументы дальше)
             python tools/dataset.py gen-synth --count 6000 --seed 20260420
  meta       собрать meta.csv из synth_rows.jsonl + real_rows.jsonl
  validate   прогнать валидатор организаторов, сохранить отчёт в датасет
  stats      сводка по типам/условиям/уникальности, пишет dataset/STATS.md
  hash       md5 изображений каталога (для проверки, что отладочный набор
             организаторов не попал в датасет)

Только стандартная библиотека (hash-режим) / numpy для статистики.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset")
GEN = os.path.join(DATASET, "generator", "generate_synthetic.py")
VALIDATOR = os.path.join(HERE, "organizers", "validate_dataset.py")

META_COLUMNS = ("image", "plate_num", "plate_type", "bbox", "quad",
                "is_vehicle", "is_synthetic", "source", "license", "conditions")


def read_jsonl(path):
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError as e:
                    print(f"WARNING {path}:{i}: битая строка ({e})", file=sys.stderr)
    return rows


def cmd_meta(_args):
    synth = read_jsonl(os.path.join(DATASET, "synth_rows.jsonl"))
    real = read_jsonl(os.path.join(DATASET, "real_rows.jsonl"))
    seen = {}
    for r in synth + real:
        key = (r["image"], r.get("bbox", ""))
        r.setdefault("is_vehicle", 1)
        r.setdefault("conditions", "day")
        seen[key] = r  # последние (real) перекрывают, если что
    rows = sorted(seen.values(), key=lambda r: (r["image"], r.get("bbox", "")))
    out = os.path.join(DATASET, "meta.csv")
    missing = set()
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(META_COLUMNS)
        for r in rows:
            img_path = os.path.join(DATASET, r["image"].replace("/", os.sep))
            if not os.path.isfile(img_path):
                missing.add(r["image"])
                continue
            w.writerow([r.get(c, "") for c in META_COLUMNS])
    print(f"meta.csv: {len(rows) - len(missing)} строк"
          + (f", ПРОПУЩЕНО без файла: {sorted(missing)[:5]}..." if missing else ""))
    if missing:
        print("Удалите строки или добавьте файлы, затем повторите.", file=sys.stderr)
    return 0


def cmd_validate(args):
    cmd = [sys.executable, VALIDATOR, DATASET,
           "--report", os.path.join(DATASET, "validation_report.txt"),
           "--json", os.path.join(DATASET, "validation_report.json")]
    if args.no_hash:
        cmd.append("--no-hash")
    if args.forbidden_hashes:
        cmd += ["--forbidden-hashes", args.forbidden_hashes]
    code = subprocess.call(cmd)
    print(f"\nКод возврата валидатора: {code} "
          f"(0 — ошибок нет; отчёт: dataset/validation_report.txt)")
    return code


def cmd_stats(_args):
    path = os.path.join(DATASET, "meta.csv")
    if not os.path.isfile(path):
        print("Сначала: python tools/dataset.py meta"); return 1
    from collections import Counter, defaultdict
    real_imgs, uniq = defaultdict(set), defaultdict(set)
    synth_imgs = set()
    conds = Counter()
    types = Counter()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=";"):
            types[r["plate_type"]] += 1
            if r["is_synthetic"] == "0":
                real_imgs[r["plate_type"]].add(r["image"])
                if "#" not in r["plate_num"] and r["plate_num"]:
                    uniq[r["plate_type"]].add(r["plate_num"])
            else:
                synth_imgs.add(r["image"])
            for c in (r["conditions"] or "").split(","):
                if c.strip():
                    conds[c.strip()] += 1
    lines = ["# Статистика датасета", "",
             "| группа | реальных изобр. (мин.) | уникальных ГРЗ (мин.) | синт. изобр. |",
             "|---|---|---|---|",
             f"| type1a | {len(real_imgs['type1a'])} (150) | {len(uniq['type1a'])} (50) | — |",
             f"| type1b | {len(real_imgs['type1b'])} (300) | {len(uniq['type1b'])} (100) | — |",
             f"| other  | {len(real_imgs['other'])} (50) | — | — |",
             f"| всего синтетики | — | — | {len(synth_imgs)} (5000) |",
             "", "Условия съёмки: " + (", ".join(f"{k}:{v}" for k, v in conds.most_common()) or "—"),
             f"\nВсего строк-знаков по типам: {dict(types)}"]
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(DATASET, "STATS.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n-> dataset/STATS.md")
    return 0


def cmd_hash(args):
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    n = 0
    for dirpath, _d, files in os.walk(args.dir):
        for name in sorted(files):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                h = hashlib.md5()
                with open(os.path.join(dirpath, name), "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                out.write(f"{h.hexdigest()}  {name}\n")
                n += 1
    if args.out:
        out.close()
    print(f"md5 {n} файлов -> {args.out or 'stdout'}", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("meta", help="собрать meta.csv")
    p = sub.add_parser("validate", help="прогнать валидатор организаторов")
    p.add_argument("--no-hash", action="store_true")
    p.add_argument("--forbidden-hashes", help="файл md5 отладочного набора")
    sub.add_parser("stats", help="статистика по meta.csv")
    p = sub.add_parser("hash", help="md5 всех изображений каталога")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None)
    sub.add_parser("gen-synth", help="запуск генератора (аргументы — как у него)")
    args, rest = ap.parse_known_args(argv)

    if args.cmd == "gen-synth":
        return subprocess.call([sys.executable, GEN] + rest)
    if args.cmd == "meta":
        return cmd_meta(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    if args.cmd == "hash":
        return cmd_hash(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
