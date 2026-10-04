"""CLI решения: python -m solution.cli --images КАТАЛОГ --out result.csv

Способы передачи пути (раздел 4.3 задания): аргументы командной строки,
переменные окружения GRZ_IMAGES_DIR / GRZ_OUT_CSV / GRZ_CONFIG, конфигуационный
файл config.yaml. Приоритет: аргументы > переменные окружения > config.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> dict:
    cfg_path = path or os.environ.get("GRZ_CONFIG") or os.path.join(PROJECT_ROOT, "config.yaml")
    cfg = {}
    if os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except ImportError:
            cfg = _parse_minimal_yaml(cfg_path)
    # env overrides
    run = cfg.setdefault("run", {})
    if os.environ.get("GRZ_IMAGES_DIR"):
        run["images"] = os.environ["GRZ_IMAGES_DIR"]
    if os.environ.get("GRZ_OUT_CSV"):
        run["out"] = os.environ["GRZ_OUT_CSV"]
    dev = os.environ.get("GRZ_DEVICE")
    if dev:
        cfg.setdefault("models", {})["device"] = dev
        cfg.setdefault("recognition", {})["device"] = dev
    # относительные пути весов — от корня проекта
    for sec, key in (("models", "detector_weights"), ("recognition", "weights")):
        v = cfg.get(sec, {}).get(key)
        if v and not os.path.isabs(v) and not os.path.isfile(v):
            cand = os.path.join(PROJECT_ROOT, v)
            if os.path.isfile(cand):
                cfg[sec][key] = cand
            elif sec == "recognition":
                cfg[sec][key] = None
            else:
                cfg[sec][key] = None
    return cfg


def _parse_minimal_yaml(path):
    """Микроразборщик key: value / вложенность по отступам — на случай, если
    PyYAML не установлен (полный конфиг всё же требует PyYAML)."""
    root, stack = {}, [(0, None)]
    cur = root
    stack = [(-1, root)]
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else root
            key, _, val = line.strip().partition(":")
            val = val.strip()
            if val == "":
                d = {}
                parent[key] = d
                stack.append((indent, d))
            else:
                if val.startswith("[") and val.endswith("]"):
                    parent[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
                else:
                    v = val.strip("'\"")
                    if v.lower() in ("true", "false"):
                        parent[key] = v.lower() == "true"
                    else:
                        try:
                            parent[key] = float(v) if "." in v else int(v)
                        except ValueError:
                            parent[key] = v
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Распознавание нестандартных ГРЗ (Волга-IT 2026)")
    ap.add_argument("--images", help="каталог с изображениями (.jpg/.png)")
    ap.add_argument("--out", help="куда писать CSV (по умолчанию result.csv)")
    ap.add_argument("--config", help="путь к config.yaml")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"),
                    help="форсировать устройство")
    ap.add_argument("--imgsz", type=int, help="размер входа детектора")
    ap.add_argument("--conf", type=float, help="порог детектора")
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N кадров")
    ap.add_argument("--recursive", action="store_true", help="рекурсивно по подкаталогам")
    ap.add_argument("--bench", action="store_true", help="печатать время на кадр и итоговое среднее")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = cfg.setdefault("run", {})
    images = args.images or run.get("images")
    out = args.out or run.get("out", "result.csv")
    if not images or not os.path.isdir(images):
        print(f"ОШИБКА: каталог изображений не задан или не найден: {images!r}\n"
              f"Укажите --images DIR, или GRZ_IMAGES_DIR, или run.images в config.yaml.",
              file=sys.stderr)
        return 2
    if args.device:
        cfg.setdefault("models", {})["device"] = args.device
        cfg.setdefault("recognition", {})["device"] = args.device
    if args.imgsz:
        cfg.setdefault("models", {})["imgsz"] = args.imgsz
    if args.conf is not None:
        cfg.setdefault("models", {})["conf"] = args.conf

    from .pipeline import PlateReader
    reader = PlateReader(cfg)

    t_all = time.perf_counter()
    def prog(i, n, p, rows):
        if args.quiet:
            return
        tail = ", ".join(f"{r.plate_num or '—'}/{r.plate_type}@{r.confidence:.2f}" for r in rows[:3])
        print(f"[{i}/{n}] {os.path.basename(p):36s} -> {len(rows)} знак(ов): {tail}")

    rows, stats = reader.process_dir(images, out_csv=out, recursive=args.recursive,
                                     limit=args.limit, progress=prog)
    dt = time.perf_counter() - t_all
    if not args.quiet:
        print(f"\nИзображений: {stats.images}, знаков найдено: {stats.detections}, "
              f"строк в CSV: {stats.emitted} -> {out}")
        print(f"Среднее время на кадр: {stats.avg_ms:.1f} мс (лимит задания 100 мс); "
              f"всего {dt:.1f} с")
    if stats.errors and not args.quiet:
        for e in stats.errors[:10]:
            print(f"  ПРОПУСК: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
