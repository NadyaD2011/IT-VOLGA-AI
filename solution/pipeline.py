"""Пайплайн «каталог изображений → CSV результатов».

Один проход по изображению:
    imread → детектор (YOLO либо contour-фолбэк) → уточнение углов →
    выпрямление (гомография) → нарезка по ГОСТ-геометрии (solution/crops.py) →
    CRNN одним батчем на все части всех знаков → сборка и постобработка по
    маскам ГОСТ → типизация (цвет/геометрия/класс детектора) → строки CSV.

Формат выхода (раздел 5 задания): image;plate_num;plate_type;confidence
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from . import plate_format as pf
from .crops import crop_boxes, assemble
from .rectify import rectify, order_quad_clockwise_tl
from .typer import analyze_crop, decide

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Row:
    image: str
    plate_num: str
    plate_type: str
    confidence: float


@dataclass
class Stats:
    images: int = 0
    detections: int = 0
    emitted: int = 0
    total_ms: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def avg_ms(self):
        return self.total_ms / max(1, self.images)


class PlateReader:
    def __init__(self, cfg: dict, detector=None, recognizer=None):
        self.cfg = cfg
        from .detector import make_detector
        from .recognizer import Recognizer

        self.detector = detector if detector is not None else make_detector(cfg)
        rec_cfg = cfg.get("recognition", {})
        if recognizer is not None:
            self.recognizer = recognizer
        else:
            weights = rec_cfg.get("weights")
            have = bool(weights) and os.path.isfile(weights)
            if weights and not have:
                import warnings
                warnings.warn(f"веса OCR не найдены ({weights}) — номера будут "
                              f"'#'; обучите: python training/train_rec.py")
            self.recognizer = Recognizer(
                weights=weights if have else None,
                device=rec_cfg.get("device", cfg.get("models", {}).get("device", "auto")),
                img_h=rec_cfg.get("img_height", 32),
                img_w=rec_cfg.get("img_width", 160),
                charset=rec_cfg.get("charset"),
                char_prob_min=rec_cfg.get("char_prob_min", 0.55),
            )
        post = cfg.get("post", {})
        self.conf_min = float(post.get("conf_out_min", 0.12))
        self.output_other = bool(post.get("output_other_rows", False))
        self.allow_hash = bool(post.get("allow_hash_rows", False))
        self.rect_h = int(post.get("rect_h", 52))

    # ------------------------------------------------------------------ #
    def _plan(self, img, det):
        """Квад → выпрямленный кроп → набор кропов-частей по геометрии типа."""
        quad = det.get("quad")
        if quad is None:
            x, y, w, h = det["bbox"]
            quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        quad = order_quad_clockwise_tl(quad)
        crop, (ow, oh) = rectify(img, quad, out_h=self.rect_h)
        feats = analyze_crop(crop)
        cls = det.get("cls", "type1")
        two_line = feats["rows"] == 2 or (cls == "type1a" and feats["aspect"] < 2.4)
        mode = "double" if two_line else "single"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        parts = []
        for (x0, y0, x1, y1), name in crop_boxes(mode, ow, oh):
            x1, y1 = max(x1, x0 + 4), max(y1, y0 + 4)
            parts.append((name, gray[y0:y1, x0:x1]))
        return {"crop": crop, "feats": feats, "mode": mode, "parts": parts}

    # ------------------------------------------------------------------ #
    def read_image(self, path: str, filename: str | None = None):
        """Возвращает (rows, stats) для одного изображения."""
        t0 = time.perf_counter()
        if cv2 is None:
            raise RuntimeError("нужен opencv-python")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"не удалось прочитать изображение: {path}")
        dets = self.detector.detect(img)
        stats = Stats(images=1, detections=len(dets))

        plans = [self._plan(img, det) for det in dets]

        # один баттерейп на все части всех знаков
        flat = []
        for pl in plans:
            pl["slice"] = (len(flat), len(flat) + len(pl["parts"]))
            flat += [p for _n, p in pl["parts"]]
        texts = self.recognizer.recognize(flat) if flat else []

        rows: list[Row] = []
        name = filename or os.path.basename(path)
        for det, pl in zip(dets, plans):
            s0, s1 = pl["slice"]
            part_texts = {pl["parts"][i][0]: texts[s0 + i][0] for i in range(s1 - s0)}
            part_conf = [texts[s0 + i][1] for i in range(s1 - s0)]
            merged = assemble(part_texts, pl["mode"])
            text_conf = float(np.mean(part_conf)) if part_conf else 0.0

            text_by_type = {}
            for t in pf.TARGET_TYPES:
                rep, ok = pf.repair(merged, t)
                text_by_type[t] = (rep if ok else merged, ok)
            ptype = decide(det["cls"], pl["feats"], text_by_type)

            if ptype == "other":
                plate_num, conf = "", 0.30
            else:
                plate_num, ok = pf.repair(merged, ptype)
                unread = plate_num.count(pf.UNKNOWN)
                if not ok or (unread * 2 > len(plate_num) and not self.allow_hash):
                    # строка не легла на маску либо читаемо меньше половины —
                    # выдать её нельзя: только ложная строка в зачёт не пойдёт
                    ptype, plate_num, conf = "other", "", 0.30
                else:
                    conf = float(det["score"]) * (0.40 + 0.60 * text_conf)
                    conf *= 1.0 - 0.55 * unread / max(1, len(plate_num))
                    conf = round(min(1.0, max(0.0, conf)), 3)

            if conf < self.conf_min:
                continue
            if ptype == "other" and not self.output_other:
                continue
            rows.append(Row(name, plate_num, ptype, conf))
        stats.emitted = len(rows)
        stats.total_ms = (time.perf_counter() - t0) * 1000
        return rows, stats

    # ------------------------------------------------------------------ #
    def process_dir(self, images_dir: str, out_csv: str | None = None,
                    recursive: bool = False, limit: int = 0, progress=None):
        if recursive:
            files = []
            for root, _d, fs in os.walk(images_dir):
                files += [os.path.join(root, f) for f in fs
                          if f.lower().endswith(IMAGE_EXTS)]
        else:
            files = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))
                     if f.lower().endswith(IMAGE_EXTS)]
        files = [f for f in sorted(files)][:limit or None]

        agg = Stats()
        all_rows: list[Row] = []
        for i, p in enumerate(files, 1):
            try:
                rows, st = self.read_image(p)
            except Exception as e:  # один битый кадр не роняет прогон
                agg.images += 1
                agg.errors.append(f"{os.path.basename(p)}: {e}")
                continue
            all_rows += rows
            agg.images += st.images
            agg.detections += st.detections
            agg.emitted += st.emitted
            agg.total_ms += st.total_ms
            if progress:
                progress(i, len(files), p, rows)
        if out_csv:
            write_csv(all_rows, out_csv)
        return all_rows, agg


def write_csv(rows, out_csv):
    d = os.path.dirname(os.path.abspath(out_csv))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(["image", "plate_num", "plate_type", "confidence"])
        for r in rows:
            w.writerow([r.image, r.plate_num, r.plate_type, f"{r.confidence:.3f}"])
