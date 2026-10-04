"""Детектор знаков.

Два бэкенда:

  UltralyticsBackend — YOLOv8n/yolov11n, дообученный на 4 класса
  (type1, type1a, type1b, other). Основной путь; веса `solution/models/detector.pt`.
  Даёт bbox; углы знака достраиваются классическим уточнением (refine_quad).

  ContourBackend — автономный cv2-детектор (белая/жёлтая маска + контуры).
  Не требует обучения: используется как фолбэк «из коробки», в тестах и для
  генерации слабыхpseudo-меток при предразметке реальных фото
  (tools/label_assist.py) — финальную разметку человек всегда подтверждает.

Оба бэкенда возвращают список словарей:
  {"bbox": [x, y, w, h], "quad": np.ndarray(4,2) | None,
   "cls": "type1|type1a|type1b|other", "score": float}
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# --------------------------------------------------------------------------- #
#  Уточнение углов по цветовой сегментации внутри bbox
# --------------------------------------------------------------------------- #
def refine_quad(img, bbox, pad: int = 4):
    """Возвращает quad (4x2, порядок TL->TR->BR->BL) или None."""
    if cv2 is None:
        return None
    H, W = img.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 110), (180, 90, 255))
    yellow = cv2.inRange(hsv, (14, 70, 110), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.35 * max(1, w * h):
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # пересечение rect с bbox не должно «убегать» далеко от знака
    x2, y2, bw2, bh2 = cv2.boundingRect(c)
    if bw2 < w * 0.6 or bh2 < h * 0.6:
        return None
    from .rectify import order_quad_clockwise_tl
    quad = order_quad_clockwise_tl(box)
    quad[:, 0] += x0
    quad[:, 1] += y0
    quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
    wq, hq = quad_size(quad)
    if hq <= 0 or not (0.8 <= wq / hq <= 6.5):
        return None
    return quad


def quad_size(quad):
    pts = np.asarray(quad, float).reshape(4, 2)
    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
    h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
    return float(w), float(h)


# --------------------------------------------------------------------------- #
#  YOLO (Ultralytics)
# --------------------------------------------------------------------------- #
class UltralyticsBackend:
    NAMES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.30,
                 iou: float = 0.45, device: str = "auto", half: bool = False,
                 refine: bool = True):
        if cv2 is None:
            raise RuntimeError("нужен opencv-python")
        from ultralytics import YOLO  # локальный импорт: ultralytics опционален
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.refine = refine
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            half = half and device != "cpu"
        self.device, self.half = device, half

    def detect(self, img_bgr: np.ndarray):
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                  verbose=False)
        if self.half:
            kw["half"] = True  # в новых ultralytics half передаём только на GPU
        res = self.model.predict(img_bgr, **kw)[0]
        out = []
        names = res.names
        for b in res.boxes:
            cls = int(b.cls[0])
            name = names.get(cls) if isinstance(names, dict) else None
            if name not in self.NAMES.values():
                name = self.NAMES.get(cls, "type1")
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bbox = [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
            quad = refine_quad(img_bgr, bbox) if self.refine else None
            if quad is None:
                x, y, w, h = bbox
                quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            out.append({"bbox": [int(v) for v in bbox], "quad": quad,
                        "cls": name, "score": float(b.conf[0])})
        return out


# --------------------------------------------------------------------------- #
#  Классический фолбэк-детектор (без нейросетей)
# --------------------------------------------------------------------------- #
class ContourBackend:
    """Находит светлые (белые) и жёлтые прямоугольные области, похожие на ГРЗ.

    Качество ниже YOLO, но решение остаётся работоспособным без обучения
    (демо/автотесты) и покрывает «закрытую» часть (other) — слишком
    узкие/широкие или тёмные отбрасываются.
    """

    def __init__(self, min_area: int = 320, max_area_frac: float = 0.55,
                 work_long: int = 1280, conf: float = 0.0):
        if cv2 is None:
            raise RuntimeError("нужен opencv-python (opencv-python-headless)")
        self.min_area, self.max_area_frac = min_area, max_area_frac
        self.work_long, self.conf = work_long, conf

    def detect(self, img_bgr: np.ndarray):
        H0, W0 = img_bgr.shape[:2]
        scale = min(1.0, self.work_long / max(H0, W0))
        img = cv2.resize(img_bgr, None, fx=scale, fy=scale) if scale < 1 else img_bgr.copy()
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # «белая пластинка»: низкая насыщенность и заметно светлее окружения;
        # для тёмных сцен добавляется локальная контрастность
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.5, (8, 8))
        bright = clahe.apply(gray)
        v_ok = (hsv[..., 2] > 115) | ((bright.astype(np.int16) - gray.astype(np.int16)) > 28)
        white = (v_ok & (hsv[..., 1] < 95)).astype(np.uint8) * 255
        yellow = cv2.inRange(hsv, (13, 60, 95), (45, 255, 255))
        yellow = cv2.bitwise_and(yellow, (hsv[..., 2] > 90).astype(np.uint8) * 255)
        mask = cv2.bitwise_or(white, yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area_frac * H * W:
                continue
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            if h <= 4 or w <= 4:
                continue
            fill = area / float(w * h)
            if fill < 0.62:  # буква-«дырка» или ветка за номером
                continue
            ar = w / h
            two_rows = None
            if 1.1 <= ar <= 2.3:
                two_rows = True
            elif 2.3 < ar <= 6.5:
                two_rows = False
            else:
                continue
            # «текст» внутри: среднеквадратичное отклонение яркости
            inner = cv2.cvtColor(img[max(0, y + 2):y + h - 2, max(0, x + 2):x + w - 2], cv2.COLOR_BGR2GRAY)
            if inner.size == 0 or float(inner.std()) < 18:
                continue
            # жёлтость
            ycount = int(cv2.countNonZero(yellow[max(0, y):y + h, max(0, x):x + w]))
            yellow_frac = ycount / max(1, w * h)
            if yellow_frac > 0.5:
                cls = "type1b"
            elif two_rows:
                cls = "type1a"
            else:
                cls = "type1"
            score = 0.30 + 0.30 * fill + 0.20 * min(1.0, area / (0.02 * H * W))
            score = float(min(0.72, score))
            if score < self.conf:
                continue
            box = order4(cv2.boxPoints(cv2.minAreaRect(c)), scale)
            cands.append({"bbox": [int(x / scale), int(y / scale), int(w / scale), int(h / scale)],
                          "quad": box, "cls": cls, "score": score,
                          "_yellow": yellow_frac})
        cands = nms(cands, 0.35)
        return cands[:30]


def order4(box, scale):
    from .rectify import order_quad_clockwise_tl
    q = order_quad_clockwise_tl(np.asarray(box, float) / scale)
    return q


def nms(dets, iou_thr):
    if not dets:
        return []
    boxes = np.array([d["bbox"] for d in dets], float)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort([-d["score"] for d in dets])
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return [dets[i] for i in keep]


def make_detector(cfg: dict, prefer_trained: bool = True):
    """Фабрика по config: backend auto|ultralytics|contour."""
    weights = cfg.get("models", {}).get("detector_weights", "")
    backend = cfg.get("models", {}).get("backend", "auto")
    import os
    have_w = weights and os.path.isfile(weights)
    if backend in ("auto", "ultralytics") and have_w and prefer_trained:
        m = cfg.get("models", {})
        return UltralyticsBackend(weights, imgsz=m.get("imgsz", 640),
                                  conf=m.get("conf", 0.30), iou=m.get("iou", 0.45),
                                  device=m.get("device", "auto"), half=m.get("half", False))
    if backend == "ultralytics" and not have_w:
        raise FileNotFoundError(f"backend=ultralytics, но веса не найдены: {weights}")
    if backend == "auto" and not have_w:
        import warnings
        warnings.warn("solution/models/detector.pt не найден — используется "
                      "классический contour-бэкенд (демо-режим). Обучите детектор: "
                      "python training/train_detector.py")
    return ContourBackend()
