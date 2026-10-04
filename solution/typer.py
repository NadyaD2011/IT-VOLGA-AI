"""Финальное решение о типе знака: детектор + геометрия + цвет + влезание в маску.

ГОСТ-ориентированные эвристики (раздел 2 задания):
  * type1  — белая однострочная пластинка 520x112 (aspect ~4.6);
  * type1a — белая двухстрочная 290x170 (aspect ~1.7), две плотные строки текста;
  * type1b — ЖЁЛТАЯ однострочная 520x112;
  * other  — всё, что не подошло (иностранные, прицепы, мотоциклы, «щиты»...).

Цвет — самый надёжный признак для 1Б (жёлтый фон встречается у 1Б практически
всегда), геометрия/строки — для 1А. Класс детектора — сильный аргумент, когда
веса обучены. Итог — взвешенное голосование; при конфликте предпочтение тому
типу, под маску ГОСТ которого лучше лёг распознанный текст.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def analyze_crop(crop_bgr: np.ndarray) -> dict:
    """Признаки выпрямленного кропа знака (до OCR)."""
    f = {"yellow_frac": 0.0, "white_frac": 0.0, "blue_frac": 0.0,
         "rows": 1, "aspect": 1.0, "text_std": 0.0, "dark_frac": 0.0}
    if cv2 is None or crop_bgr is None or crop_bgr.size == 0:
        return f
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f["yellow_frac"] = float(np.mean((H >= 14) & (H <= 40) & (S > 90) & (V > 90)))
    f["white_frac"] = float(np.mean((S < 70) & (V > 130)))
    f["blue_frac"] = float(np.mean((H > 90) & (H < 130) & (S > 90) & (V > 60)))
    f["dark_frac"] = float(np.mean(V < 60))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    f["text_std"] = float(gray.std())
    f["aspect"] = w / max(h, 1)
    # две строки? — профиль тёмных (черные символы) по горизонтали
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 7)
    prof = thr.mean(axis=1) / 255.0
    mid = int(h * 0.35), int(h * 0.65)
    top = prof[: mid[0]].mean() if mid[0] > 0 else 0.0
    bot = prof[mid[1]:].mean() if mid[1] < h else 0.0
    midv = prof[mid[0]: mid[1]].mean() if h > 4 else 0.0
    if midv < 0.22 * max(top, bot, 1e-6) and top > 0.04 and bot > 0.04 and f["aspect"] < 2.4:
        f["rows"] = 2
    return f


def decide(det_cls: str, feats: dict, text_by_type: dict) -> str:
    """text_by_type: {type: (text, fits_bool)} для целевых типов. Возвращает тип."""
    votes = {t: 0.0 for t in ("type1", "type1a", "type1b", "other")}
    votes[det_cls] = votes.get(det_cls, 0.0) + 1.0

    if feats.get("yellow_frac", 0) > 0.45 and feats.get("rows", 1) == 1:
        votes["type1b"] += 1.2
    if feats.get("rows") == 2:
        votes["type1a"] += 1.1
    elif feats.get("aspect", 0) > 2.6 and feats.get("yellow_frac", 0) <= 0.45:
        votes["type1"] += 0.4
    if feats.get("blue_frac", 0) > 0.22 and feats.get("white_frac", 0) < 0.35:
        votes["other"] += 1.0          # EU-полоса / синева иностранного знака
    if feats.get("yellow_frac", 0) > 0.45 and feats.get("rows") == 2:
        votes["type1a"] += 0.3          # жёлтый квадрат в РФ редок, но не other

    # маска ГОСТ: тип, под который идеально лёг текст, получает большой вес
    for t, (_txt, fits) in text_by_type.items():
        if fits:
            votes[t] += 1.4
        else:
            votes[t] -= 0.9

    best = max(votes, key=votes.get)
    # вето по цвету: сине-белый «флаг» (транзит) / широкие синие полосы (ЕС,
    # дипломатические) / чёрный фон (ВС РФ, «чёрные» номера) — вне области
    if feats.get("blue_frac", 0) > 0.34 and feats.get("yellow_frac", 0) < 0.45 \
            and feats.get("white_frac", 1) < 0.45:
        best = "other"
    if feats.get("dark_frac", 0) > 0.62 and feats.get("yellow_frac", 0) < 0.3:
        best = "other"
    if best == "type1b" and text_by_type.get("type1b") and not text_by_type["type1b"][1] \
            and text_by_type.get("type1") and text_by_type["type1"][1] and feats.get("yellow_frac", 0) < 0.6:
        best = "type1"  # жёлтый «выцветший», а текст явно ЛЦЦЦЛЛ -> не такси
    return best
