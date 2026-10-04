"""Процедурные фоны для синтетики (без внешних картинок — лицензия чистая).

Каждая функция возвращает np.ndarray uint8 (H, W, 3) RGB и подсказку, куда
вставить знак (центр в относительных координатах).

rng — обёртка Rng над np.random.default_rng (детерминированной); методы
uniform/normal/random/choice/int/randint совместимы с random.Random.
"""
from __future__ import annotations

import math

import numpy as np


class Rng:
    """Единый совместимый API для всех модулей генератора."""

    def __init__(self, seed):
        self.g = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def random(self):
        return float(self.g.random())

    def uniform(self, a, b):
        return float(self.g.uniform(a, b))

    def normal(self, mu, sigma, size=None):
        return self.g.normal(mu, sigma, size)

    def randint(self, a, b):
        return int(self.g.integers(a, b + 1))

    def choice(self, seq):
        return seq[int(self.g.integers(0, len(seq)))]

    def gauss(self, mu, sigma):
        return float(self.g.normal(mu, sigma))


def _noise(bg, rng, sigma=6.0):
    bg += np.clip(rng.normal(0, sigma, bg.shape), -40, 40)
    return bg


def asphalt(w, h, rng):
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    base = 96 + 26 * (x * 0 + y)  # серый градиент
    bg = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        bg[..., c] = base + rng.normal(0, 9, (h, w))
    # трещины/пятна
    for _ in range(rng.randint(3, 8)):
        cx, cy = rng.random() * w, rng.random() * h
        r = rng.uniform(0.04, 0.22) * max(w, h)
        yy, xx = np.ogrid[:h, :w]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        bg[m] *= rng.uniform(0.72, 0.9)
    # разметка
    if rng.random() < 0.6:
        ly = rng.uniform(0.65, 0.9) * h
        bg[int(ly):int(ly + max(2, h * 0.012)), :] = 190
    return np.clip(bg, 0, 255).astype(np.uint8), (0.5, 0.5)


def grass_bokeh(w, h, rng):
    hue = rng.uniform(0.28, 0.45)
    import colorsys
    r0, g0, b0 = colorsys.hsv_to_rgb(hue, rng.uniform(0.35, 0.6), rng.uniform(0.35, 0.7))
    bg = np.zeros((h, w, 3), np.float32)
    bg[..., 0], bg[..., 1], bg[..., 2] = r0 * 255, g0 * 255, b0 * 255
    for _ in range(rng.randint(18, 45)):
        cx, cy = rng.random() * w, rng.random() * h
        r = rng.uniform(0.015, 0.09) * max(w, h)
        bright = rng.uniform(0.7, 1.5)
        yy, xx = np.ogrid[:h, :w]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        bg[m] = np.clip(bg[m] * bright, 0, 255)
    bg += rng.normal(0, 7, bg.shape)
    # размытие box-фильтром через интегральное изображение (без cv2)
    k = rng.choice([3, 5, 7])
    kk = 2 * k + 1
    pad = np.pad(bg.astype(np.float64), ((k, k), (k, k), (0, 0)), mode="edge")
    s = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1, 3), np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    bg = (s[kk:, kk:] - s[:-kk, kk:] - s[kk:, :-kk] + s[:-kk, :-kk]) / (kk * kk)
    return np.clip(bg, 0, 255).astype(np.uint8), (0.5, rng.uniform(0.45, 0.65))


CAR_COLORS = [(150, 152, 158), (225, 224, 220), (30, 36, 46), (120, 18, 24),
              (28, 52, 96), (90, 96, 104), (188, 170, 96), (60, 84, 60)]


def car_rear(w, h, rng):
    """Параметрическая «кормовая ниша»: кузов, бампер, ниша под знак, фонари."""
    body = np.array(rng.choice(CAR_COLORS), np.float32)
    bg = np.zeros((h, w, 3), np.float32)
    y = np.linspace(0, 1, h)[:, None]
    shade = (1.18 - 0.42 * y)  # блисk сверху, тень снизу
    bg += (body[None, None, :] * shade[..., None]).astype(np.float32)

    # задняя часть стекла/крыша
    roof_h = int(h * rng.uniform(0.06, 0.16))
    bg[:roof_h] *= 0.42
    # ниша под знак
    cx, cy = 0.5 * w, h * rng.uniform(0.52, 0.66)
    niche_w, niche_h = w * rng.uniform(0.36, 0.52), h * rng.uniform(0.13, 0.20)
    x0, x1 = int(cx - niche_w / 2), int(cx + niche_w / 2)
    y0, y1 = int(cy - niche_h / 2), int(cy + niche_h / 2)
    bg[y0:y1, x0:x1] *= 0.82  # затемнённая ниша
    bg[y0:y0 + 3, x0:x1] *= 0.7  # верхняя кромка тени
    # фонари
    for sgn in (-1, 1):
        tw, th = w * rng.uniform(0.06, 0.12), h * rng.uniform(0.05, 0.10)
        tx = int(cx + sgn * (niche_w / 2 + tw * rng.uniform(0.6, 1.6)) - tw / 2)
        ty = int(cy - th / 2 - h * 0.02)
        tx0, tx1 = max(0, min(w - 2, tx)), max(2, min(w, tx + int(tw)))
        ty0, ty1 = max(0, ty), min(h, ty + int(th))
        if tx1 - tx0 > 6 and ty1 - ty0 > 4:
            glow = np.array([196, 26, 18], np.float32)
            bg[ty0:ty1, tx0:tx1] = 0.55 * bg[ty0:ty1, tx0:tx1] + 0.45 * glow
    # номерной выступ
    bg[y0 + 2:y1 - 2, x0 + 2:x1 - 2] *= 1.10
    bg += rng.normal(0, 5.5, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8), (cx / w, cy / h)


def billboard(w, h, rng, screen=False):
    """Щит рекламы / экран витрины (негативные is_vehicle=0 кадры)."""
    if not screen:
        g = np.linspace(1.25, 0.65, h)[:, None]
        bg = np.dstack([np.full((h, w), 120.0) * g,
                        np.full((h, w), 150.0) * g,
                        np.full((h, w), 180.0) * g])
        bx, by, bw2, bh2 = 0.08, 0.10, 0.84, 0.62
        x0, y0 = int(bx * w), int(by * h)
        x1, y1 = int((bx + bw2) * w), int((by + bh2) * h)
        bg[y0:y1, x0:x1] = rng.choice([236, 210, 176])
        bg[y1:y1 + int(0.25 * h), x0:x1] *= 0.55  # опоры
        for _ in range(6):  # «рекламные» цветные полосы
            xx = int(rng.uniform(x0 + 8, x1 - 40))
            yy = int(rng.uniform(y0 + 8, y1 - 30))
            ww = int(rng.uniform(20, (x1 - x0) * 0.4))
            hh = int(rng.uniform(8, (y1 - y0) * 0.25))
            bg[yy:yy + hh, xx:xx + ww] = [rng.randint(40, 230) for _ in range(3)]  # Rng API
    else:
        bg = np.full((h, w, 3), 26.0)
        x0, y0, x1, y1 = int(0.12 * w), int(0.18 * h), int(0.88 * w), int(0.80 * h)
        bg[y0:y1, x0:x1] = rng.choice([200, 224, 236])
        for yy in range(y0, y1, 3):
            bg[yy] *= rng.uniform(0.86, 1.0)
    bg += rng.normal(0, 5, bg.shape)
    cx, cy = rng.uniform(0.4, 0.6), rng.uniform(0.35, 0.6)
    return np.clip(bg, 0, 255).astype(np.uint8), (cx, cy)
