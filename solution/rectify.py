"""Выпрямление (гомография) знака по четырём углам.

Квад хранится как 4 точки [x, y] в пиксельных координатах кадра,
по часовой стрелке от левого верхнего угла — ровно так, как требует
валидатор датасета (раздел 6.2 задания).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # cv2 обязателен для инференса, но не для юнит-тестов формата
    cv2 = None


def quad_area(quad) -> float:
    """Знаковая площадь шнурками; в экранных координатах (y вниз)
    положительная = по часовой стрелке (как в валидаторе)."""
    pts = np.asarray(quad, dtype=float).reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def order_quad_clockwise_tl(points) -> np.ndarray:
    """Приводит 4 точки к порядку: левый верхний, правый верхний, правый
    нижний, левый нижний (по часовой в экранных координатах)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError(f"ожидалось 4 точки, получено {len(pts)}")
    center = pts.mean(axis=0)
    # первый — минимальная сумма x+y (левый верхний, так же проверяет валидатор)
    first = int(np.argmin(pts.sum(axis=1)))
    pts = np.roll(pts, -first, axis=0)
    if quad_area(pts) < 0:  # против часовой -> разворачиваем остальные
        pts = np.concatenate([pts[:1], pts[:0:-1]], axis=0)
    return pts


def quad_bbox(quad):
    pts = np.asarray(quad, dtype=float).reshape(4, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2 - x1)), int(np.ceil(y2 - y1))


def quad_size(quad):
    """(w, h) по длинам сторон квада — устойчивее к повороту, чем bbox."""
    pts = np.asarray(quad, dtype=float).reshape(4, 2)
    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
    h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0
    return float(w), float(h)


def rectify(img, quad, out_h: int = 48, out_w: int | None = None,
            max_aspect: float = 6.0, min_aspect: float = 1.0):
    """Возвращает выпрямленный кроп знака (BGR np.ndarray) и (w, h) результата."""
    if cv2 is None:
        raise RuntimeError("для выпрямления нужен opencv-python (cv2)")
    pts = order_quad_clockwise_tl(quad).astype(np.float32)
    w, h = quad_size(pts)
    if out_w is None:
        aspect = float(np.clip(w / max(h, 1e-6), min_aspect, max_aspect))
        out_w = int(round(out_h * aspect))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    H = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return crop, (out_w, out_h)


def crop_quad_fullres(img, quad, pad: float = 0.02):
    """Ориентированный кроп «как есть» (без нормализации размера) — используется
    для аналитики (цвет, строки) до выпрямления OCR-входа."""
    return rectify(img, quad, out_h=max(24, int(quad_size(quad)[1])),
                   out_w=max(24, int(quad_size(quad)[0])))


def homography_from_quads(src, dst):
    """Обучающая утилита генератора: H, переводящая src-прямоугольник в dst-квад."""
    if cv2 is None:
        raise RuntimeError("нужен opencv")
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
