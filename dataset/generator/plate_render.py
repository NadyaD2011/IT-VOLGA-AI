"""Рендеринг синтетических знаков ГРЗ по геометрии ГОСТ Р 50577-2018.

Только Pillow + стандартная библиотека (без opencv/torch) — генератор должен
запускаться где угодно и быть воспроизводимым по seed.

ВАЖНО: словарь GEOM ниже продублирован из solution/crops.py. Пайплайн
распознавания и обучение режут знак по одним и тем же доля — при изменении
одного файла синхронизируйте второй (есть проверка в tests/selftest.py).
"""
from __future__ import annotations

import glob
import os

from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"

GEOM = {
    "single": {
        "body":   {"box": (0.030, 0.745, 0.10, 0.92), "expected": 6},
        "region": {"box": (0.760, 0.990, 0.05, 0.62), "expected": 3},
        "flag":   {"box": (0.760, 0.990, 0.64, 0.97), "expected": 0},
    },
    "double": {
        "top":    {"box": (0.020, 0.980, 0.045, 0.420), "expected": 4},
        "series": {"box": (0.020, 0.450, 0.455, 0.680), "expected": 2},
        "region": {"box": (0.460, 0.990, 0.455, 0.680), "expected": 3},
        "flag":   {"box": (0.700, 0.990, 0.700, 0.980), "expected": 0},
    },
}

# канонические размеры знака, мм -> пиксели канвы (x SCALE)
SIZE_MM = {"type1": (520, 112), "type1b": (520, 112), "type1a": (290, 170)}
BG_COLOR = {"type1": (240, 240, 238), "type1a": (240, 240, 238),
            "type1b": (252, 204, 8), "other": (238, 238, 232)}
FG_COLOR = (18, 18, 18)

_FONT_CACHE = {}


def find_font(prefer: str | None = None):
    """Ищем шрифт: env GRZ_FONT -> prefer -> каталог generator/fonts -> системные DejaVu."""
    cands = []
    if prefer:
        cands.append(prefer)
    env = os.environ.get("GRZ_FONT")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += sorted(glob.glob(os.path.join(here, "fonts", "*.ttf")))
    cands += sorted(glob.glob(os.path.join(here, "..", "fonts", "*.ttf")))
    cands += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        if path:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        else:
            try:
                _FONT_CACHE[key] = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1
                _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _place(draw, text, box, W, H, font, anchor_center=True, right=False):
    """Раскладывает символы текста равномерно по box=(x0,x1,y0,y1) (доли)."""
    x0, x1, y0, y1 = (box[0] * W, box[1] * W, box[2] * H, box[3] * H)
    chars = list(text)
    n = len(chars)
    if n == 0:
        return
    slot = (x1 - x0) / n
    for i, ch in enumerate(chars):
        bb = draw.textbbox((0, 0), ch, font=font)
        cw, chh = bb[2] - bb[0], bb[3] - bb[1]
        cx = x0 + slot * i + (slot - cw) / 2 - bb[0]
        cy = (y0 + y1) / 2 - chh / 2 - bb[1]
        draw.text((cx, cy), ch, font=font, fill=FG_COLOR)


def draw_flag_and_rus(draw, box, W, H):
    """Триколор + «RUS» (как на реальных знаках, в правом нижнем блоке)."""
    x0, x1, y0, y1 = box[0] * W, box[1] * W, box[2] * H, box[3] * H
    bh = max(6.0, y1 - y0)
    flag_h = bh * 0.52
    flag_w = min(x1 - x0, flag_h * 1.5)
    fx = x0 + (x1 - x0 - flag_w) / 2
    stripes = [(245, 245, 245), (40, 80, 200), (200, 40, 40)]
    seg = max(1.0, flag_h / 3)
    for k, col in enumerate(stripes):
        ya = int(y0 + k * seg)
        yb = int(min(y0 + (k + 1) * seg, y0 + flag_h))
        draw.rectangle([int(fx), ya, int(fx + flag_w), max(ya + 1, yb)],
                       fill=col, outline=(130, 130, 130), width=1)
    try:
        fnt = get_font(FONT_PATH, max(7, int(bh * 0.26)))
    except Exception:
        fnt = None
    txt = "RUS"
    bb = draw.textbbox((0, 0), txt, font=fnt)
    draw.text(((x0 + x1) / 2 - (bb[2] - bb[0]) / 2 - bb[0],
               y0 + flag_h + (bh - flag_h) / 2 - (bb[3] - bb[1]) / 2 - bb[1]),
              txt, font=fnt, fill=FG_COLOR)


FONT_PATH = find_font()


def render_plate(text: str, ptype: str, scale: int = 2, font_path: str | None = None,
                 corner_radius: int | None = None, other_style: str | None = None) -> Image.Image:
    """Синтетический знак. text — полный номер вида 'A123BC125' (type1/1a) или
    'AB12399' (type1b). 'other' — произвольная строка; other_style:
    eu | mil (чёрный) | transit | moto.
    """
    fp = font_path or FONT_PATH or find_font()
    if ptype == "other":
        return render_foreign(text, scale, fp, style=other_style or "eu")
    w_mm, h_mm = SIZE_MM[ptype]
    W, H = w_mm * scale, h_mm * scale
    img = Image.new("RGB", (W, H), BG_COLOR.get(ptype, (240, 240, 238)))
    draw = ImageDraw.Draw(img)
    rad = corner_radius if corner_radius is not None else max(2, H // 14)
    # рамка знака
    draw.rounded_rectangle([1, 1, W - 2, H - 2], radius=rad,
                            outline=FG_COLOR, width=max(2, H // 45))
    geom = GEOM["double" if ptype == "type1a" else "single"]

    if ptype == "type1a":
        fnt = get_font(fp, int(H * 0.30))
        _place(draw, text[:4], geom["top"]["box"], W, H, fnt)
        _place(draw, text[4:6], geom["series"]["box"], W, H, fnt)
        _place(draw, text[6:], geom["region"]["box"], W, H, fnt)
        draw_flag_and_rus(draw, geom["flag"]["box"], W, H)
    else:
        body_len = 5 if ptype == "type1b" else 6
        fnt = get_font(fp, int(H * 0.60))
        _place(draw, text[:body_len], geom["body"]["box"], W, H, fnt)
        _place(draw, text[body_len:], geom["region"]["box"], W, H,
               get_font(fp, int(H * 0.42)))
        draw_flag_and_rus(draw, geom["flag"]["box"], W, H)
    return img


def render_foreign(text: str, scale: int = 2, font_path: str | None = None,
                   style: str = "eu") -> Image.Image:
    """Знаки 'other': eu (полосы со звёздами), mil (чёрный фон/белый текст),
    transit (сине-бело-синие полосы), moto (квадратный 190x145)."""
    fp = font_path or FONT_PATH or find_font()
    if style == "moto":
        W, H = int(190 * scale * 0.7), int(145 * scale * 0.7)
    else:
        W, H = int(250 * scale * 1.15), int(56 * scale * 1.15)
    bg = {"eu": (245, 245, 245), "mil": (16, 16, 18), "transit": (40, 70, 180),
          "moto": (245, 245, 245)}[style]
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fg = (235, 235, 235) if style == "mil" else (20, 20, 20)
    d.rectangle([0, 0, W - 1, H - 1], outline=(70, 70, 70), width=max(2, scale))
    if style == "transit":
        h3 = H // 3
        d.rectangle([2, h3, W - 3, 2 * h3], fill=(245, 245, 245))
    if style == "eu":
        band = int(H * 0.30)
        for x in (2, W - band - 2):
            d.rectangle([x, 2, x + band, H - 3], fill=(24, 60, 160))
            d.ellipse([x + band * 0.30, H * 0.22, x + band * 0.70, H * 0.62],
                      outline=(240, 220, 90), width=max(1, scale))
    fnt = get_font(fp, int(H * (0.52 if style != "moto" else 0.34)))
    txt = (text or "AA123BB").upper()
    if style == "moto":
        mid = H // 2
        bb1 = d.textbbox((0, 0), txt[:4], font=fnt)
        d.text(((W - bb1[2] + bb1[0]) / 2, mid * 0.15), txt[:4], font=fnt, fill=fg)
        bb2 = d.textbbox((0, 0), txt[4:], font=fnt)
        d.text(((W - bb2[2] + bb2[0]) / 2, mid * 1.05), txt[4:], font=fnt, fill=fg)
    else:
        body = txt
        bb = d.textbbox((0, 0), body, font=fnt)
        x0 = int(H * 0.5) if style == "eu" else 0
        d.text(((W - bb[2] + bb[0] + x0) / 2, (H - bb[3] + bb[1]) / 2 - bb[1]),
               body, font=fnt, fill=fg)
    return img
