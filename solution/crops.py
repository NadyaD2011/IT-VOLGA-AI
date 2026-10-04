"""Геометрическая модель строки знака и план нарезки кропов под OCR.

Один и тот же набор долей (всё в долях ширины/высоты ВЫПРЯМЛЕННОГО знака)
используют:
  * dataset/generator/plate_render.py — рисует синтетические знаки;
  * solution/pipeline.py             — режет реальные знаки на части перед CRNN;
  * training/build_rec_dataset.py    — режет датасет для обучения CRNN.
Значения продублированы в plate_render.py (датасет самодостаточен); тест
tests/selftest.py сверяет их.

ГОСТ Р 50577-2018: у типа 1/1Б справа колонка «код региона, под ним флаг и RUS»;
у типа 1А сверху строка «Л ЦЦЦ», снизу слева «ЛЛ», снизу справа код региона,
под ним флаг+RUS.

Знак, который не влез в маску выбранного типа, понижается до other — см. typer.
"""
from __future__ import annotations

# part -> (x0, x1, y0, y1) доли кадра; expected — ожидаемая длина текста части
GEOM = {
    "single": {  # type1 и type1b: одна строка + правая колонка региона
        "body":   {"box": (0.030, 0.745, 0.10, 0.92), "expected": 6},
        "region": {"box": (0.760, 0.990, 0.05, 0.62), "expected": 3},
    },
    "double": {  # type1a: две строки
        "top":     {"box": (0.020, 0.980, 0.045, 0.420), "expected": 4},
        "series":  {"box": (0.020, 0.450, 0.455, 0.680), "expected": 2},
        "region":  {"box": (0.460, 0.990, 0.455, 0.680), "expected": 3},
    },
}
PLAN_ORDER = {"single": ["body", "region"], "double": ["top", "series", "region"]}


def crop_boxes(mode: str, w: int, h: int):
    """Пиксельные рамки частей знака для кропа размера (w, h)."""
    boxes = []
    for name in PLAN_ORDER[mode]:
        x0, x1, y0, y1 = GEOM[mode][name]["box"]
        boxes.append(((max(0, int(x0 * w)), max(0, int(y0 * h)),
                       min(w, int(x1 * w)), min(h, int(y1 * h))), name))
    return boxes


def pad_part(text: str, name: str, mode: str) -> str:
    """Дописание недостающих позиций '#' по expected-длине части."""
    exp = GEOM[mode][name]["expected"]
    text = (text or "").upper()
    if name == "region":
        # регион — 2 или 3 цифры; если OCR молчит, честно ставим '##'
        if len(text) == 0:
            return "##"
        return text[:exp] if len(text) > exp else text
    if len(text) < exp:
        return text + "#" * (exp - len(text))
    return text[:exp] if len(text) > exp + 1 else text


def assemble(parts: dict, mode: str) -> str:
    """Сборка полного номера из распознанных частей (порядок слева-направо/сверху-вниз)."""
    out = []
    for name in PLAN_ORDER[mode]:
        out.append(pad_part(parts.get(name, ""), name, mode))
    return "".join(out)
