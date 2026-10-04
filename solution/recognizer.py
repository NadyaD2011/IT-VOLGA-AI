"""OCR-распознаватель: CRNN + CTC-декод + постобработка по маске ГОСТ.

Принимает список выпрямленных кропов строк знака (grayscale ndarray),
возвращает (текст, средняя уверенность символа) на каждый кроп. Части знака
(body/top/series/region) режет pipeline по геометрии solution/crops.py.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def prep_crop(crop_gray: np.ndarray, img_h: int = 32, img_w: int = 160) -> np.ndarray:
    """Масштабирование с сохранением пропорций + паддинг. Общий для инференса
    и подготовки обучающей выборки — кропы обязаны резаться одинаково."""
    if crop_gray.ndim == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)
    h, w = crop_gray.shape[:2]
    if w <= 4 or h <= 4:
        return np.full((img_h, img_w), -1.0, np.float32)
    s_h = img_h / h
    nw = int(round(w * s_h))
    if nw <= img_w:
        r = cv2.resize(crop_gray, (max(8, nw), img_h),
                       interpolation=cv2.INTER_AREA if s_h < 1 else cv2.INTER_CUBIC)
        out = np.zeros((img_h, img_w), np.float32)
        out[:, : min(img_w, nw)] = r.astype(np.float32)
    else:  # очень широкая строка — вписываем по ширине, по центру по высоте
        s = img_w / w
        nh = max(6, int(round(h * s)))
        r = cv2.resize(crop_gray, (img_w, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((img_h, img_w), np.float32)
        y0 = (img_h - nh) // 2
        out[y0:y0 + nh, :] = r.astype(np.float32)
    return ((out / 255.0) - 0.5) / 0.5


class Recognizer:
    def __init__(self, weights: str | None = None, device: str = "auto",
                 img_h: int = 32, img_w: int = 160, charset: str = None,
                 char_prob_min: float = 0.55):
        from .crnn_model import DEFAULT_CHARSET
        self.img_h, self.img_w = img_h, img_w
        self.char_prob_min = char_prob_min
        self.torch = None
        self.model = None
        self.device = "cpu"
        self.charset = charset or DEFAULT_CHARSET
        if weights:
            self._load(weights, device)

    # -- загрузка --------------------------------------------------------- #
    def _load(self, weights: str, device: str):
        import torch
        from .crnn_model import build_model
        self.torch = torch
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.charset = ck.get("charset", self.charset)
        self.img_h = ck.get("img_h", self.img_h)
        self.img_w = ck.get("img_w", self.img_w)
        cls = build_model()
        self.model = cls(num_classes=len(self.charset) + 1)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)

    @property
    def ready(self) -> bool:
        return self.model is not None

    # -- препроцессинг ----------------------------------------------------- #
    def _prep(self, crop_gray: np.ndarray) -> np.ndarray:
        return prep_crop(crop_gray, self.img_h, self.img_w)

    # -- инференс ---------------------------------------------------------- #
    def recognize(self, crops_gray, batch_size: int = 16):
        """crops_gray: list[np.ndarray]. Без весов возвращает пустые строки
        (решение остаётся работоспособным: детекция/тип есть, текст '#')."""
        if not self.ready:
            return [("", 0.0) for _ in crops_gray]
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(crops_gray), batch_size):
                batch = np.stack([self._prep(c) for c in crops_gray[i:i + batch_size]])
                x = torch.from_numpy(batch).unsqueeze(1).to(self.device)
                logits = self.model(x)
                probs = logits.softmax(-1).float().cpu().numpy()
                for p in probs:
                    outs.append(self._decode(p))
        return outs

    def _decode(self, probs: np.ndarray):
        ids = probs.argmax(axis=1)
        mx = probs.max(axis=1)
        text, ps, prev = [], [], -1
        for t, idx in enumerate(ids):
            if idx != 0 and idx != prev:
                ch = self.charset[idx - 1] if 0 < idx <= len(self.charset) else "#"
                text.append(ch if mx[t] >= self.char_prob_min else "#")
                ps.append(float(mx[t]))
            prev = idx
        s = "".join(text)
        conf = float(sum(ps) / len(ps)) if ps else 0.0
        return s, conf


class StubRecognizer(Recognizer):
    """Для автотестов: всегда пусто, но конвейер полностью исполняется."""

    def __init__(self):
        super().__init__(weights=None)
