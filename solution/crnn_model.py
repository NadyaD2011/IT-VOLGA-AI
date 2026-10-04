"""CRNN (CNN + BiLSTM + CTC) для распознавания строки знаков ГРЗ.

Архитектура выбрана под требования задания:
  * компактная (веса < 8 МБ, инференс строки 32x160 на GTX 1050 Ti < 3 мс);
  * конец-to-конец обучается CTC-лоссом без выравнивания по символым;
  * словарь фиксирован (12 букв + 10 цифр), что вместе с постобработкой
    по маске ГОСТ отсеивает невозможные символы.

Модуль не импортирует torch на верхнем уровне — тесты формата можно гонять
без глубокого обучения.
"""
from __future__ import annotations

DEFAULT_CHARSET = "0123456789ABEKMHOPCTYX"
IMG_H = 32
IMG_W = 160


def build_model():
    """Возвращает nn.Module CRNN. Вызывается из training/train_rec.py и
    solution/recognizer.py (через torch)."""
    import torch
    from torch import nn

    class CRNN(nn.Module):
        """Вход (B,1,32,W) -> выход (B,T,C), где C = len(charset)+1 (0 = CTC blank)."""

        def __init__(self, num_classes: int, feat_width_pool: bool = True):
            super().__init__()
            def block(ci, co, pool):
                layers = [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
                if pool:
                    layers.append(nn.MaxPool2d(2))
                return layers

            c = []
            c += block(1, 64, True)      # 32x160 -> 16x80
            c += block(64, 128, True)    # -> 8x40
            c += block(128, 256, False)  # -> 8x40
            c += block(256, 256, True)   # -> 4x20
            c += block(256, 512, False)  # -> 4x20
            c += block(512, 512, False)  # -> 4x20
            self.cnn = nn.Sequential(*c)
            self.lin = nn.Linear(512 * 4, 256)
            self.lstm = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(512, num_classes)
            self._num_classes = num_classes

        def forward(self, x):
            x = self.cnn(x)                       # (B,512,4,Wf)
            B, C, H, W = x.shape
            x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)
            x = torch.relu(self.lin(x))
            x, _ = self.lstm(x)                   # (B,W,512)
            x = self.fc(x)
            return x

    return CRNN


def char_probs_to_text(probs, charset: str, blank: int = 0, prob_min: float = 0.0):
    """Жадная CTC-декодировка. probs: (T, C) numpy. Возвращает (text, mean_p, per_char)."""
    import numpy as np
    ids = probs.argmax(axis=1)
    maxp = probs.max(axis=1)
    text, ps = [], []
    prev = -1
    for t, idx in enumerate(ids):
        if idx != blank and idx != prev:
            ch = charset[idx - 1] if idx - 1 < len(charset) else ""
            text.append(ch if maxp[t] >= prob_min else "#")
            ps.append(float(maxp[t]))
        prev = idx
    mean_p = float(sum(ps) / len(ps)) if ps else 0.0
    return "".join(text), mean_p, ps
