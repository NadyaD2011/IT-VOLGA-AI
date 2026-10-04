#!/usr/bin/env python3
"""Обучение CRNN-распознавателя строк знака (CTC).

    python training/train_rec.py --data .runs/rec --epochs 40 --batch 128 --device auto

Лог: раз в эпоху — train loss, val CER (редакционное расстояние по символам).
Лучшая модель (минимум CER) -> solution/models/crnn.pt (+ словарь и размер входа
в чекпоинте: solution/recognizer.py подхватывает их автоматически).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from solution.crnn_model import DEFAULT_CHARSET, build_model  # noqa: E402


class LineDataset(Dataset):
    def __init__(self, root: str, split: str, img_h: int, img_w: int, charset: str,
                 augment: bool = False):
        self.samples = []
        with open(os.path.join(root, "manifest.tsv"), encoding="utf-8") as f:
            for ln in f:
                rel, txt = ln.rstrip("\n").split("\t")
                if rel.startswith(split + "/"):
                    self.samples.append((os.path.join(root, rel.replace("/", os.sep)), txt))
        self.img_h, self.img_w = img_h, img_w
        self.augment = augment
        self.char2id = {c: i + 1 for i, c in enumerate(charset)}
        self.blank = 0

    def __len__(self):
        return len(self.samples)

    def _prep(self, img):
        import cv2
        h, w = img.shape[:2]
        s = self.img_h / h
        nw = int(round(w * s))
        if nw <= self.img_w:
            r = cv2.resize(img, (max(8, nw), self.img_h), interpolation=cv2.INTER_AREA)
            out = np.zeros((self.img_h, self.img_w), np.float32)
            out[:, : min(self.img_w, r.shape[1])] = r
        else:
            s = self.img_w / w
            nh = max(6, int(round(h * s)))
            r = cv2.resize(img, (self.img_w, nh), interpolation=cv2.INTER_AREA)
            out = np.zeros((self.img_h, self.img_w), np.float32)
            y0 = (self.img_h - nh) // 2
            out[y0:y0 + nh] = r
        if self.augment:
            if np.random.random() < 0.3:
                out += np.random.normal(0, 8, out.shape).astype(np.float32)
            if np.random.random() < 0.25:
                out = np.roll(out, np.random.randint(-3, 4), axis=1)
            if np.random.random() < 0.2:
                out *= np.random.uniform(0.75, 1.25)
            if np.random.random() < 0.15:  # лёгкий сдвиг/поворот
                M = cv2.getRotationMatrix2D((self.img_w / 2, self.img_h / 2),
                                            np.random.uniform(-3, 3), 1.0)
                out = cv2.warpAffine(out.astype(np.uint8), M, (self.img_w, self.img_h),
                                     borderValue=0).astype(np.float32)
        out = np.clip(out, 0, 255)
        return (out / 255.0 - 0.5) / 0.5

    def __getitem__(self, i):
        import cv2
        path, txt = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, 128), np.uint8)
            txt = ""
        x = self._prep(img).astype(np.float32)
        ids = [self.char2id[c] for c in txt if c in self.char2id]
        return torch.from_numpy(x[None]), torch.tensor(ids, dtype=torch.long), len(ids)


def greedy_decode(logits, charset):
    ids = logits.argmax(-1).tolist()
    out, prev = [], -1
    for t in ids:
        if t != 0 and t != prev:
            out.append(charset[t - 1] if t - 1 < len(charset) else "#")
        prev = t
    return "".join(out)


def dist(a: str, b: str) -> int:
    """Расстояние Левенштейна по символам (metric = CER)."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def collate(batch):
    """x: (B,1,H,W); tids: список тензоров целей разной длины (CTC их конкатенирует)."""
    xs = torch.stack([b[0] for b in batch])
    tids = [b[1] for b in batch]
    ns = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return xs, tids, ns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, ".runs", "rec"))
    ap.add_argument("--out", default=os.path.join(ROOT, "solution", "models", "crnn.pt"))
    ap.add_argument("--charset", default=DEFAULT_CHARSET)
    ap.add_argument("--img-h", type=int, default=32)
    ap.add_argument("--img-w", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--blank-bias", type=float, default=-1.5,
                    help="стартовое смещение bias blank-класса (уход от плато CTC; 0 = выкл.)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_ds = LineDataset(args.data, "train", args.img_h, args.img_w, args.charset, augment=True)
    val_ds = LineDataset(args.data, "val", args.img_h, args.img_w, args.charset, augment=False)
    if len(train_ds) < 100:
        print(f"ВНИМАНИЕ: обучающих строк всего {len(train_ds)} — сначала "
              f"соберите выборку: python training/build_rec_dataset.py", file=sys.stderr)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.num_workers,
                          pin_memory=device != "cpu", collate_fn=collate)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate)

    charset = args.charset
    model = build_model()(num_classes=len(charset) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer = float("inf")
    if args.blank_bias:  # уходим от «всё blank» локального минимума CTC
        with torch.no_grad():
            model.fc.bias[0] += args.blank_bias
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, tids, _n in train_dl:
            x = x.to(device)
            logits = model(x)  # (B,T,C)
            T = logits.shape[1]
            logp = logits.log_softmax(-1).permute(1, 0, 2).cpu()
            tgt, tl = [], []
            for ids in tids:
                tgt.append(ids)
                tl.append(int(ids.numel()))
            if not tgt or min(tl) == 0:
                keep = [i for i, v in enumerate(tl) if v > 0]
                if not keep:
                    continue
                tgt2 = torch.cat([tids[i] for i in keep])
                tl2 = torch.tensor([tl[i] for i in keep], dtype=torch.long)
                logp = logp[:, keep]
                T_len = torch.full((len(keep),), T, dtype=torch.long)
                loss = ctc(logp, tgt2, T_len, tl2)
            else:
                tgt = torch.cat(tgt)
                tl = torch.tensor(tl, dtype=torch.long)
                loss = ctc(logp, tgt, torch.full_like(tl, T), tl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * x.shape[0]
        sched.step()
        tr = tot / max(1, len(train_ds))

        model.eval()
        nchar = nerr = 0
        with torch.no_grad():
            for x, tids, _n in val_dl:
                x = x.to(device)
                lg = model(x).cpu()
                for k in range(x.shape[0]):
                    pred = greedy_decode(lg[k], charset)
                    ref = "".join(charset[i - 1] for i in tids[k].tolist() if i)
                    nerr += dist(ref, pred)
                    nchar += max(1, len(ref))
        if nchar:
            cer_ = nerr / nchar
            key = cer_
            disp = f"val_cer={100 * cer_:.2f}%"
        else:  # пустая валидация: выбираем чекпойнт по train-loss
            key, disp = tr, "val_cer=n/a (пустой val)"
        print(f"ep {epoch:03d} loss={tr:.4f} {disp} ({time.time() - t0:.0f}s)")
        if key < best_cer:
            best_cer = key
            torch.save({"state_dict": model.state_dict(), "charset": charset,
                        "img_h": args.img_h, "img_w": args.img_w,
                        "val_cer": (cer_ if nchar else None), "epoch": epoch}, args.out)
    tail = (f"val CER {100 * best_cer:.2f}%" if nchar else f"train loss {best_cer:.4f}")
    print(f"Лучшая модель: {args.out} ({tail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
