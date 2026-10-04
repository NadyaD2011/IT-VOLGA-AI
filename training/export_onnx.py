#!/usr/bin/env python3
"""Экспорт моделей в ONNX (и, опционально, TensorRT) для ускорения инференса.

    python training/export_onnx.py            # detector.onnx + crnn.onnx
    python training/export_onnx.py --trt      # + trtexec-скрипт для GTX 1050 Ti

Затем solution читает ONNX, если в config.yaml указано:
    models.detector_weights: solution/models/detector.onnx  (нужен onnxruntime)
    recognition.weights:     solution/models/crnn.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def export_crnn(weights, out, img_h=32, img_w=160):
    import torch
    from solution.crnn_model import build_model
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    charset = ck["charset"]
    model = build_model()(num_classes=len(charset) + 1)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x = torch.randn(1, 1, img_h, img_w)
    torch.onnx.export(model, x, out, opset=17,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch", 1: "time"}},
                      dynamo=False)
    print("CRNN ONNX:", out)
    import json
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"charset": charset}, f)


def export_detector(weights, out, imgsz=640):
    from ultralytics import YOLO
    y = YOLO(weights)
    res = y.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True,
                   half=False)
    src = str(res)
    if os.path.abspath(src) != os.path.abspath(out):
        import shutil
        shutil.move(src, out)
    print("Detector ONNX:", out)


def trt_hint(onnx_path):
    return (f"# TensorRT (установка один раз, далее офлайн):"
            f"\n#   docker run --gpus all nvcr.io/nvidia/tensorrt:24.02-py3 \\"
            f"\n#     trtexec --onnx={onnx_path} --fp16 --saveEngine={onnx_path[:-5]}.engine"
            f"\n# на 1050 Ti даёт ~x1.6 по сравнению с ONNX Runtime CPU+GPU")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=os.path.join(ROOT, "solution/models/detector.pt"))
    ap.add_argument("--crnn", default=os.path.join(ROOT, "solution/models/crnn.pt"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "solution/models"))
    ap.add_argument("--trt", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.isfile(args.crnn):
        export_crnn(args.crnn, os.path.join(args.outdir, "crnn.onnx"))
    else:
        print("crnn.pt не найден — пропуск (сначала python training/train_rec.py)")
    if os.path.isfile(args.detector):
        try:
            export_detector(args.detector, os.path.join(args.outdir, "detector.onnx"))
        except ImportError:
            print("ultralytics не установлен — пропуск детектора")
        if args.trt:
            print(trt_hint(os.path.join(args.outdir, "detector.onnx")))
            print(trt_hint(os.path.join(args.outdir, "crnn.onnx")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
