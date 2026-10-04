"""HTTP-сервер решения (локально «на месте» или в облаке — п. 3 раздела 3).

Только стандартная библиотека; офлайн; принимает изображение bytes, отдаёт JSON.

Запуск:
    python -m solution.server --host 0.0.0.0 --port 8090
Эндпоинты:
    GET  /health          -> {"status": "ok", "ready_det": bool, "ready_ocr": bool}
    POST /predict         body = jpg/pngbytes          -> {"plates": [...]}
    POST /predict_dir     body = {"dir": "/path"}      -> пишет CSV, возвращает путь
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .cli import load_config

_READER = None


def reader():
    global _READER
    if _READER is None:
        from .pipeline import PlateReader
        _READER = PlateReader(load_config())
    return _READER


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # тише
        pass

    def _send(self, code: int, payload):
        body = (json.dumps(payload, ensure_ascii=False) if not isinstance(payload, (bytes, bytearray))
                else payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            r = reader()
            self._send(200, {"status": "ok",
                             "detector": type(r.detector).__name__,
                             "ocr_ready": r.recognizer.ready})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        if self.path == "/predict":
            if not data:
                return self._send(400, {"error": "empty body"})
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "frame.jpg")
                with open(p, "wb") as f:
                    f.write(data)
                rows, stats = reader().read_image(p, filename="upload.jpg")
                return self._send(200, {
                    "plates": [{"image": r.image, "plate_num": r.plate_num,
                                "plate_type": r.plate_type, "confidence": r.confidence}
                               for r in rows],
                    "time_ms": round(stats.total_ms, 1)})
        if self.path == "/predict_dir":
            try:
                req = json.loads(data or b"{}")
                d = req["dir"]
                out = req.get("out", "result.csv")
            except Exception as e:
                return self._send(400, {"error": str(e)})
            if not os.path.isdir(d):
                return self._send(400, {"error": f"нет каталога {d}"})
            rows, stats = reader().process_dir(d, out_csv=out)
            return self._send(200, {"out": out, "rows": len(rows),
                                    "avg_ms": round(stats.avg_ms, 1)})
        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("GRZ_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRZ_PORT", "8090")))
    args = ap.parse_args()
    reader()  # прогреть модели до приёма запросов
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"GRZ server http://{args.host}:{args.port}  (POST /predict, GET /health)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
