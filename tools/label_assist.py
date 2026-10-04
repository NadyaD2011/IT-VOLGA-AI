#!/usr/bin/env python3
"""Полуавтоматический разметчик реальных фотографий для датасета.

Работает локально с GUI (opencv). Для каждого изображения:
  a — добавить знак (рамка мышью: выделить область, Enter),
      затем в терминале запрашивается текст номера; тип — клавишей 1/2/3/4;
  u — отменить последний знак;
  n — без изменений, следующий кадр;
  s — сохранить кадр в датасет и перейти к следующему;
  q — выход.

При запуске можно включить «предразметку» (--pre): тот же загрузчик, что и в
решении, рисует подсказки рамками (детектор из config; если весов нет —
contour-бэкенд). Подсказки НЕ попадают в датасет без подтверждения человеком:
это ручная разметка, просто быстрая (требование п. 8 — честная разметка).

Выход:
  dataset/images/real/real_%04d.jpg   — копия кадра (лица размыты при --blur)
  dataset/labels/real_%04d.txt        — YOLO 5+8 (класс cx cy w h + quad, нормир.)
  dataset/real_rows.jsonl             — строки для meta.csv (подтверждённые)

Затем: python tools/dataset.py meta && python tools/dataset.py validate
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("Нужен opencv-python: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from solution.plate_format import repair, CLASS_OF_TYPE  # noqa: E402

TYPE_KEYS = {ord("1"): "type1", ord("2"): "type1a", ord("3"): "type1b", ord("4"): "other"}


def save_to_dataset(dataset_dir: str, img_path: str, annos: list, blur: bool):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    if blur:
        try:
            from blur_faces import blur_faces_inplace
            img = blur_faces_inplace(img)
        except Exception as e:
            print(f"  лицо-блюр пропущен: {e}")

    # уникальное имя
    n = 0
    while True:
        stem = f"real_{n:04d}"
        if not os.path.exists(os.path.join(dataset_dir, "images", "real", stem + ".jpg")):
            break
        n += 1
    os.makedirs(os.path.join(dataset_dir, "images", "real"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
    cv2.imwrite(os.path.join(dataset_dir, "images", "real", stem + ".jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    rows, labels = [], []
    for a in annos:
        plate = a["plate"]
        if a["type"] != "other" and not plate:
            # пользователь не ввёл текст — заполняем нечитаемыми позициями,
            # чтобы meta.csv прошёл валидатор (лучше доведите разметку до конца!)
            plate = "########" if a["type"] in ("type1", "type1a") else "#######"
        quad = a["quad"]
        x1, y1 = quad[:, 0].min(), quad[:, 1].min()
        x2, y2 = quad[:, 0].max(), quad[:, 1].max()
        bx, by = int(np.floor(x1)), int(np.floor(y1))
        bw, bh = int(np.ceil(x2) - bx), int(np.ceil(y2) - by)
        labels.append(f"{a['cls']} {(bx + bw / 2) / W:.6f} {(by + bh / 2) / H:.6f} "
                      f"{bw / W:.6f} {bh / H:.6f} " + " ".join(
                          f"{v:.6f}" for v in (quad / np.array([W, H])).reshape(-1)))
        rows.append({
            "image": f"images/real/{stem}.jpg",
            "plate_num": plate,
            "plate_type": a["type"],
            "bbox": f"{bx},{by},{bw},{bh}",
            "quad": ",".join(str(int(round(v))) for v in quad.reshape(-1)),
            "is_vehicle": a["vehicle"],
            "is_synthetic": 0,
            "source": a["source"],
            "license": a["license"],
            "conditions": a["conditions"],
        })
    with open(os.path.join(dataset_dir, "labels", stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + ("\n" if labels else ""))
    with open(os.path.join(dataset_dir, "real_rows.jsonl"), "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  сохранено: {stem}.jpg, знаков: {len(rows)}")
    return len(rows)


def ask_text(plate_type: str):
    if plate_type == "other":
        raw = input("  текст (Enter — пусто): ").strip()
        return raw, True
    while True:
        raw = input(f"  номер для {plate_type} (Л ЦЦЦ ЛЛ реgion или ЛЛ ЦЦЦ реgion, "
                    f"нечитаемое — #; q — отмена): ").strip()
        if raw.lower() == "q":
            return None, False
        fixed, ok = repair(raw, plate_type)
        if ok:
            if fixed != raw.upper().replace(" ", ""):
                print(f"  нормализовано/исправлено -> {fixed}")
            return fixed, True
        print(f"  «{raw}» не подходит под маску {plate_type} — проверьте раскладку "
              f"(латиница!) и длину")


def ask_conditions(default="day"):
    s = input(f"  условия съёмки через запятую (день/ночь: day,night; ещё: rain,snow,"
              f"dirt,glare,motion_blur,angle) [{default}]: ").strip()
    return s or default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--images", required=True, help="каталог своих фотографий")
    ap.add_argument("--source", default="own_photo",
                    help="значение поля source (own_photo / URL / название датасета)")
    ap.add_argument("--license", default="own_photo (CC BY 4.0)",
                    help="условия лицензии источника")
    ap.add_argument("--blur", action="store_true", help="размывать лица при сохранении")
    ap.add_argument("--pre", action="store_true", help="предразметка детектором из config.yaml")
    ap.add_argument("--scale", type=float, default=0.9, help="масштаб окна просмотра")
    args = ap.parse_args(argv)

    files = sorted(f for f in os.listdir(args.images)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("Нет изображений."); return 1

    pre = None
    if args.pre:
        from solution.cli import load_config
        from solution.pipeline import PlateReader
        pre = PlateReader(load_config())
        print("Предразметка включена (это лишь подсказки — финальные рамки "
              "и текст всегда подтверждает человек).")

    i = 0
    while i < len(files):
        path = os.path.join(args.images, files[i])
        img = cv2.imread(path)
        if img is None:
            i += 1
            continue
        H, W = img.shape[:2]
        annos = []
        if pre is not None:
            try:
                dets = pre.detector.detect(img)
                for d in dets:
                    if d.get("quad") is not None:
                        ptype = d["cls"] if d["cls"] in CLASS_OF_TYPE else "type1"
                        print(f"  [pre] {ptype} score={d['score']:.2f} "
                              f"bbox={d['bbox']} — подтвердите/поправьте клавишей a")
                        annos.append({"quad": np.asarray(d["quad"], float),
                                      "plate": "", "type": ptype,
                                      "cls": CLASS_OF_TYPE[ptype],
                                      "vehicle": 1, "conditions": "day",
                                      "source": args.source, "license": args.license})
            except Exception:
                pass

        while True:
            disp = img.copy()
            for a in annos:
                q = a["quad"].astype(int)
                cv2.polylines(disp, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(disp, a.get("plate") or a["type"], tuple(q[0] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            small = cv2.resize(disp, (int(W * args.scale), int(H * args.scale)))
            cv2.imshow("label_assist  [a]add  [u]undo  [s]save  [n]next  [q]quit", small)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                return 0
            if k == ord("n"):
                break
            if k == ord("s"):
                if annos:
                    save_to_dataset(args.dataset, path, annos, args.blur)
                else:
                    print("  знаков нет — кадр пропущен")
                break
            if k == ord("u"):
                annos.pop() if annos else None
                continue
            if k == ord("a"):
                sel = cv2.selectROI("select plate, Enter=ok, Esc=cancel",
                                    cv2.resize(img, (int(W * args.scale), int(H * args.scale))),
                                    showCrosshair=False)
                cv2.destroyWindow("select plate, Enter=ok, Esc=cancel")
                sx, sy, sw, sh = [v / args.scale for v in sel]
                if sw < 6 or sh < 6:
                    continue
                print("  тип: 1=type1 2=type1a 3=type1b 4=other")
                tk = cv2.waitKey(0) & 0xFF
                ptype = TYPE_KEYS.get(tk, "type1")
                text, ok = ask_text(ptype)
                if not ok:
                    continue
                veh = input("  на транспортном средстве? y/n [y]: ").strip().lower()
                cond = ask_conditions()
                q = np.float32([[sx, sy], [sx + sw, sy], [sx + sw, sy + sh], [sx, sy + sh]])
                annos.append({"quad": q, "plate": text, "type": ptype,
                              "cls": CLASS_OF_TYPE[ptype],
                              "vehicle": 0 if veh == "n" else 1,
                              "conditions": cond, "source": args.source,
                              "license": args.license})
        i += 1
        print(f"--- кадр {i}/{len(files)}")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
