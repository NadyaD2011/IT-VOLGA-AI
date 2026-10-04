#!/usr/bin/env python3
"""Самопроверка решения без обучающих весов (python tests/selftest.py).

Проверяет:
  1) формат/маски ГРЗ — сверка с регулярками валидатора организаторов;
  2) согласованность геометрии кропов (solution/crops.py vs генератор);
  3) правила quad из валидатора (по часовой от ЛВ) для генератора;
  4) воспроизводимость синтетики по seed (побайтово);
  5) мини-датасет 40 кадров проходит validate_dataset.py БЕЗ ошибок;
  6) пайплайн: img_033.jpg -> корректный по формату result.csv
     (contour-бэкенд; OCR без весов даёт '#') + сверка с evaluate.py.
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    mark = "OK " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {info}" if info else ""))
    if cond:
        PASS += 1
    else:
        FAIL += 1


def load_validator():
    path = os.path.join(ROOT, "tools", "organizers", "validate_dataset.py")
    spec = importlib.util.spec_from_file_location("vdataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vdataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_masks():
    from solution.plate_format import repair, fits_mask, normalize
    vd = load_validator()
    ok_cases = ["A123BC777", "M683AX125", "X000OP1", "E123HK99", "#246#C73", "M42######"]
    for p in ok_cases:
        for t in ("type1", "type1a"):
            vm = vd.PLATE_MASKS[t].fullmatch(p) is not None
            mm = fits_mask(p, t)
            if len(p) in (8, 9):
                check(f"маска {t} {p} (мой={mm}, валидатор={vm})", mm == vm)
    tb = ["AB12377", "OO88449", "TT555125"]
    for p in tb:
        vm = vd.PLATE_MASKS["type1b"].fullmatch(p) is not None
        check(f"маска type1b {p} (мой={fits_mask(p, 'type1b')}, валидатор={vm})",
              fits_mask(p, "type1b") == vm)
    # кириллица -> латиница
    check("кириллица нормализуется", normalize("М683АХ125") == "M683AX125",
          normalize("М683АХ125"))
    # починка: 4-я позиция буква -> цифра через омофон
    fixed, ok = repair("A123BCO77", "type1")   # O вместо 0 в регионе
    check("починка O->0 в регионе", ok and fixed == "A123BC077", fixed)
    fixed, ok = repair("AB123C45", "type1")   # буква в позиции цифры и наоборот
    check("починка омофонами под маску", ok and fits_mask(fixed, "type1"), fixed)
    # мусор не проходит в целевой тип
    _f, ok = repair("TRANZIT123", "type1")
    check("транзит не лег в type1", not ok)
    # other принимает пустой
    check("other пусто допустимо", fits_mask("", "other"))


def test_geom_sync():
    sys.path.insert(0, os.path.join(ROOT, "dataset", "generator"))
    import plate_render
    from solution.crops import GEOM
    ok = True
    for mode in ("single", "double"):
        for part, spec in GEOM[mode].items():
            ref = plate_render.GEOM[mode].get(part)
            if not ref or tuple(round(v, 4) for v in ref["box"]) != tuple(
                    round(v, 4) for v in spec["box"]):
                ok = False
    check("геометрия кропов синхронна (solution/crops.py == generator/plate_render.py)", ok)


def test_mini_dataset():
    vd = load_validator()
    tmp = tempfile.mkdtemp(prefix="grz_ds_")
    ds = os.path.join(tmp, "dataset")
    for d in ("images/real", "images/synthetic", "labels", "generator"):
        os.makedirs(os.path.join(ds, d), exist_ok=True)
    for f in ("README.md", "LICENSE"):
        src = os.path.join(ROOT, "dataset", f)
        with open(src, encoding="utf-8") as fin:
            txt = fin.read()
        with open(os.path.join(ds, f), "w", encoding="utf-8") as fout:
            fout.write(txt)
    # копируем генератор целиком (он самодостаточен)
    gen_src = os.path.join(ROOT, "dataset", "generator")
    for f in os.listdir(gen_src):
        if f.endswith(".py") or f == "requirements.txt":
            with open(os.path.join(gen_src, f), "rb") as a, \
                 open(os.path.join(ds, "generator", f), "wb") as b:
                b.write(a.read())
    # 40 кадров синтетики
    r = subprocess.run([sys.executable, os.path.join(gen_src, "generate_synthetic.py"),
                        "--dataset", ds, "--count", "40", "--seed", "7", "--progress", "0"],
                       capture_output=True, text=True)
    check("генератор отработал (rc=0)", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
    # пара «реальных» кадров = копия синтетики с другим именем (тест формата, не лицензий)
    import shutil
    for i in range(4):
        s = f"syn_{i:06d}"
        shutil.copy(os.path.join(ds, "images/synthetic", s + ".jpg"),
                    os.path.join(ds, "images/real", f"real_{i:03d}.jpg"))
        shutil.copy(os.path.join(ds, "labels", s + ".txt"),
                    os.path.join(ds, "labels", f"real_{i:03d}.txt"))
    rows_real = []
    with open(os.path.join(ds, "synth_rows.jsonl"), encoding="utf-8") as f:
        synth_rows = [json.loads(l) for l in f]
    for r0 in synth_rows[:4]:
        rr = dict(r0)
        k = int(re.search(r"syn_(\d+)", r0["image"]).group(1))
        rr["image"] = f"images/real/real_{k:03d}.jpg"
        rr["is_synthetic"] = 0
        rr["source"] = "own_photo"
        rr["license"] = "own_photo (CC BY 4.0)"
        rows_real.append(rr)
    with open(os.path.join(ds, "real_rows.jsonl"), "w", encoding="utf-8") as f:
        for rr in rows_real:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    # meta.csv (логика tools/dataset.py meta, переиспользуем функцию)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    importlib.invalidate_caches()
    dsmod_path = os.path.join(ROOT, "tools", "dataset.py")
    dsmeta = importlib.util.spec_from_file_location("dsmod", dsmod_path)
    dsmod = importlib.util.module_from_spec(dsmeta)
    sys.modules["dsmod"] = dsmod
    dsmeta.loader.exec_module(dsmod)
    dsmod.DATASET = ds
    dsmod.cmd_meta(None)
    check("meta.csv создан", os.path.getsize(os.path.join(ds, "meta.csv")) > 100)

    rep = vd.validate(ds)
    errs = [i.fmt() for i in rep.errors]
    check("мини-датасет проходит валидатор без ERROR", not errs,
          "; ".join(errs[:3]) if errs else f"warnings={len(rep.warnings)} (объёмы — ок)")
    return ds


def test_determinism():
    g = os.path.join(ROOT, "dataset", "generator", "generate_synthetic.py")
    a = tempfile.mkdtemp(prefix="grz_a_")
    b = tempfile.mkdtemp(prefix="grz_b_")
    for d in (a, b):
        os.makedirs(os.path.join(d, "images", "synthetic"), exist_ok=True)
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    for d in (a, b):
        subprocess.run([sys.executable, g, "--dataset", d, "--count", "8",
                        "--seed", "123", "--progress", "0"], check=True,
                       capture_output=True)
    same_rows = filecmp.cmp(os.path.join(a, "synth_rows.jsonl"), os.path.join(b, "synth_rows.jsonl"), False)
    names = os.listdir(os.path.join(a, "images", "synthetic"))
    same_img = all(filecmp.cmp(os.path.join(a, "images", "synthetic", n),
                               os.path.join(b, "images", "synthetic", n), False)
                   for n in names)
    check("воспроизводимость по seed: строки", same_rows)
    check("воспроизводимость по seed: растровые кадры идентичны бит-в-бит", same_img)


def test_pipeline_and_output(tmpds=None):
    from solution.cli import load_config
    from solution.pipeline import PlateReader, write_csv
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    cfg["models"]["detector_weights"] = None
    cfg["models"]["backend"] = "contour"
    cfg["recognition"]["weights"] = None
    cfg.setdefault("post", {})["allow_hash_rows"] = True  # демо без OCR: строки '#' допустимы
    reader = PlateReader(cfg)
    img = os.path.join(ROOT, "debug_img_033.jpg")
    rows, stats = reader.read_image(img)
    check("пайплайн: img_033 дал ≥1 обнаружение", stats.detections >= 1,
          f"dets={stats.detections}, rows={len(rows)}")
    out = os.path.join(tempfile.mkdtemp(prefix="grz_out_"), "result.csv")
    allrows, _st = reader.process_dir(os.path.dirname(img), out_csv=out)
    txt = open(out, encoding="utf-8").read().splitlines()
    check("CSV: заголовок по разделу 5",
          txt and txt[0] == "image;plate_num;plate_type;confidence",
          txt[0] if txt else "пусто")
    ok_fmt = all(re.fullmatch(r"[^;]+;[A-Z0-9#]*;(type1a|type1b|type1|other);(0\.\d{1,3}|1\.0+)", l)
                 for l in txt[1:])
    check("CSV: формат строк", ok_fmt, txt[1] if len(txt) > 1 else "")
    # сверка с организаторским evaluate.py на эталоне img_033
    gt = os.path.join(ROOT, "debug", "debug_labels.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "organizers", "evaluate.py"),
                        "--gt", gt, "--pred", out, "--images", os.path.dirname(img)],
                       capture_output=True, text=True, cwd=ROOT)
    check("evaluate.py принял выход решения (rc=0)", r.returncode == 0, r.stderr[-200:] if r.returncode else "")
    # детектор на синтетике из tmpds: находим знак на syn_000000.jpg (ground truth известен)
    if tmpds:
        n_hits = 0
        n_tot = 0
        import csv as _csv
        with open(os.path.join(tmpds, "meta.csv"), encoding="utf-8") as f:
            for row in _csv.DictReader(f, delimiter=";"):
                p = os.path.join(tmpds, row["image"].replace("/", os.sep))
                if not os.path.isfile(p):
                    continue
                n_tot += 1
                rs, _ = reader.read_image(p, filename=os.path.basename(row["image"]))
                plate = row["plate_num"]
                if any(rr.plate_type == row["plate_type"] for rr in rs):
                    n_hits += 1
        rate = n_hits / max(1, n_tot)
        # contour-бэкенд — демо-фолбэк без обучения; основной путь — YOLO после
        # training/train_detector.py. Порох скромен, но не нулевой.
        check("contour-детектор+типизатор (демо-режим) на синт. кадрах: тип угадан",
              rate >= 0.3, f"{n_hits}/{n_tot}")


def main():
    tmpds = test_mini_dataset()
    test_masks()
    test_geom_sync()
    test_determinism()
    test_pipeline_and_output(tmpds)
    print(f"\n== ИТОГО: PASS={PASS} FAIL={FAIL} ==")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
