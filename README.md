# Распознавание нестандартных ГРЗ (type1 / 1А / 1Б / other)

Полуфинал Волга-IT 2026, номинация «Искусственный интеллект и анализ данных».
Офлайн-решение: детектор знаков (YOLOv8n, 4 класса) → выпрямление по quad →
CRNN+CTC-распознавание строк → постобработка по маскам ГОСТ → CSV.

```
solution/          пакет инференса (cli, pipeline, detector, recognizer, typer,
                   crops, plate_format, rectify, server)
  models/          веса: detector.pt, crnn.pt (кладутся в сдачу!)
training/          скрипты обучения: train_detector.py, build_rec_dataset.py,
                   train_rec.py, export_onnx.py, benchmark.py
tools/             менеджер датасета, разметчик, импорт внешних данных,
                   блюр лиц, проверка формата выхода; organizers/ — скрипты
                   валидатора и эвалуатора
dataset/           ДАТАСЕТ: images/real, images/synthetic, labels/, generator/
tests/             selftest.py (без обучения) и smoke_e2e.sh (сквозной прогон)
docs/              пояснительная записка
config.yaml        конфигурация запуска
debug/             отладочный набор организаторов (в датасет НЕ включать!)
```

---

## 0. Установка (референсная конфигурация: i5-7600 + GTX 1050 Ti + 16 ГБ)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Для инференса без обучения достаточно: `numpy opencv-python Pillow PyYAML torch`.
Linux-серверу без монитора — `opencv-python-headless`.

## 1. КАК ЗАПУСТИТЬ РЕШЕНИЕ (главная команда)

```bash
python -m solution.cli --images /путь/к/каталогу/с/картинками --out result.csv
```

Вход — каталог с `.jpg/.png` (ноль, один или несколько ГРЗ на кадре).
Выход — `result.csv` (UTF-8, `;`, заголовок):

```
image;plate_num;plate_type;confidence
img_0042.jpg;A123BC716;type1a;0.930
img_0043.jpg;AB12399;type1b;0.881
img_0044.jpg;H741C###;type1;0.622
```

Способы передать путь (все допустимы п. 4.3, приоритет сверху вниз):

```bash
python -m solution.cli --images DIR --out result.csv   # 1) аргументы
GRZ_IMAGES_DIR=DIR GRZ_OUT_CSV=result.csv python -m solution.cli   # 2) env
# 3) config.yaml: run.images / run.out
```

Полезные флаги: `--device cpu|cuda:0`, `--imgsz 640`, `--conf 0.3`,
`--recursive`, `--limit N`, `--bench` (время на кадр), `--quiet`.
Без найденных знаков файл просто не содержит строк для этого изображения.
Формат можно проверить: `python tools/check_output_format.py result.csv`.

Серверный режим (п. 3 «локально/в облаке»): `python -m solution.server
--host 0.0.0.0 --port 8090`, затем `POST /predict` бинарником jpg/png → JSON.

Если весов в `solution/models/` нет, решение не падает: детекция идёт
классическим contour-бэкендом, OCR возвращает `#` (режим демонстрации).
Для сдачи веса обязательны (см. п. 2–4).

## 2. Сборка датасета (раздел 6 задания)

Структура уже на месте: `dataset/{images/real,images/synthetic,labels,generator}`.

**2.1. Синтетика (не менее 5000 кадров, воспроизводимо по seed):**

```bash
python tools/dataset.py gen-synth --count 6000 --seed 20260420
```

Генератор `dataset/generator/generate_synthetic.py` (numpy+Pillow, офлайн):
рисует знаки по геометрии ГОСТ 520×112 / 290×170, вклеивает случайной
гомографией в процедурные фоны (ниша авто / асфальт / боке / щит-экран),
накладывает дождь/снег/грязь/блик/ночь/смаз/JPEG. Метки (YOLO 5+8 чисел)
и строки датасета — точные, т.к. геометрия известна. Тот же seed → тот же
набор побайтово.

**2.2. Реальные данные** (минимумы задания: 150 кадр. 1А / 300 кадр. 1Б /
50 негативов; уникальных читаемых номеров — 50/100):

* своя съёмка/видеорегистратор; открытые источники только с лицензией
  CC0 / CC BY / public domain (см. раздел «Лицензии»);
* лица размыть: `python tools/blur_faces.py --dir dataset/images/real --inplace`;
* разметка GUI-разметчиком (рамка → тип → текст; валидация по маске на лету):

  ```bash
  python tools/label_assist.py --images путь/к/фото --blur --pre
  ```
  `--pre` включает предразметку обученным детектором — это только подсказки,
  каждую рамку и номер подтверждает/исправляет человек (требование честной
  разметки). Альтернатива: разметка в LabelMe/X-AnyLabeling + импорт
  `python tools/import_labelme.py --dir ...`; импорт.csv-формат — в
  `tools/import_public.py`.

**2.3. Сборка meta.csv и отчёт валидатора (прикладывается к сдаче):**

```bash
python tools/dataset.py meta
python tools/dataset.py validate            # -> dataset/validation_report.txt
python tools/dataset.py stats               # -> dataset/STATS.md
# защита от попадания отладочного набора:
python tools/dataset.py hash --dir debug/ --out dataset/debug_hashes.txt
python tools/dataset.py validate --forbidden-hashes dataset/debug_hashes.txt
```

Валидатор должен вернуть **0 ошибок** (предупреждения по объёмам допустимо
снизить, догнав данные).

## 3. Обучение моделей

> Если RAM мало (<8 ГБ) или нет SSD — уменьшите `--batch` (например, 8) и поставьте
> `--workers 0`: dataloader c воркерами + mosaic потребляет ~1.5–2.5 ГБ поверх torch.

```bash
# 3.1. Детектор (4 класса). Первый запуск скачает yolov8n.pt (~6 МБ), дальше офлайн.
python training/train_detector.py --epochs 120 --imgsz 640 --batch 16 --device 0
# результат: solution/models/detector.pt

# 3.2. Распознаватель CRNN (CTC).
python training/build_rec_dataset.py            # кропы строк из dataset/
python training/train_rec.py --epochs 60 --batch 128 --device 0
# результат: solution/models/crnn.pt
```

Нюанс CTC: первые ~10–20 эпох loss может стоять на ~3.5–4 («всё в blank»).
В скрипте это снято флагом `--blank-bias -1.5` (старт. смещение blank-класса,
по умолчанию включён) и cosine-lr. Если loss упорно не падает — поднимите
`--lr 2.5e-3` и число эпох. Проверенный на smoke-прогоне профиль:
val CER долина ~2% при полном датасете достигается за 40–60 эпох.

Ручная подстройка по метрикам: см. раздел «Скорость и качество» в
docs/пояснительная_записка.md. Дообучение — те же скрипты (для детектора
`--base solution/models/detector.pt`).

**Рекомендации по данным для обучения (важно для «длинного хвоста»):**
в датасете и, следовательно, в обучении доля 1А/1Б должна быть не менее
~55% знаков; negative-класс `other` (прицепы/мото/иностранные/щиты) — ≥8%;
аугментации генератора (дождь/ночь/snow/angle…) специально перекрывают
«тяжёлые» условия скрытого теста.

## 4. Ускорение и самопроверка

```bash
python training/benchmark.py --images debug/ --iters 3      # avg/p50/p95 мс на кадр
python training/export_onnx.py                                # ONNX/TensorRT (опц.)
python tests/selftest.py                                      # 29 проверок формата/пайплайна, без весов
bash tests/smoke_e2e.sh                                       # сквозной тест: 600 кадров -> 1 эпоха YOLO -> CRNN -> evaluate
```

Оценка на отладочном наборе (нужны картинки debug_set/ от организаторов —
в репозиторий не включены, возьмите из их облака; эталон лежит в debug/):

```bash
python -m solution.cli --images debug_set/ --out debug_result.csv
python tools/organizers/evaluate.py --gt debug/debug_labels.csv --pred debug_result.csv --images debug_set/ --details details.csv
```

## 5. Лицензии и «юридическая» сборка (п. 6.3, 8)

* Датасет — CC BY 4.0 (`dataset/LICENSE`); код — MIT (`LICENSE`).
* В `meta.csv` каждой реальной строки проставлены `source`/`license`;
  синтетика — «generated:script», лицензия CC0 (сгенерировано участником).
* Использованы: собственные фото/видео; шрифт DejaVu Sans Bold (Bitstream
  Vera, свободно). Внешние датасеты с NC/ND/SA лицензиями — не включались.
* Отладочный набор организаторов в датасет не входит (проверка md5 см. п. 2.3).
* Решение работает полностью офлайн: никаких сетевых запросов в рантайме,
  веса лежат в solution/models/ и передаются со сдачей.

## 6. Что сдавать (чек-лист)

1. Архив/репозиторий с исходниками этого README (весь `grz-solution/`,
   включая обученные `solution/models/*.pt` и `dataset/`).
2. `dataset/validation_report.txt` — отчёт валидатора (из п. 2.3).
3. `training/*` + `dataset/generator/*` — скрипты обучения и генерации.
4. Инструкция — этот README (команда запуска — раздел 1).
5. `docs/пояснительная_записка.md` (≤5 страниц) — с подставленными замерами
   скорости (`training/benchmark.py`) и метриками evaluate на своём валиде.

Типовые проблемы жюри-конвейера, которые уже закрыты:
* «нет каталога/файла» — структура dataset/ полная;
* «регистр имён» — все имена генератора нижний регистр, meta.csv собирается
  из файлов, а не наоборот;
* «quad против часовой» — генератор/разметчик выдают ЛВ→по часовой;
* «кириллица в plate_num» — нормализация в solution/plate_format.py;
* 100 мс/кадр — YOLOv8n@640 fp16 ≈ 15–25 мс (1050 Ti) + CRNN-батч ≈ 4–8 мс.
