# Генератор синтетики

Самодостаточен: только numpy + Pillow, работает офлайн, воспроизводим по seed.

    pip install -r requirements.txt
    python generate_synthetic.py --count 6000 --seed 20260420

Ключи: --mix (доли типов), --mode-mix (доли фонов car/flat/billboard),
--jpeg-min/--jpeg-max, --no-strong-aug (быстрее и проще). Результат:
`images/synthetic/*.jpg`, `labels/*.txt`, `synth_rows.jsonl` (строки meta.csv
собираются командой `python ../tools/dataset.py meta` из корня решения).

При воспроизведении с тем же seed файл `synth_rows.jsonl` совпадает побайтово
(проверено). Шрифт по умолчанию — системный DejaVu Sans Bold; можно положить
свой .ttf в `fonts/` рядом или задать env `GRZ_FONT`.
