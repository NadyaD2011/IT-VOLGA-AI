#!/bin/bash
# Полный сквозной smoke-тест: датасет -> обучение (1-2 эпохи) -> инференс -> метрики
set -e
cd /home/user/grz-solution
DS=/tmp/train_ds
rm -rf $DS /tmp/test_imgs /tmp/run_det /tmp/run_rec
mkdir -p $DS /tmp/test_imgs
for d in images/real images/synthetic labels generator; do mkdir -p $DS/$d; done
cp dataset/generator/*.py dataset/generator/requirements.txt $DS/generator/
cp dataset/README.md dataset/LICENSE $DS/

echo "=== [1/6] генерация 600 кадров"
python3 dataset/generator/generate_synthetic.py --dataset $DS --count 600 --seed 777 --progress 200

echo "=== [2/6] meta.csv"
python3 - <<EOF
import sys; sys.path.insert(0, 'tools')
import importlib.util
spec = importlib.util.spec_from_file_location('dsmod','tools/dataset.py')
m = importlib.util.module_from_spec(spec); sys.modules['dsmod']=m; spec.loader.exec_module(m)
m.DATASET='$DS'
m.cmd_meta(None)
EOF

echo "=== [3/6] валидатор"
python3 tools/organizers/validate_dataset.py $DS --report - --max-per-code 3 | tail -6 || true

echo "=== [4/6] YOLO детектор, 1 эпоха"
python3 training/train_detector.py --dataset $DS --work /tmp/run_det --out /tmp/models/detector.pt --epochs 1 --imgsz 320 --batch 16 --device cpu

echo "=== [5/6] CRNN: сборка выборки + 3 эпохи"
python3 training/build_rec_dataset.py --dataset $DS --out /tmp/run_rec
python3 training/train_rec.py --data /tmp/run_rec --out /tmp/models/crnn.pt --epochs 3 --batch 64 --device cpu --num-workers 0

echo "=== [6/6] инференс + метрики"
mkdir -p /tmp/test_imgs && python3 - <<EOF
import os, shutil, csv
src = '$DS'
rows = list(csv.DictReader(open(os.path.join(src,'meta.csv'), encoding='utf-8'), delimiter=';'))
gt = open('/tmp/test_gt.csv','w',encoding='utf-8')
gt.write('image;plate_num;plate_type;is_vehicle\n')
n=0
for i, r in enumerate(rows):
    if i % 12: continue
    p = os.path.join(src, r['image'])
    if not os.path.isfile(p): continue
    name=os.path.basename(p)
    shutil.copy(p, '/tmp/test_imgs/'+name)
    gt.write(f"{name};{r['plate_num']};{r['plate_type']};{r['is_vehicle']}\n")
    n+=1
gt.close(); print('test frames:', n)
EOF
python3 tests/run_infer_eval.py --images /tmp/test_imgs --gt /tmp/test_gt.csv \
    --det /tmp/models/detector.pt --crnn /tmp/models/crnn.pt --imgsz 320 | head -22
echo "=== SMOKE DONE"
