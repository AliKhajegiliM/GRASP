#!/bin/bash

conda init
conda activate cuda6

python3 ./codes/result_reporter.py \
  --batch_size 8 \
  --num_folds 1 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 5 \
  --num_classes 2 \
  --models GRASP \
  --path_to_outputs ./assets/model_outputs/ \
  --encoder KimiaNet/gcn/ \
  --mags 5x 10x 20x \