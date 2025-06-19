#!/bin/bash


encoder="KimiaNet"


if [ "$encoder" == "convnext_base" ] || [ "$encoder" == "vit" ] || [ "$encoder" == "CTransPath" ] || [ "$encoder" == "Phikon" ]; then
  feat_size=768
elif [ "$encoder" == "PLIP" ]; then
  feat_size=512
elif [ "$encoder" == "resnet50" ]; then
  feat_size=2048
elif [ "$encoder" == "swin" ] || [ "$encoder" == "KimiaNet" ]; then
  feat_size=1024
elif [ "$encoder" == "Lunit-Dino" ]; then
  feat_size=384
fi

conda init
conda activate grasp

# For a full set with 5 subtypes, you should enter classes like this --classes CC:0 EC:1 HGSC:2 LGSC:3 MC:4  ; However, here we test 
# with a toy dataset, so --classes CC:0 LGSC:1 has been passed.

python3 ./codes/run_test_graph.py \
  --model_name GRASP \
  --split_name fold-1 \
  --batch_size 8 \
  --mags "5x 10x 20x" \
  --hidden_layers 256 128 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --seed 256 \
  --epochs 5 \
  --feature_size "$feat_size" \
  --classes CC:0 LGSC:1 \
  --spatial_gcn False \
  --conv_layer gcn \
  --path_to_folds ./assets/files/KimiaNet_data_folds_graph.json \
  --path_to_save ./assets/model_outputs/${encoder}/gcn/ \