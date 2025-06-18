#!/bin/bash
#SBATCH --mem 24000
#SBATCH --cpus-per-task 2
#SBATCH --output /projects/ovcare/classification/Ali/Bladder_project/Dataset/error_out/job_%a.out
#SBATCH --error /projects/ovcare/classification/Ali/Bladder_project/Dataset/error_out/job_%a.err
#SBATCH -p rtx5000,dgxV100,gpuA6000,gpu3090
#SBATCH --gres=gpu:1

model=$1
fold=$2

source /projects/ovcare/classification/Ali/miniconda3/etc/profile.d/conda.sh
conda activate cuda6
python3 /projects/ovcare/classification/Ali/Bladder_project/codes/test_external.py \
    --model_name "$model" \
    --split_name "$fold" \
    --batch_size 1 \
    --mag 20x \
    --lr 0.0001 \
    --num_workers 2 \
    --pooling None \
    --weight_decay 0.0001 \
    --seed 256 \
    --epochs 50 \
    --feature_size 512 \
    --classes UCC:0 MicroP:1 \
    --path_to_folds /projects/ovcare/classification/Ali/Bladder_project/Dataset/scripts/resnet18_data_folds_pt.json \
    --path_to_save /projects/ovcare/classification/Ali/Bladder_project/results/ \
