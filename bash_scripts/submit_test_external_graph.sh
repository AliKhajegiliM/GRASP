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
python3 /projects/ovcare/classification/Ali/Bladder_project/codes/test_external_graph.py \
    --model_name ZoomMIL \
    --split_name "fold-1" \
    --batch_size 1 \
    --mag 5x 10x 20x \
    --lr 0.001 \
    --weight_decay 0.01 \
    --seed 256 \
    --epochs 50 \
    --feature_size 768 \
    --classes CC:0 EC:1 HGSC:2 LGSC:3 MC:4 \
    --path_to_folds /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/cross_validation/mil_folds/public/CTransPath_OV_data_folds_bin.json \
    --path_to_save /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs/test_sets/public/ \
    --path_to_load /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs/CTransPath/others/
