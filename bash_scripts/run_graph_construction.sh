#!/bin/bash
#SBATCH --job-name graph_const.
#SBATCH --cpus-per-task 1
#SBATCH --output /projects/ovcare/classification/Ali/Heram/codes/singularity_module/run_tests/output_gc_%a.out
#SBATCH --error /projects/ovcare/classification/Ali/Heram/codes/singularity_module/run_tests/output_gc_%a.err
#SBATCH --workdir /projects/ovcare/classification/singularity_modules/singularity_multi_mag_analysis
#SBATCH -p gpu3090,rtx5000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alikhm@student.ubc.ca
SINGULARITY_PATH=$(ls /opt/ | grep singularity | tail -1)
PATH="${PATH}:/opt/$SINGULARITY_PATH/bin"

singularity exec -B /projects/ovcare/classification/ singularity_disc.sif python app.py \
    --mags 5 10 20 \
    --feat_location /projects/ovcare/classification/TCGA_features/Diagnostics/LUAD/swin/ \
    --graph_location /projects/ovcare/classification/Ali/Heram/Dataset/NSCLC_dataset/swin_graphs/LUAD/ \
    --manifest_location /projects/ovcare/classification/Ali/Heram/Dataset/NSCLC_dataset/nsclc_manifest.csv \