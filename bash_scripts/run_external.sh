#!/bin/bash
#SBATCH --job-name submit
#SBATCH --cpus-per-task 1
#SBATCH --output /projects/ovcare/classification/Ali/Bladder_project/Dataset/error_out/submit_%a.out
#SBATCH --error /projects/ovcare/classification/Ali/Bladder_project/Dataset/error_out/submit_%a.err
#SBATCH -p gpu3090,rtx5000,gpu2080,dgxV100,gpuA6000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alikhm@student.ubc.ca

declare -a models=("Clam_SB" "Clam_MB")
declare -a split_names=("fold-1" "fold-2" "fold-3")

# Submit jobs for each combination of model, split name, and seed
for model in "${models[@]}"; do
  for split_name in "${split_names[@]}"; do
    sbatch --job-name "Blad_${model}_S_${split_name}" /projects/ovcare/classification/Ali/Bladder_project/codes/submit_external.sh "$model" "$split_name"
  done
done
