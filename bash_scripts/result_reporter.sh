
python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --models DeepMIL VarMIL Clam_SB Clam_MB \
  --mags 20x \
  --batch_size 1 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --num_classes 5\
  --path_to_outputs 

python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --mags 20x \
  --batch_size 1 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 100 \
  --num_classes 5\
  --models DeepMIL VarMIL Clam_SB Clam_MB PatchGCN_spatial PatchGCN_latent DGCN_spatial DGCN_latent H2MIL HiGT GRASP ZoomMIL \
  --path_to_outputs /projects/ovcare/classification/Ali/Heram/codes/ov_codes/results/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder Phikon \

python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --mags 20x \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 100 \
  --num_classes 5\
  --models GRASP \
  --batch_size 2 \
  --path_to_outputs /projects/ovcare/classification/Ali/Heram/codes/ov_codes/results/CTransPath \


python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --mags 20x \
  --batch_size 1 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --is_external True \
  --num_classes 5\
  --models DeepMIL VarMIL Clam_SB H2MIL HiGT ZoomMIL \
  --path_to_outputs /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs/test_sets/public/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder KimiaNet/others/ \

python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --mags 20x \
  --batch_size 4 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --is_external True \
  --num_classes 5\
  --models GRASP \
  --path_to_outputs /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs/test_sets/public/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder KimiaNet/sageconv/ \


python3 /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/scripts/orgnaize_best_outputs_for_maryam.py \
  --mags 20x \
  --batch_size 4 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --is_external True \
  --num_classes 5\
  --models GRASP \
  --path_to_outputs /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs/test_sets/public/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder Phikon/sageconv/ 



python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --batch_size 1 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --is_external True \
  --num_classes 5\
  --models DeepMIL \
  --path_to_outputs /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs_zero-shot/test_sets/public/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder KimiaNet/others/ \
  --mags 5x \

python3 /projects/ovcare/classification/Ali/Bladder_project/codes/result_reporter.py \
  --batch_size 4 \
  --num_folds 3 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --epochs 50 \
  --is_external True \
  --num_classes 5\
  --models GRASP \
  --path_to_outputs /projects/ovcare/classification/Ali/Multi_mag_backbone/dataset/ocean_comp/mil_outputs_zero-shot/test_sets/private/ \
  --path_to_save_fig /projects/ovcare/classification/Ali/Heram/codes/OCEAN/plots/ \
  --encoder prov-GigaPath/sageconv/ \
  --mags 5x \