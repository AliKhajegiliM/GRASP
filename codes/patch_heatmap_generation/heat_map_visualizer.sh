

python /projects/ovcare/classification/Ali/Bladder_project/codes/patch_heatmap_generation/heat_map_visualizer.py \
 --model GRASP \
 --encoder swin \
 --hidden_layers 256 128 \
 --mags 5 10 20 \
 --batch_size 16 \
 --num_folds 3 \
 --feat_size 768 \
 --lr 0.001 \
 --weight_decay 0.01 \
 --epochs 20 \
 --classes UCC:0 MicroP:1 \
 --crop_size 224 \
 --path_to_checkpoints /projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/results/vit/ \
 --path_to_outputs /projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/results/vit/ \
 --patch_location /projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/heatmap_scratch/ECCV/ \
 --path_to_save_heatmaps /projects/ovcare/classification/Ali/Heram/codes/Bladder_codes/new_benchmark/heatmaps/ \

