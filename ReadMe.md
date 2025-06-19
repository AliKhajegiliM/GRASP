# GRASP
Note: This repository is being gradually updated!

[GRASP](https://openreview.net/pdf?id=k6xlOfZnTC) official object-oriented implementation, demonstrating graph-based aggregation of histopathology instances using multiple microscopic magnifications. The repository includes sample data and scripts for building graphs from feature files, training a GRASP model, and reporting the evaluation metrics.

## Directory layout

- `assets/raw_features/` – Example feature files per slide (HDF5 format).
- `assets/graphs/` – Generated DGL graphs will be placed here.
- `assets/model_outputs/` – Model checkpoints and outputs are written here.
- `assets/files/` – Supporting files including the manifest and data splits.
- `bash_scripts/` – Bash scripts that illustrate the full pipeline.
- `codes/` – Python sources for training and evaluation.

## Requirements

- Python 3.9+
- PyTorch and DGL (GPU enabled is recommended)
- Use Conda to create the `grasp` environment defined in `assets/grasp.yml`

Run the following to create and activate the environment:
```bash
conda env create -f assets/grasp.yml
conda activate grasp
```

## 1. Graph construction

`bash_scripts/run_1_graph_construction.sh` builds graph files from the raw feature matrices. Features are expected under `assets/raw_features/` and a manifest describing each slide is provided in `assets/files/manifest.csv`.

```bash
bash bash_scripts/run_1_graph_construction.sh
```

This expands to:

```bash
python app.py \
  --mags 5 10 20 \
  --feat_location ./assets/raw_features/ \
  --graph_location ./assets/graphs/ \
  --manifest_location ./assets/files/manifest.csv
```

Graphs are stored in `assets/graphs/raw_features/*.bin`.

## 2. Train GRASP

`bash_scripts/run_2_submit_grasp.sh` trains the GRASP model on the graphs. The script chooses an encoder (here `KimiaNet`) and sets the feature dimensionality accordingly. The command executed is similar to the following:

```bash
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
  --feature_size 1024 \
  --classes CC:0 LGSC:1 \
  --spatial_gcn False \
  --conv_layer gcn \
  --path_to_folds ./assets/files/KimiaNet_data_folds_graph.json \
  --path_to_save ./assets/model_outputs/KimiaNet/gcn/
```

Modify the script to adjust magnifications, data splits, encoder name or other hyperparameters.

## 3. Report results

After training, run `bash_scripts/run_3_result_reporter.sh` to aggregate metrics across seeds and produce simple plots:

```bash
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
  --mags 5x 10x 20x
```

Metrics such as accuracy, balanced accuracy, F1, and AUC are printed to the console.

## Notes

The provided data and scripts are for demonstration. Replace the sample feature files and manifest with your own dataset following the same folder structure. Ensure the `--classes` argument in the training script matches your subtype labels.

