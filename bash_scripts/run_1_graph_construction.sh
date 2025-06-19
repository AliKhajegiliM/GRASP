

conda init
conda activate grasp

#!/bin/bash

python app.py \
    --mags 5 10 20 \
    --feat_location ./assets/raw_features/ \
    --graph_location ./assets/graphs/ \
    --manifest_location ./assets/files/manifest.csv