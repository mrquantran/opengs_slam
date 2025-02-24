#!/bin/bash

CONFIG_DIR="configs/mono/waymo"

CONFIG_FILES=(
    "13476.yaml"
    "100613.yaml"
    "405841.yaml"
    "152706.yaml"
    "158686.yaml"
    "153495.yaml"
    "132384.yaml"
    "163454.yaml"
    "106762.yaml"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    for i in {1..1}; do
        echo "Running python slam.py --config $CONFIG_DIR/$CONFIG_FILE (Run $i)"
        CUDA_VISIBLE_DEVICES=0 python slam.py --config "$CONFIG_DIR/$CONFIG_FILE"
    done
done
