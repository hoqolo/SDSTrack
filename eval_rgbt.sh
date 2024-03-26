#!/bin/bash

CURRENT_DIR=$PWD
EXPERIMENT="cvpr2024_rgbt"

ln -s  $CURRENT_DIR/output/checkpoints/train/sdstrack/$EXPERIMENT/SDSTrack_ep0040.pth.tar  ./models/SDSTrack_$EXPERIMENT.pth.tar

python ./RGBT_workspace/test_rgbt_mgpus.py --script_name sdstrack  --num_gpus 1 --threads 4  --epoch 40 --dataset_name LasHeR --yaml_name $EXPERIMENT 

python ./RGBT_workspace/test_rgbt_mgpus.py --script_name sdstrack  --num_gpus 1 --threads 4  --epoch 40 --dataset_name RGBT234 --yaml_name $EXPERIMENT