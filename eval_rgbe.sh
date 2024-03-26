#!/bin/bash

CURRENT_DIR=$PWD
EXPERIMENT="cvpr2024_rgbe"

ln -s  $CURRENT_DIR/output/checkpoints/train/sdstrack/$EXPERIMENT/SDSTrack_ep0050.pth.tar  ./models/SDSTrack_$EXPERIMENT.pth.tar

python ./RGBE_workspace/test_rgbe_mgpus.py --script_name sdstrack  --num_gpus 1 --threads 4 --epoch 50 --yaml_name $EXPERIMENT
