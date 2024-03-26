#!/bin/bash 

source ~/.bashrc
conda activate sdstrack
# nvidia-smi -a

EXPERIMENT="cvpr2024_rgbe"

python tracking/train.py --script sdstrack --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0 --config $EXPERIMENT
