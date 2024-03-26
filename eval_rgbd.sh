#!/bin/bash

CURRENT_DIR=$PWD
WORKSPACE="Depthtrack_workspace"
EXPERIMENT="cvpr2024_rgbd"

ln -s  $CURRENT_DIR/output/checkpoints/train/sdstrack/$EXPERIMENT/SDSTrack_ep0015.pth.tar  $CURRENT_DIR/models/SDSTrack_$EXPERIMENT.pth.tar


sed -i '1,2d' ./$WORKSPACE/trackers.ini
sed -i  "1i [$EXPERIMENT]" ./$WORKSPACE/trackers.ini
sed -i  "2i label = $EXPERIMENT" ./$WORKSPACE/trackers.ini

sed -i '6d' ./$WORKSPACE/trackers.ini
sed -i  "5a paths = $CURRENT_DIR/lib/test/vot" ./$WORKSPACE/trackers.ini

sed -i "10d " ./lib/test/vot/sdstrack_baseline.py
sed -i  "9a run_vot_exp('sdstrack', '$EXPERIMENT', vis=False, out_conf=True, channel_type='rgbd')" ./lib/test/vot/sdstrack_baseline.py


cd $WORKSPACE
echo Begin to evaluate $EXPERIMENT.yml in $WORKSPACE, come on!
vot evaluate --workspace ./ $EXPERIMENT
if [ $? -eq 0 ]; then
    echo evaluate $EXPERIMENT.yml in $WORKSPACE success!
    vot analysis --nocache --name $EXPERIMENT
    if [ $? -eq 0 ]; then
        echo Finish evaluate and analisis $EXPERIMENT.yml in $WORKSPACE, congratulations!
    else
        echo "====analisis failed!===="
        exit 1
    fi
else
    echo "====evaluate failed!===="
	exit 1
fi
cd ..
