#!/bin/bash

# Gehe ins Pointcept-Verzeichnis
cd ~/Pointcept

# Setze PYTHONPATH
export PYTHONPATH=.

# Starte das Training mit Resume vom besten Modell
python tools/train.py \
  --config-file configs/s3dis/semseg-pt-v3m1-0-base.py \
  --options task.name=SemanticSegmentationS3DISArea5 \
           batch_size_per_gpu=1 \
           enable_amp=True \
           save_path=exp/ptv3res_fixed_epoch \
          #  resume=True \
          #  weight=exp/gpu0/model/model_last.pth

#--config-file configs/s3dis/semseg-pt-v3m1-0-base.py \
#semseg-sonata-v1m1-3a-s3dis-lin.py