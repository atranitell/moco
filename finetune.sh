#!/bin/bash

python main_lincls.py \
  -a extra.mobilenet_v2 \
  --use-yuv \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained checkpoint_0799.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data/jk/imagenet/ILSVRC/Data/CLS-LOC/ > Exp08.moco2.mobilenet_v2.yuv709f.auotaug.e800.finetune.log 2>&1