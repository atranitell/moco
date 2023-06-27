#!/bin/bash

python main_lincls.py \
  -a extra.mobilenet_v2 \
  --evaluate \
  --resume model_best.pth.tar \
  --use-yuv \
  --lr 30.0 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/jk/imagenet/ILSVRC/Data/CLS-LOC/