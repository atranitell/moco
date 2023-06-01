#!/bin/bash

python main_lincls.py \
  -a extra.mobilenet_v2 \
  --lr 15.0 \
  --batch-size 128 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/jk/imagenet/ILSVRC/Data/CLS-LOC/