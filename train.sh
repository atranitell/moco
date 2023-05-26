#!/bin/bash

# python main_moco.py \
#   -a resnet50 \
#   --lr 0.0125 \
#   --batch-size 128 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   --mlp --moco-t 0.2 --aug-plus --cos \
#   /data/jk/imagenet/ILSVRC/Data/CLS-LOC/

# python main_moco.py \
#   -a resnet50 \
#   --lr 0.0125 \
#   --use-yuv \
#   --batch-size 128 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   --mlp --moco-t 0.2 --aug-plus --cos \
#   /data/jk/imagenet/ILSVRC/Data/CLS-LOC/

python main_moco.py \
  -a extra.mobilenet_v2 \
  --lr 0.0125 \
  --batch-size 128 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  /data1/jk/imagenet/ILSVRC/Data/CLS-LOC/