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

# 53.584, 200epoch
# python main_moco.py \
#   -a extra.mobilenet_v2 \
#   --lr 0.0125 \
#   --batch-size 128 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   --mlp --moco-t 0.2 --aug-plus --cos \
#   /data1/jk/imagenet/ILSVRC/Data/CLS-LOC/ > Exp03.moco2.mobilenet_v2.e200.log 2>&1


# python main_moco.py \
#   -a extra.mobilenet_v2_050 \
#   --use-yuv \
#   --use-autoaug \
#   --epochs 800 \
#   --input-size 224
#   --schedule 480 640 \
#   --lr 0.0125 \
#   --batch-size 128 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   --mlp --moco-t 0.2 --aug-plus --cos \
#   /data1/jk/imagenet/ILSVRC/Data/CLS-LOC/ > Exp07.moco2.mobilenet_v2_050.yuv709f.autoaug.e800.log 2>&1

# python main_moco.py \
#   -a extra.mobilenet_v2 \
#   --use-yuv \
#   --use-autoaug \
#   --epochs 800 \
#   --input-size 224 \
#   --schedule 480 640 \
#   --lr 0.03 \
#   --batch-size 256 \
#   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#   --mlp --moco-t 0.2 --aug-plus --cos \
#   /data/jk/imagenet/ILSVRC/Data/CLS-LOC/ > Exp08.moco2.mobilenet_v2.yuv709f.auotaug.e800.log 2>&1

python main_moco.py \
  -a extra.mobilenet_v2_050 \
  --use-yuv \
  --use-autoaug \
  --epochs 800 \
  --input-size 384 \
  --schedule 480 640 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  /data/jk/imagenet/ILSVRC/Data/CLS-LOC/ > Exp09.moco2.mobilenet_v2_050.yuv709f.i384.e800.log 2>&1