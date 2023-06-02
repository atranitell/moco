#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import cv2

import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import extra_models
import extra_transforms

augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    extra_transforms.Cutout(n_holes=1, length=20),
    extra_transforms.RGBtoYUV709F(),
]

train_dataset = datasets.ImageFolder(
  '/data1/jk/imagenet/ILSVRC/Data/CLS-LOC/val/', 
  transforms.Compose(augmentation))

for data in train_dataset:
  data = data[0]
  print(data.shape, data)
  final = extra_transforms.YUV709FtoRGB()(data) * 255
  print(final.shape, final)

  image = final.permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
  cv2.imwrite('demo.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

  exit(0)