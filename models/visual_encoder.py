# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch
import torch.nn as nn
from torch.backends import cudnn

from layers.laconv import *

class LaConvNet(nn.Module):
    def __init__(self, model, text_dim, dims = (16, 64, 128, 256, 512), groups = (4, 4, 8, 8, 16),strides = (1, 2, 2, 2, 2), packing_sizes = (8,4,2,1,1)):
        super(LaConvNet, self).__init__()

        if model=='LaConvNetS':
            kernels = ((3, 5), (7, ), (7, 7), (7, 7, 7, 7), (7, ))
        elif model=='LaConvNetB':
            kernels = ((3, 5, 7), (7, 7, 7), (7, 7, 7, 7), (7, 7, 7, 7, 7, 7), (7, 7, 7))
        else:
            assert NotImplementedError

        self.stem = Pool(3, 16, ksize=3)

        blocks=[]
        for i in range(len(groups)):
            blocks.append(self._make_layer(LaConvBlock, dims[i], groups[i], len(kernels[i]), stride=strides[i],
                                           text_dim=text_dim, kernels=kernels[i], packing_size=packing_sizes[i]))
        self.blocks=nn.ModuleList(blocks)

    def _make_layer(self, block, dim, group, n_layer, stride=1, text_dim=512, kernels=(3, 5), packing_size=1):
        layers = []
        if stride==2:
            layers.append(Pool(dim//4 if packing_size==4 else dim//2,dim,ksize=1))
        for _ in range(0, n_layer):
            layers.append(block(dim=dim, cond_dim=dim*packing_size//2 if dim<128 else dim, group=group, text_dim=text_dim,
                                kernel=kernels[_], packing_size=packing_size))
        return nn.ModuleList(layers)

    def forward(self,  x,y, y_mask):

        x=self.stem(x)
        feats=[]
        for blocks in self.blocks:
            for layer in blocks:
                x = layer(x, y, y_mask)
            feats.append(x)

        return feats