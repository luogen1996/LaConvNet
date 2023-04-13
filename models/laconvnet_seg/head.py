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

import numpy as np

import torch
import torch.nn as nn

from utils.utils import bboxes_iou
from layers.conv_layer import aspp_decoder


class REShead(nn.Module):
    """
    segmentation layer
    """
    def __init__(self,in_ch):
        super(REShead, self).__init__()
        self.sconv=nn.Sequential(aspp_decoder(in_ch,in_ch,1),
                                 nn.UpsamplingBilinear2d(scale_factor=8)
                                 )
        self.loss_fn=nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, xin,yin, x_label=None,y_label=None):

        batchsize = xin.shape[0]
        mask=self.sconv(yin)
        devices=xin.device


        if x_label is None:  # not training
            mask=(mask.sigmoid()>0.5).float().squeeze(1)
            box=torch.zeros(batchsize,5,device=devices)
            return box,mask

        loss_seg=self.loss_fn(mask,y_label)*batchsize
        loss=loss_seg.sum()
        return loss,torch.zeros_like(loss),loss_seg
