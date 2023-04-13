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

import torch
import torch.nn as nn

from models.laconvnet_seg.head import REShead
from models.language_encoder import language_encoder
from models.visual_encoder import LaConvNet
from layers.neck_layers import *



class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.lang_encoder=language_encoder(__C,pretrained_emb,token_size)
        self.visual_encoder=LaConvNet(__C.BACKBONE,text_dim=__C.HIDDEN_SIZE)
        self.multi_scale_manner=PAFPN(in_channels=[128,256,512])

        self.detection_neck=PAFPN_OnePath(in_channels=[128,256,512])
        self.head = REShead(128)

    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False


    def forward(self, x, y, det_label=None, seg_label=None):

        # vision and language encoding
        y = self.lang_encoder(y)
        x = self.visual_encoder(x, y['lang_feat'], y['lang_feat_mask'])

        # multi-scale vision features
        x = self.multi_scale_manner(x[-3:])

        #segmentation necks
        x = self.detection_neck(x)

        # output
        if self.training:
            loss, loss_det, loss_seg = self.head(x[2], x[0], det_label, seg_label)
            return loss, loss_det, loss_seg
        else:
            box, mask = self.head(x[2], x[0])
            return box, mask


if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.USE_GLOVE = False
            self.WORD_EMBED_SIZE = 300
            self.HIDDEN_SIZE = 512
            self.N_SA = 0
            self.FLAT_GLIMPSES = 8
            self.DROPOUT_R = 0.1
            self.LANG_ENC = 'lstm'
            self.VIS_ENC = 'darknet'
            self.VIS_PRETRAIN = True
            self.PRETTRAIN_WEIGHT = './darknet.weights'
            self.ANCHORS = [[116, 90], [156, 198], [373, 326]]
            self.ANCH_MASK = [[0, 1, 2]]
            self.N_CLASSES = 0
            self.VIS_FREEZE = True
    cfg=Cfg()
    model=Net(cfg,torch.zeros(1),100)
    # model.train()
    img=torch.zeros(2,3,224,224)
    lang=torch.randint(10,(2,14))
    seg, det=model(img,lang)
    print(seg.size(),det.size())