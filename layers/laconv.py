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
from torchvision.models.resnet import conv1x1
import  torch.nn.functional as F
import math
from layers.lib.sa.modules import Aggregation


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Pool(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x,y=None,z=None):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class MHAtt2D(nn.Module):
    def __init__(self, q_planes,kv_planes,hiddens,out_planes,n_head=8,drop_r=0.1):
        super(MHAtt2D, self).__init__()
        self.n_head=n_head
        self.hiddens=hiddens

        self.linear_v = nn.Linear(kv_planes, hiddens)
        self.linear_k = nn.Linear(kv_planes, hiddens)
        self.linear_q = nn.Conv2d(q_planes, hiddens,1)
        self.linear_merge = nn.Conv2d(hiddens, out_planes,1)

        self.dropout = nn.Dropout(drop_r)

    def forward(self, v, k, q, mask):
        n_batches,_,h,w = q.size()

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.n_head,
            int(self.hiddens / self.n_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.n_head,
            int(self.hiddens / self.n_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            self.n_head,
            int(self.hiddens / self.n_head),
            -1,
        ).transpose(2,3)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(2,3).contiguous().view(
            n_batches,
            self.hiddens,
            h,
            w
        )

        atted = self.linear_merge(atted)

        return F.relu(atted)

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, not_mask):
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=not_mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class ParameterGenerator(nn.Module):
    def __init__(self,  planes,cond_planes,out_planes,tsize=4):
        super(ParameterGenerator, self).__init__()
        self.pos=PositionEmbeddingSine(planes//2)
        self.mha=MHAtt2D(planes*tsize*tsize,cond_planes,out_planes,out_planes*tsize*tsize)
        self.tsize=tsize
        self.norm=nn.BatchNorm2d(out_planes)
    def pixel_packing(self,x):
        b,c,h,w=x.size()
        x=x.view(b, c,  h // self.tsize, self.tsize, w // self.tsize,self.tsize). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c * self.tsize * self.tsize, h // self.tsize, w // self.tsize)
        return x
    def token_recovering(self,x):
        b,c,h,w=x.size()
        x=x.view(b,c//self.tsize//self.tsize,self.tsize,self.tsize,h,w). \
            permute(0,1,4,2,5,3).contiguous().view(b,-1,h*self.tsize,w*self.tsize)
        return x
    def forward(self, x,y,y_mask):
        b,c,h,w=x.size()
        b,l,_=y.size()
        visual_position = self.pos(torch.ones(b, h, w).float().to(x.device))
        x=x+visual_position
        if self.tsize>1:
            x=self.pixel_packing(x)
        conds=self.mha(q=x,k=y,v=y,mask=y_mask)
        if self.tsize>1:
            conds=self.token_recovering(conds)
        return F.relu(self.norm(conds))

class LaConv(nn.Module):
    def __init__(
            self, num_features: int, cond_features: int,kernel_size: int,bias: bool = False, weight_softmax: bool = False,
            weight_dropout: float = 0.1,dilation:int=1,stride:int=1,group=8
    ):
        super().__init__()
        self.linear = nn.Conv2d(cond_features, pow(kernel_size,2)*group,kernel_size=1,stride=stride)
        self.unfold=nn.Unfold(kernel_size,dilation=dilation,padding=kernel_size//2,stride=stride)
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.bias=None
        self.group=group
        self.kernel=kernel_size
        self.stride=stride
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation,
                                       pad_mode=0)
        if bias:
            self.bias=nn.Linear(cond_features, num_features)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        weight=self.linear(cond).view(b,self.group,self.kernel* self.kernel,-1)

        if self.weight_softmax:
            weight = F.softmax(weight, 1)
        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        x = self.aggregation(x, weight)

        if self.bias is not None:
            bias=self.bias(cond).view(b,c,1,1)
            x=x+bias
        return x

class LaConvBlock(nn.Module):
    expansion = 4
    def __init__(self, dim,text_dim,cond_dim, stride=1,
                 kernel=7, dilation=1, norm_layer=None,packing_size=1,group=8):
        super(LaConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        #parameter generator
        self.cond_generator=ParameterGenerator(dim,text_dim,cond_dim,tsize=packing_size)
        # block 1
        self.conv1 = LaConv(dim, cond_dim,kernel_size=kernel,stride=stride,weight_softmax=False,group=group)
        self.bn1 = norm_layer(dim)
        self.act =  nn.SiLU(inplace=True)
        #block2
        self.block1 = nn.Sequential(
            conv1x1(dim , dim* self.expansion),
            norm_layer(dim* self.expansion),
            nn.SiLU(inplace=True),
            conv1x1(dim* self.expansion, dim),
            norm_layer(dim)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim, 3,padding=1, stride=stride,groups=dim),
            nn.Conv2d(dim, dim, 1)
        ) if stride>1 else None
        self.stride = stride
        self.kernel=kernel

    def forward(self, x,cond,cond_mask):
        identity = x
        conds=self.cond_generator(x,cond,cond_mask)
        out = self.conv1(x,conds)
        out=self.bn1(out)
        if self.downsample:
            identity=self.downsample(identity)
        out=self.act(identity+out)
        out=self.act(self.block1(out)+out)
        return out