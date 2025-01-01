# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips
from mmseg.models.utils import *
import attr

from IPython import embed

import cv2
from .hypercorre import hypercorre_topk2
from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
import torch.nn.functional as F

import math

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# @HEADS.register_module()
# class SegFormerHead(BaseDecodeHead):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, feature_strides, **kwargs):
#         super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
#         assert len(feature_strides) == len(self.in_channels)
#         assert min(feature_strides) == feature_strides[0]
#         self.feature_strides = feature_strides

#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

#         decoder_params = kwargs['decoder_params']
#         embedding_dim = decoder_params['embed_dim']


#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

#         self.linear_fuse = ConvModule(
#             in_channels=embedding_dim*4,
#             out_channels=embedding_dim,
#             kernel_size=1,
#             # norm_cfg=dict(type='SyncBN', requires_grad=True)
#             norm_cfg=dict(type='GN', num_groups=1)
#         )

#         self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

#     def encode_key(self, frame, need_sk=True, need_ek=True): 
#         return

#     def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
#         return
    
#     def update_memory(self, query_key, query_selection, memory_key, 
#                     memory_shrinkage, memory_value):
#         return

#     def forward(self, inputs):
#         x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
#         c1, c2, c3, c4 = x

#         print(c1.shape, c2.shape, c3.shape, c4.shape)

#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape

#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

#         x = self.dropout(_c)
#         x = self.linear_pred(x)

#         # print(torch.cuda.memory_allocated(0))

#         return x

class pooling_mhsa(nn.Module):
    def __init__(self,dim):
        super().__init__()

        # pool_ratios=[1,2,3,4]
        # pool_ratios=[12, 16, 20, 24]

        pool_ratios = [[1,2,3,4], [2,4,6,8], [4,8,12,16], [8,16,24,32]]
        # self.pool_ratios = pool_ratios

        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

        self.d_convs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pool_ratios])

        # self.d_convs1 = nn.ModuleList([nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pool_ratios[0]])
        # self.d_convs2 = nn.ModuleList([nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pool_ratios[1]])
        # self.d_convs3 = nn.ModuleList([nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pool_ratios[2]])
        # self.d_convs4 = nn.ModuleList([nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pool_ratios[3]])

    def forward(self,x,redu_ratios):

        B, N, C, hy, wy = x.shape

        pools = []

        x = x.reshape(-1,C,hy,wy)

        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # hy_final = round(hy/redu_ratios[0])+round(hy/redu_ratios[1])+round(hy/redu_ratios[2])+round(hy/redu_ratios[3]) #8,3,4,5
        # wy_final = round(wy/redu_ratios[0])+round(wy/redu_ratios[1])+round(wy/redu_ratios[2])+round(wy/redu_ratios[3])

        # print("hy_wy ",hy_final,wy_final)

        for (l,redu_ratio) in zip(self.d_convs,redu_ratios):

            pool = F.adaptive_avg_pool2d(x, (round(hy/redu_ratio), round(wy/redu_ratio)))

            # print("rrr  ",round(hy/redu_ratio),round(wy/redu_ratio))

            pool = pool + l(pool) # fix backward bug in higher torch versions when training

            pools.append(pool.view(B*N, C, -1))
        
        # print("pp ",pools[0].shape,pools[1].shape,pools[2].shape,pools[3].shape)
        
        pools = torch.cat(pools, dim=2)

        pools = self.norm(pools.permute(0,2,1))#B,-1,C

        # print("pools ",pools.shape)

        pools = pools.reshape(B,N,-1,C)
        
        return pools
        
def softmax_w_top(x, top=20):
    print('x',x.shape)
    x = x.squeeze(1)
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x

def label_to_onehot(gt, num_classes, ignore_index=255):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)          

    return onehot.permute(0, 3, 1, 2)

class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False
                 ):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
   
            #ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
#            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        b_p,c,k,_ = proxy.shape
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(b_p, self.key_channels, -1)
        value = self.f_down(proxy).view(b_p, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=False)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=False)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False, 
                 use_bg=False,
                 fetch_attention=False
                 ):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention
                                                     )

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False
                 ):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention
                                                           )
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            #ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output

class SpatialTemporalGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialTemporalGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs,clip_num ,memory=None,memory_num=None):
#        if self.use_gt and gt_probs is not None:
#            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
#            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
#            gt_probs = gt_probs.view(batch_size, c, -1)
#            feats = feats.view(batch_size, feats.size(1), -1)
#            feats = feats.permute(0, 2, 1) # batch x hw x c 
#            gt_probs = F.normalize(gt_probs, p=1, dim=2)# batch x k x hw
#            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
#            return ocr_context               
        
        assert(probs.size(0)==feats.size(0))
        # print(feats.shape,probs.shape)
        # if not self.training:
        #     probs = resize(probs, size=(120,120),mode='bilinear',align_corners=False)
        probs_s = torch.split(probs,split_size_or_sections=int(probs.size(0)/(clip_num+1)), dim=0)
        feats_s = torch.split(feats,split_size_or_sections=int(feats.size(0)/(clip_num+1)), dim=0)
        if memory is None:
            contexts=[]
    
            for probs,feats in zip(probs_s,feats_s):
                batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
                probs = probs.view(batch_size, c, -1)
                feats = feats.view(batch_size, feats.size(1), -1)
                feats = feats.permute(0, 2, 1) # batch x hw x c 
                probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
                ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
                # print('none',feats.shape,probs.shape)
                contexts.append(ocr_context.unsqueeze(0))
            
            contexts = torch.cat(contexts,dim=0)
            
            #contexts,_ = torch.max(contexts,dim=0)
            contexts = torch.mean(contexts,dim=0)
        else:
            # if len(memory)>0:
            #     memory= [m.detach() for m in memory]
            for probs,feats in zip(probs_s,feats_s):
                batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
                probs = probs.view(batch_size, c, -1)
                feats = feats.view(batch_size, feats.size(1), -1)
                feats = feats.permute(0, 2, 1) # batch x hw x c 
                probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
                # print(feats.shape,probs.shape)
                ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
                
                # print(len(memory),memory_num,id(memory))
                while len(memory)>memory_num:
                    memory.pop(0)
                    # print("pop")
                    # exit()
                memory.append(ocr_context.unsqueeze(0))
            # print("type",type(memory),type(memory[0]))
            memory = [tensor.to('cuda') for tensor in memory]
            # print("fff",[i.device for i in memory])
            contexts = torch.cat(memory,dim=0)
#            contexts,_ = torch.max(contexts,dim=0)
            contexts = torch.mean(contexts,dim=0)
        
        # print("ss",contexts.shape)
        
        return contexts

@HEADS.register_module()
class SegFormerHead_clipsNet(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides,dim=[64, 128, 320, 512], **kwargs):
        super(SegFormerHead_clipsNet, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.dim=dim
        
        self.num_classes = 124

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.embeding = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.pooling_mhsa_c1 = pooling_mhsa(c1_in_channels)
        self.pooling_mhsa_c2 = pooling_mhsa(c2_in_channels)
        self.pooling_mhsa_c3 = pooling_mhsa(c3_in_channels)
        self.pooling_mhsa_c4 = pooling_mhsa(c4_in_channels)

        self.pooling_linear = nn.Linear(embedding_dim*4,embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.memory_module = FeatureMemory()

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # self.linear1 = nn.Linear(dim[0],dim[0],bias=True)
        # self.linear2 = nn.Linear(dim[0],dim[0],bias=True)

        self.sdeco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.sdeco2=small_decoder2(embedding_dim,512, self.num_classes)
        self.sdeco3=small_decoder2(embedding_dim,512, self.num_classes)
        self.sdeco4=small_decoder2(embedding_dim,512, self.num_classes)

        # self.deco1=small_decoder2(embedding_dim,256+64, self.num_classes)
        # self.deco2=small_decoder2(embedding_dim,512+64, self.num_classes)
        # self.deco3=small_decoder2(embedding_dim,512+64, self.num_classes)
        # self.deco4=small_decoder2(embedding_dim,512+64, self.num_classes)
        
        # self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        # self.deco2=small_decoder2(embedding_dim,512, self.num_classes)
        # self.deco3=small_decoder2(embedding_dim,512, self.num_classes)
        # self.deco4=small_decoder2(embedding_dim,512, self.num_classes)

        self.hypercorre_module=hypercorre_topk2(dim=self.in_channels, backbone=self.backbone)
        
        self.spatial_context_head = SpatialTemporalGather_Module(124)
        
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05)
        
        self.memory=[]
        
        # self.refine_block = nn.Sequential(
        #     # nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.LayerNorm(512)
        # )

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True
        
        # self.fusion_strategy = "sigmoid-do1"
        
        # if self.fusion_strategy in ["sigmoid-do1", "sigmoid-do2", "sigmoid-do3", "concat"]:

        #     dropprob = 0.3 if self.fusion_strategy in ["sigmoid-do1", "concat", "sigmoid-do3"] else 0.0
        #     self.dropout1 = nn.Dropout2d(dropprob)
        #     self.dropout2 = nn.Dropout2d(dropprob)
        #     self.bn1 = nn.BatchNorm2d(dim[3], eps=1e-03)
        #     self.bn2 = nn.BatchNorm2d(dim[3], eps=1e-03)

        # if self.fusion_strategy != "concat":
        #     self.conv_layer_1 = nn.Conv2d(dim[3],  2*dim[3], (3, 3), stride=1, padding=1, bias=True)
        #     self.conv_layer_2 = nn.Conv2d(dim[3],  2*dim[3], (3, 3), stride=1, padding=1, bias=True)
        #     if self.fusion_strategy == "sigmoid-do3":
        #         self.conv_layer_34 = nn.Conv2d(2*dim[3], 2*dim[3], (3, 3), stride=1, padding=1, bias=True)
        #     else:
        #         self.conv_layer_3 = nn.Conv2d(2*dim[3], 2*dim[3], (3, 3), stride=1, padding=1, bias=True)
        #         self.conv_layer_4 = nn.Conv2d(2*dim[3], 2*dim[3], (3, 3), stride=1, padding=1, bias=True)

        #     self.conv_down_sample1 = nn.Conv2d(2*dim[3], dim[3], (3, 3), stride=1, padding=1, bias=True)
        #     self.bn = nn.BatchNorm2d(2*dim[3], eps=1e-03)
        
        # self.num_feats_per_cls = 1
        
    
    def forward(self, mode, *args, **kwargs):
        if mode == 'init_memory':
            self.init_memory(*args, **kwargs)
        elif mode == 'update_memory':
            self.update_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.forward_features(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward_features(self,feats,segmentation,memorylist,inputs, batch_size=None, num_clips=None):
        #每一层特征下做down_sample,按通道cancat,每一次s*c->s*4c
        # print('a ',len(inputs))
        # print('b ',inputs[0].shape)
        
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        
        # print('shape2',[i.shape for i in inputs])
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        
        # print('c40 ',c4.shape)

        ############## MLP decoder on C1-C4 ###########
        # print(c4.shape)
        # torch.Size([2048, 15, 15])
        
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) 

        #downsample to c1(512,512)

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        # print('c41 ',c4.shape)
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        
        
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
        supp_c1,supp_c2,supp_c3,supp_c4 = supp_frame[0],supp_frame[1],supp_frame[2],supp_frame[3] 
        
        # query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        # query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # query_c1=self.sr1(query_c1)
        # query_c2=self.sr2(query_c2)
        # query_c3=self.sr3(query_c3)

        # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])

        # query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        # query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        
        # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])
        
        
        # if len(feats)>0:
        #     B,num_clips_select,cx,hx_supp,wx_supp = supp_c1.shape
        #     supp_frame_selected = supp_c1.permute(0,1,3,4,2)
        #     supp_frame_selected = self.linear1(supp_frame_selected.reshape(B,num_clips_select*hx_supp*wx_supp,cx)) #b,-1,cx
            
        #     B,cx,hx,wx=feats.shape 
        #     # B,cx=feats.shape 
            
        #     memory_feature = self.linear2(feats.permute(0,2,3,1).reshape(B,hx*wx,cx))
        #     # memory_feature = self.linear2(feats)
            
        #     atten = torch.matmul(supp_frame_selected,memory_feature.transpose(-1,-2)) #b,h,w,c c,b = b,h,w,b * b,c 

        #     #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
        #     # print('ss',supp_frame_selected.shape,memory_feature.transpose(-1,-2).shape)
        #     # print('ts',torch.matmul(atten,memory_feature).shape)
            
        #     supp_c1 = torch.matmul(atten,memory_feature).reshape(B,hx_supp,wx_supp,cx).permute(0,3,1,2)
        #     supp_c1 = supp_c1.unsqueeze(1)
        
        
        supp_frame = [supp_c1, supp_c2, supp_c3, supp_c4]
        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print('check1',[i.shape for i in query_frame])
        # print('check2',[i.shape for i in supp_frame])

        final_feature = self.hypercorre_module(query_frame,supp_frame) 
        

        ####生成attention——weight

        supp_feats = final_feature

        # atten=self.hypercorre_module(query_frame, supp_frame)
        # atten=F.softmax(atten,dim=-1)

        #(B,N,-1,C)
        # print("atten shape",atten.shape)
        # shape torch.Size([1, 3, 3600, 256])


        # pooling_c1 = self.pooling_mhsa_c1(c1[:,:-1],[8,16,24,32])
        # pooling_c2 = self.pooling_mhsa_c2(c2[:,:-1],[4,8,12,16])
        # pooling_c3 = self.pooling_mhsa_c3(c3[:,:-1],[2,4,6,8])  
        # pooling_c4 = self.pooling_mhsa_c4(c4[:,:-1],[1,2,3,4])  #<----

        # print("c1 c2",c1[:,:-1].shape,c2[:,:-1].shape,c3[:,:-1].shape,c4[:,:-1].shape)
        # torch.Size([1, 3, 64, 120, 120]) torch.Size([1, 3, 128, 60, 60]) torch.Size([1, 3, 320, 30, 30]) torch.Size([1, 3, 512, 15, 15])

        # print("pooling",pooling_c1.shape,pooling_c2.shape,pooling_c3.shape,pooling_c4.shape)
        # torch.Size([1, 3, 256, 64]) torch.Size([1, 3, 256, 128]) torch.Size([1, 3, 256, 320]) torch.Size([1, 3, 256, 512])

        # [torch.Size([2, 1, 64, 120, 120]), torch.Size([2, 1, 128, 60, 60]), torch.Size([2, 1, 320, 30, 30]), torch.Size([2, 1, 512, 15, 15])]

        # pooling_all_scale = torch.cat((pooling_c1,pooling_c2,pooling_c3,pooling_c4),dim=3) # B,N,-1,4C

        # print('scale',pooling_all_scale.shape)
        # torch.Size([1, 3, 256, 1024])

        # down_pooling_all_scale = self.pooling_linear(pooling_all_scale) #B,N,-1,C --- 1,4,130,256

        # down_pooling_all_scale = down_pooling_all_scale.reshape(batch_size,num_clips-1,-1,self.embeding)

        # down_pooling_all_scale = down_pooling_all_scale.transpose(-1,-2)
        

        h2=int(h/2)
        w2=int(w/2)
        # # h3,w3=shape_c3[-2], shape_c3[-1]
        
        
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        
        # print('c21',_c2.shape,batch_size,num_clips,h2,w2)
        
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)

        # # _c_further=_c2[:,:-1].reshape(batch_size, num_clips-1, -1, h3*w3)
        # _c3=self.sr1_feat(_c2)
        # _c3=_c3.reshape(batch_size, num_clips, -1, _c3.shape[-2]*_c3.shape[-1]).transpose(-2,-1)
        # # _c_further=_c3[:,:-1].reshape(batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], -1)    ## batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], c
        # _c_further=_c3[:,:-1]        ## batch_size, num_clips-1, _c2.shape[-2]*_c2.shape[-1], c
        # # print(_c_further.shape, topk_mask.shape, torch.unique(topk_mask.sum(2)))
        # _c_further=_c_further[topk_mask].reshape(batch_size,num_clips-1,-1,_c_further.shape[-1])    ## batch_size, num_clips-1, s, c
        # supp_feats=torch.matmul(atten,_c_further)#qk*v

        # from here !!!!!!

        # supp_feats=supp_feats.reshape(batch_size, (num_clips-1), self.embeding, -1)

        # for i in range(0,4):
        #     supp_feats[i]=supp_feats[i].transpose(-2,-1).reshape(batch_size,-1,self.embeding,h2,w2)

        supp_feats = [ supp.transpose(-2,-1).reshape(batch_size,-1, self.embeding, h2,w2) for supp in supp_feats ]
        
        new_supp = []

        for i in range(0,3):
            new_supp.append(torch.cat([supp_feats[i],_c2_split[:,i+1:i+2]],dim=2))

        supp_feats = new_supp

        supp_feats.insert(0,_c2_split[:,0])

        supp_feats=[ii.squeeze(1) for ii in supp_feats]

        outs=supp_feats
        
        if self.training:
            outs[1] = F.interpolate(input=outs[1], size=(h, w), mode='bilinear', align_corners=False)
            outs[2] = F.interpolate(input=outs[2], size=(h, w), mode='bilinear', align_corners=False)
            outs[3] = F.interpolate(input=outs[3], size=(h, w), mode='bilinear', align_corners=False)
        else:
            outs[1] = F.interpolate(input=outs[1], size=(h, w), mode='bilinear', align_corners=False)
            outs[2] = F.interpolate(input=outs[2], size=(h, w), mode='bilinear', align_corners=False)
            outs[3] = F.interpolate(input=outs[3], size=(h, w), mode='bilinear', align_corners=False)

        outs_feature = torch.cat(outs[1:],dim=0)
        
        
        # if not self.training:
        # if len(self.memory)!=0:
        #     # print("on memory",len(self.memory),id(self.memory),self.memory[0].shape)
        #     context = self.spatial_context_head(outs_feature,x[:,1:].squeeze(0),1,self.memory,8)
        # else:
        #     # print("empty memory")
        #     self.memory=[]
        context = self.spatial_context_head(outs_feature,x[:,1:].squeeze(0),1,memorylist,8)
        # else:
        #     context = self.spatial_context_head(outs_feature,x[:,1:].squeeze(0),1)
        
        # print("shape",outs_feature.shape,context.shape)
        # shape torch.Size([3, 512, 120, 120]) torch.Size([1, 512, 124, 1])
        
        outs_feat = self.spatial_ocr_head(outs_feature,context)
        
        b_s,cx_o,h_o,w_o = outs_feat.shape
        
        sub_feature = torch.split(outs_feat,1,dim=0)
        
        # print("sq",[i.shape for i in sub_feature],b_s)
        
        outs[1] = sub_feature[0]
        outs[2] = sub_feature[1]
        outs[3] = sub_feature[2]
        
        # print("sssqq",[i.shape for i in outs])
        
        # print('s ',[i.shape for i in outs])
        
        # ends here !!!!!!

        # print('p',c1[:,0].shape,c2[:,0].shape,c3[:,0].shape,c4[:,0].shape)

        # from here!!!

        # pooling_c1 = self.pooling_mhsa_c1(c1[:,0].unsqueeze(1),[8,16,24,32])
        # pooling_c2 = self.pooling_mhsa_c2(c2[:,0].unsqueeze(1),[4,8,12,16])
        # pooling_c3 = self.pooling_mhsa_c3(c3[:,0].unsqueeze(1),[2,4,6,8])  
        # pooling_c4 = self.pooling_mhsa_c4(c4[:,0].unsqueeze(1),[1,2,3,4])  #<----

        # pooling_ct = torch.cat([pooling_c1,pooling_c2,pooling_c3,pooling_c4],dim=3)

        # pooling_ct = self.pooling_linear(pooling_ct)

        # supp_feats.insert(0,pooling_ct)

        # outs = [i.squeeze(1).permute(0,2,1).reshape(1,self.embeding,1,-1) for i in supp_feats]

        # ends here!!!

        # supp_feats=supp_feats.transpose(-2,-1).reshape(batch_size,(num_clips-1),self.embeding,h2,w2)
        # supp_feats=(torch.chunk(supp_feats, (num_clips-1), dim=1)) #b,channel,-1 * (nums_clip-1)
        # supp_feats=[ii.squeeze(1) for ii in supp_feats]
        # supp_feats.append(_c2_split[:,-1])

        # supp_feats.append(pooling_c4[:,-1])

        # print("shape",[i.shape for i in supp_feats])
        
        # [torch.Size([1, 256, 225]), torch.Size([1, 256, 225]), torch.Size([1, 256, 225])]

        # supp_feats.append(_c2_split[:,-1]) #(batch_size, -1, h2, w2) 60*60

        # outs=[i.reshape(batch_size,self.embeding,h2,w2) for i in outs]

        # print([i.shape for i in outs])
        # [torch.Size([1, 256, 225]), torch.Size([1, 256, 225]), torch.Size([1, 256, 225])]
        
        out1=resize(self.sdeco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out2=resize(self.sdeco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out3=resize(self.sdeco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out4=resize(self.sdeco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        
        # outs_new=[]
        # if len(feats)>0:
        #     # 算label
        #     for supp_feats_i in supp_feats:
        #         batch_size, num_channels, h, w = supp_feats_i.size()
        #         # print("s",batch_size, num_channels, h, w)
        #         # extract the history features
        #         # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        #         new_x = x[:,-1]
                
        #         new_batch_size, new_num_channels, new_h, new_w = new_x.size()
                
        #         new_x = new_x.permute(0,2,3,1)
        #         weight_cls = new_x.reshape(-1, self.num_classes)
        #         weight_cls = F.softmax(weight_cls, dim=-1)
                
        #         # print("sss",weight_cls.shape,feats.shape)
                
        #         labels = weight_cls.argmax(-1).reshape(-1, 1)
        #         onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
        #         weight_cls = onehot
                    
        #         #算weight
        #         # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        #         selected_memory_list = []
        #         for idx in range(self.num_feats_per_cls):
        #             memory = feats[:, idx, :]
        #             # print("st",weight_cls.shape,memory.shape)
        #             selected_memory = torch.matmul(weight_cls, memory)
        #             selected_memory_list.append(selected_memory.unsqueeze(1))
                    
                    
        #         # print("sha ",[i.shape for i in selected_memory_list]) 
        #         #14400*124 * 124*64
                
        #         # calculate selected_memory according to the num_feats_per_cls
        #         #融合memory算输出
        #         if self.num_feats_per_cls > 1:
        #             relation_selected_memory_list = []
        #             for idx, selected_memory in enumerate(selected_memory_list):
        #                 # --(B*H*W, C) --> (B, H, W, C)
        #                 selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        #                 # --(B, H, W, C) --> (B, C, H, W)
        #                 selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        #                 # --append
        #                 relation_selected_memory_list.append(self.self_attentions[idx](supp_feats, selected_memory))
        #             # --concat
        #             selected_memory = torch.cat(relation_selected_memory_list, dim=1)
        #             selected_memory = self.fuse_memory_conv(selected_memory)
        #         else:
        #             assert len(selected_memory_list) == 1
        #             selected_memory = selected_memory_list[0].squeeze(1)
        #             # --(B*H*W, C) --> (B, H, W, C)
        #             selected_memory = selected_memory.view(new_batch_size, new_h, new_w, selected_memory.shape[-1])
        #             # --(B, H, W, C) --> (B, C, H, W)
        #             selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        #             # --feed into the self attention module
        #             # selected_memory = self.self_attention(feats, selected_memory)
                    
        #             # print("ttt",supp_feats_i.shape,selected_memory.shape)
        #             # selected_memory_atten = torch.matmul(supp_feats_i,selected_memory.transpose(-1,-2))
        #             # out_feats = torch.matmul(selected_memory_atten,selected_memory)
        #         memory_down_feat = F.interpolate(selected_memory,size=(h,w),mode='bilinear',align_corners=False)
        #         # print("sh",out_feats.shape,selected_memory.shape)
        #         # print("shape ",supp_feats_i.shape,memory_down_feat.shape)
        #         outs_new.append(torch.cat([supp_feats_i,memory_down_feat],dim=1))
        
        #     outs = outs_new
        
        #     _, _, h, w=_c.shape
        #     out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        
        # else:
        #     out1=resize(self.sdeco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out2=resize(self.sdeco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out3=resize(self.sdeco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        #     out4=resize(self.sdeco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        
        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w
        # output=torch.cat([x,out4],dim=1)


        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            # print('shape ',out4.squeeze(1).shape)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output
        # return output[:,-1:]

   
class small_decoder2(nn.Module):

    def __init__(self,
                 input_dim=256, hidden_dim=256, num_classes=124,dropout_ratio=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.smalldecoder=nn.Sequential(
            # ConvModule(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            # ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1)
            )
        # self.dropout=
        
    def forward(self, input):

        output=self.smalldecoder(input)

        return output