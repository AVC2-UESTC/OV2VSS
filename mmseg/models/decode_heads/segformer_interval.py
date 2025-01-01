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
from .utils.utils import save_cluster_labels
import time
from torch.nn import functional as F
from .segformer_head import *
from .segformer_clip import *

from .memory import *

@HEADS.register_module()
class SegFormerHead_clips(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips, self).__init__(input_transform='multiple_select', **kwargs)
        self.net = SegFormerHead_clipsNet(feature_strides = feature_strides,**kwargs)
        # self.memory = FeatureMemory()
        # self.memorylist=[]
        

    def forward(self, inputs,batch_size, num_clips,gt_semantic_seg,memory):
        #每隔四帧做一次预测，每隔5帧交互生成一次memory_feature

        # if self.mode == 'TRAIN':
        # print('1len ',len(inputs[0]))
        
        num_infer = self.num_infer
        
        # print(inputs[0].shape)
        # n,c,_,_ = inputs[0].shape
        # torch.Size([4, 64, 120, 216])
        
        frame_len = inputs[0].shape[0]
        
        frame_gap = 2
        frame_gap_l = frame_gap*3+1
        
        out_to = []
        
        # print(frame_len,frame_gap_l)
        # 4 7
        
        # if self.training:
        #     for i in range(frame_len):
        #         if i == 0:
        #             if memory == None:
        #                 memoryFeature = self.memory(mode='init_memory',feats=inputs[-1][:5,:])
        #             else:
        #                 # print('mttttt',memory.shape)
        #                 memoryFeature = self.memory(mode='set_memory',feats=memory.squeeze(0))
        #             # print('ss1 ',i,memory==None,memoryFeature==None)
        #         elif i%num_infer==0 and i>num_infer:
        #             num_digit = i / 10
        #             num_decimal = i % 10
        #             carry = 1 if num_decimal>=5 else 0
        #             # print('qqqqq ',inputs[-1].shape,num_digit*10+carry*5-5,num_digit*10+carry*5)
        #             memoryFeature = self.memory(mode ='update_memory',feats=inputs[-1][int(num_digit*10+carry*5-5):int(num_digit*10+carry*5),:])
        #         # if memoryFeature != None:
        #         #     print(i,memoryFeature.shape)
        #         #     print('ss2 ',i,memory.shape)
        #         if i >= frame_gap_l:
        #             frame_in = []
        #             for sh in range(4):
        #                 frame_in.append(torch.stack([inputs[sh][q,:] for q in range(i-frame_gap_l,i,frame_gap)],dim=0))
        #             # print("memorysssssss",i,memoryFeature.shape)
        #             out = self.net(mode='segment',feats=memoryFeature,inputs=frame_in,batch_size=1,num_clips=4)
        #             # print('see ',i)
        #             out_to.append(out)
        # else:
        #     memoryFeature = []
        #     out = self.net(mode='segment',feats=memoryFeature,inputs=inputs,batch_size=1,num_clips=frame_len)
        #     out_to.append(out)
        
        # out = None
        former_frame = 0
        skip_frame = 0
        if memory != None:
            skip_frame = 2 #总长度-1-单次跑的帧
            former_frame = 8 #单次跑的帧-1
        
        memorylist = torch.split(memory,split_size_or_sections=1)
        memorylist = [i for i in memorylist]
        # print("shape12",type(memorylist),memorylist[0].shape)
        # exit()
        # 第一是可能出现overlap,第二是可能出现0-5帧的重复更新特征
        if not self.training:
            gt_semantic_seg = torch.empty(2,2)
        if frame_len < frame_gap_l:
            # if self.training:
            memoryFeature = []
            # else:
            #     memoryFeature = self.memory.memory.data
            out = self.net(mode='segment',feats=memoryFeature,memorylist=memorylist,inputs=inputs,batch_size=1,num_clips=frame_len,segmentation=gt_semantic_seg)
            return out,memoryFeature
        else:
            memoryFeature = None
            for i in range(frame_len):
                if i >= frame_gap_l-1:
                    frame_in = []
                    for sh in range(4):
                        frame_in.append(torch.stack([inputs[sh][q,:] for q in range(i-frame_gap_l+1,i+1,frame_gap)],dim=0))
                    # print('shape1',[i.shape for i in frame_in])
                    # print("segment",i,memoryFeature.shape)
                    out = self.net(mode='segment',feats=memoryFeature,memorylist=memorylist,inputs=frame_in,batch_size=1,num_clips=4,segmentation=gt_semantic_seg[:,:1])
                    # print('o2 ',out.shape,frame_len)
                    out_to.append(out)
            # for i in range(frame_len):
            #     if i == 0 and memory == None:
            #         memoryFeature = self.memory(mode='init_memory',feats=inputs[0][:num_infer,:],segmentation=gt_semantic_seg[:,:1])
            #         # print("init",memoryFeature.shape)
            #     elif ((i+skip_frame)==former_frame or i==0) and memory != None:
            #         memoryFeature = self.memory(mode='set_memory',feats=memory.squeeze(0),segmentation=gt_semantic_seg[:,:1])
            #         # print("set",memoryFeature.shape)
            #     if (i+skip_frame)%num_infer==0 and (i+skip_frame)>num_infer and (i+skip_frame)>former_frame:
            #         # num_digit = i / 10
            #         # num_decimal = i % 10
            #         # carry = 1 if num_decimal>=5 else 0
            #         # print('qqqqq ',i,frame_len,int(i/5-1)*5,int((i/5)*5))
            #         # print('sss',int((i/num_infer-1)*num_infer),int((i/num_infer)*num_infer))
            #         # print("update memory ",i)
            #         memoryFeature = self.memory(mode ='update_memory',feats=inputs[0][int((i/num_infer-1)*num_infer):int((i/num_infer)*num_infer),:],segmentation=gt_semantic_seg[:,:1])
            #         # print("update",memoryFeature.shape)
            #     # if memoryFeature != None:
            #     #     print(i,memoryFeature.shape)
            #     #     print('ss2 ',i,memory.shape)
            #     # print('test ',i,frame_len,frame_gap_l-1)
            #     if i >= frame_gap_l-1:
            #         frame_in = []
            #         for sh in range(4):
            #             frame_in.append(torch.stack([inputs[sh][q,:] for q in range(i-frame_gap_l+1,i+1,frame_gap)],dim=0))
            #         # print('shape1',[i.shape for i in frame_in])
            #         # print("segment",i,memoryFeature.shape)
            #         out = self.net(mode='segment',feats=memoryFeature,inputs=frame_in,batch_size=1,num_clips=4,segmentation=gt_semantic_seg[:,:1])
            #         # print('o2 ',out.shape,frame_len)
            #         out_to.append(out)
        
        if not self.training:
            out_to = [out_to[-1]]
        out = torch.cat(out_to,dim=1)
        
        # print(type(memorylist),len(memorylist))
        memorylist_back = [torch.tensor(i).to('cuda') for i in memorylist]
        memorylist_back = torch.cat(memorylist_back,dim=0)
        # print("mm",memorylist_back.shape)
        # exit()
        
        return out,memorylist_back
