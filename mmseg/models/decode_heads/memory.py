import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

# from .ocr_module import SpatialTemporalGather_Module

# s [torch.Size([1, 3, 64, 120, 120]), torch.Size([1, 3, 128, 60, 60]), torch.Size([1, 3, 320, 30, 30]), torch.Size([1, 3, dim, 15, 15])]


class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        
        
        self.memory = nn.Parameter(torch.zeros([1,64,120,120]), requires_grad = False)
        # self.memory = nn.Parameter(torch.zeros(num_classes,num_feats_per_cls,feats_channels,dtype=torch.float), requires_grad = False)
        
        dim = 64
        
        self.linear1 = nn.Linear(dim,dim)
        self.linear2 = nn.Linear(dim,dim)
        self.linear3 = nn.Linear(dim,dim)
        
        self.initlinear = nn.Linear(dim,dim)
        self.initlinear2 = nn.Linear(dim,dim)
        self.initlinear3 = nn.Linear(dim,dim)
        
        # self.average_pooling = nn.AdaptiveAvgPool2d(2)
        
    def set_memory(self,feats):
        self.memory.data=feats
        return self.memory.data
    
    def handle_squeeze(self,feats):
        b,n,cx,hx,wx = feats.shape
        
        # feats = feats.reshape(b,n*cx,hx,wx)
        
        # feats = self.down_linear(feats.reshape(b,n*cx,hx,wx).permute(0,2,3,1)).permute(0,3,1,2)
        
        f_first = feats[:,:1]
        f_later = feats
        
        f_first = f_first.permute(0,1,3,4,2).reshape(b,1*hx*wx,cx)
        f_later = f_later.permute(0,1,3,4,2).reshape(b,n*hx*wx,cx)
        
        # print('f',f_first.shape)
        
        f_first_q = self.initlinear(f_first)
        f_later_k = self.initlinear2(f_later)
        f_later_v = self.initlinear3(f_later)
        
        feats_atten = torch.matmul(f_first_q,f_later_k.transpose(-1,-2)) #b,1*hx*wx,cx b,cx,n*hx*wx
        feats = torch.matmul(feats_atten,f_later_v) #b,hx*wx,n*hx*wx b,n*hx*wx,cx = b,hx*wx,cx
        
        feats = feats.reshape(b,hx,wx,cx).permute(0,3,1,2)
        
        # feats = torch.mean(feats.reshape(b,cx,-1),dim=2)
        # feats = self.average_pooling(feats)
        
        return feats
        
    def init_memory(self,feats):
        feats = self.handle_squeeze(feats)
        
        self.memory.data = feats
        return self.memory.data

    def update_memory(self,feats):
        feats = self.handle_squeeze(feats)
        
        # B,cx = feats.shape
        B,cx,hx,wx = feats.shape
        
        feats = feats.reshape(B,cx,hx*wx).permute(0,2,1)
        # feats = feats.reshape(B,cx)
        
        feats_k = self.linear1(feats)
        feats_v = self.linear2(feats)
        
        b_m,c_m,hx_m,wx_m = self.memory.data.shape
        memory_f = self.memory.data.permute(0,2,3,1).reshape(b_m,-1,c_m)
        memory_feature = self.linear3(memory_f)

        # b_m,c_m = self.memory.data.shape
        # memory_f = self.memory.data
        # memory_feature = self.linear3(memory_f)

        # torch.Size([1, 3, 225, dim]) torch.Size([1, 3, dim, 225])
        
        atten = torch.matmul(memory_feature,feats_k.transpose(-1,-2)) #b,c b,c,h*w

        # out = torch.matmul(atten,feats_v) #b,h*w b,h*w,c = b,c
        out = torch.matmul(atten,feats_v).permute(0,2,1).reshape(B,cx,hx_m,wx_m) #b,h*w b,h*w,c = b,n,h*w,c
        
        self.memory.data = out
        return self.memory.data

    def forward(self,mode,feats,segmentation):
        feats = feats.unsqueeze(0)
        
        if mode == 'init_memory':
            return self.init_memory(feats)#start of epoch,
        elif mode == 'update_memory':
            return self.update_memory(feats)#each frame,todo:对齐元素大小
        elif mode == 'set_memory':
            return self.set_memory(feats)#start of iteration,todo:对齐元素大小即可