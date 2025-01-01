import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

# from .ocr_module import SpatialTemporalGather_Module

class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        self.num_classes = 124
        self.num_feats_per_cls = 1
        self.feats_channels = 64
        
        num_classes=self.num_classes 
        num_feats_per_cls=self.num_feats_per_cls 
        feats_channels=self.feats_channels 
        
        # self.memory = nn.Parameter(torch.zeros([1,5,512,15,15]), requires_grad = False)
        self.memory = nn.Parameter(torch.zeros(num_classes,num_feats_per_cls,feats_channels,dtype=torch.float), requires_grad = False)
        
        self.linear1 = nn.Linear(512,512)
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,512)
        
        self.initlinear = nn.Linear(512,512)
        self.initlinear2 = nn.Linear(512,512)
        self.initlinear3 = nn.Linear(512,512)
        
        self.average_pooling = nn.AdaptiveAvgPool2d(2)
        
        # self.down_conv = nn.Conv2d(512 * 4, 512 , kernel_size=3, stride=1, padding=1, bias=False)
        
        # self.down_linear = nn.Linear(4*512,512)
        
    def set_memory(self,feats):
        self.memory.data=feats.squeeze(0)
        return self.memory.data
    
    def handle_squeeze(self,feats,segmentation,ignore_index=255):
        pass
        
    def init_memory(self,feats,segmentation,ignore_index=255):
        feats = feats[:,:1].squeeze(1)
        
        feats = F.interpolate(feats,size=(480,480),mode='bilinear',align_corners=False)
        
        batch_size, num_channels, h, w = feats.size()
        
        segmentation = segmentation.long()
        
        #print('shape',segmentation.shape,feats.shape)
        feats = feats.permute(0,2,3,1).contiguous()
        feats = feats.view(batch_size*h*w,num_channels)
        
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index : continue
            seg_cls = segmentation.view(-1)
            feats_cls = feats[seg_cls == clsid]
            
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
        
        return self.memory.data

    def update_memory_momentum(self,features,segmentation,ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        features = features[:,:1].squeeze(1)
        # print('f ',features.shape)
        
        features = F.interpolate(features,size=(480,480),mode='bilinear',align_corners=False)
        batch_size, num_channels, h, w = features.size()
        # momentum = momentum_cfg['base_momentum']
        # if momentum_cfg['adjust_by_learning_rate']:
        #     momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        
        momentum = 0.1
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue
            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1), 
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)
            
        return self.memory.data

    def update_memory(self,feats,preds):
        batch_size, num_channels, h, w = feats.size()
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)
        
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:,idx,:]
            selected_memory = torch.matmul(weight_cls,memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))
            
        return self.memory.data, memory_output
        # feats = self.handle_squeeze(feats)
        
        # B,cx,hx,wx = feats.shape
        
        # # print('feats ',feats.shape,hx,wx)
        # feats = feats.reshape(B,cx,hx*wx).permute(0,2,1)
        # # feats = feats.reshape(B,cx)
        
        # # print('f',feats.shape) #torch.Size([1, 512, 15, 27])
        # feats_k = self.linear1(feats)
        # feats_v = self.linear2(feats)
        
        # b_m,c_m,hx_m,wx_m = self.memory.data.shape
        # memory_f = self.memory.data.permute(0,2,3,1).reshape(b_m,-1,c_m)
        # memory_feature = self.linear3(memory_f)

        # # b_m,c_m = self.memory.data.shape
        # # memory_f = self.memory.data
        # # memory_feature = self.linear3(memory_f)

        # # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
        # atten = torch.matmul(memory_feature,feats_k.transpose(-1,-2)) #b,c b,c,h*w

        # # out = torch.matmul(atten,feats_v) #b,h*w b,h*w,c = b,c
        # out = torch.matmul(atten,feats_v).permute(0,2,1).reshape(B,cx,hx,wx) #b,h*w b,h*w,c = b,n,h*w,c
        
        # self.memory.data = out
        # return self.memory.data

    def forward(self,mode,feats,segmentation):
        # for u in feats:
        #     print(u.shape)
        feats = feats.unsqueeze(0)
        
        if mode == 'init_memory':
            return self.init_memory(feats,segmentation)#start of epoch,
        elif mode == 'update_memory':
            return self.update_memory_momentum(feats,segmentation)#each frame,todo:对齐元素大小
        elif mode == 'set_memory':
            return self.set_memory(feats)#start of iteration,todo:对齐元素大小即可