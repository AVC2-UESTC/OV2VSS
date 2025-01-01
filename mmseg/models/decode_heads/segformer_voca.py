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
# from .hyper_voca import hypercorre_topk2
from .hypercorre import hypercorre_topk2

from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
from torch.nn import functional as F

from .zero_shot_predictor import TransformerZeroshotPredictor
from .cat_zeroshot_classifier import CatClassifier
from .per_pixel import BasePixelDecoder
from .hyper_correlation import Corr

from einops import rearrange

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    
    result = result[:,:, : img_size[0], : img_size[1]]
    # print('ss',result.shape,img_size,output_height,output_height)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )
    return result

def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg

def semantic_inference2(self, mask_cls, mask_pred):
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    return semseg

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

    def forward(self,x,redu_ratios):

        B, N, C, hy, wy = x.shape

        pools = []

        x = x.reshape(-1,C,hy,wy)
        
        redu_ratios = [2,4,6,8]

        for (l,redu_ratio) in zip(self.d_convs,redu_ratios):

            pool = F.adaptive_avg_pool2d(x, (round(hy/redu_ratio), round(wy/redu_ratio)))

            pool = pool + l(pool) # fix backward bug in higher torch versions when training

            pools.append(pool.view(B*N, C, -1))
                
        pools = torch.cat(pools, dim=2)

        pools = self.norm(pools.permute(0,2,1))#B,-1,C

        pools = pools.reshape(B,N,-1,C)
        
        return pools
        
def label_to_onehot(gt, num_classes, ignore_index=-1):
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
    
class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c 
            gt_probs = F.normalize(gt_probs, p=1, dim=2)# batch x k x hw
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context               
        else:
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c 
            probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context

class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        q = rearrange(q,'b c h w->b (h w) c')
        k = rearrange(k,'b c h w->b (h w) c')
        v = rearrange(v,'b c h w->b (h w) c')
        # print(q.shape)
        B, N, C = q.shape
        B0, M0, C0 = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B0, M0, self.num_heads, C0 // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B0, M0, self.num_heads, C0 // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@HEADS.register_module()
class SegFormerHead_CAT(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_CAT, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.embeding = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_down_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_down_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_down_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_down_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.pooling_mhsa_c1 = pooling_mhsa(embedding_dim)
        self.pooling_mhsa_c2 = pooling_mhsa(embedding_dim)
        self.pooling_mhsa_c3 = pooling_mhsa(embedding_dim)
        self.pooling_mhsa_c4 = pooling_mhsa(embedding_dim)
        
        all_embed_dim = c1_in_channels + c2_in_channels + c3_in_channels +c4_in_channels

        self.pooling_linear = nn.Linear(embedding_dim*4,embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )


        self.hypercorre_module=hypercorre_topk2(dim=[256 for i in range(4)], backbone=self.backbone)

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
        self.predictor = CatClassifier()
        
        self.linearconv1 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        
        
        self.clip_classification = True
        self.ensembling = True
        self.ensembling_all_cls = True
        
        
        self.linearc2 = nn.Linear(c2_in_channels,self.embeding)
        self.corrconv1 = nn.Conv2d(self.embeding,self.embeding,kernel_size=(3,3))
        self.corrconv2 = nn.Conv2d(self.embeding,self.embeding,kernel_size=(3,3))
        self.actirelu = nn.ReLU()
        
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.embeding, self.embeding,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.embeding),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.embeding, self.num_seen_classes,kernel_size=1, stride=1, padding=0, bias=True))
        
        self.dsn_head = nn.Sequential(
            nn.Conv2d(self.embeding, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(512,self.num_seen_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        
        self.ocrn = SpatialGather_Module(self.num_seen_classes)
        self.mul_attention = MulitHeadAttention(dim=self.embeding)
        self.linear_ref = nn.Linear(1024,256)
        
        self.corr = Corr()
        
    @torch.no_grad() 
    def inference_stage(self,inputs,kernel=384, overlap=0.333, out_res=[640, 640]):
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)
        

    def forward(self,inputs, img,batch_size=None, num_clips=None):
        
        start_time=time.time()
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        
        
        c1, c2, c3, c4 = x
        

        ############## MLP decoder on C1-C4 ###########
        ratio = 1
            
        n, _, h, w = c1.shape
        
        h2=int(h/2)
        w2=int(w/2)
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) 

        _c_f = resize(_c,size=(h2,w2),mode='bilinear',align_corners=False)

        fuse_f = torch.split(_c_f,split_size_or_sections=ratio,dim=0)

        #downsample to c1(512,512)

        _, _, h, w=_c.shape

        self.type_inference = "patch"
        
        c4 = self.linear_down_c4(c4).reshape(-1,c4.shape[2], c4.shape[3],self.embeding).permute(0,3,1,2)
        c3 = self.linear_down_c3(c3).reshape(-1,c3.shape[2], c3.shape[3],self.embeding).permute(0,3,1,2)
        c2 = self.linear_down_c2(c2).reshape(-1,c2.shape[2], c2.shape[3],self.embeding).permute(0,3,1,2)
        c1 = self.linear_down_c1(c1).reshape(-1,c1.shape[2], c1.shape[3],self.embeding).permute(0,3,1,2)
        
        h2,w2 = c2.shape[2], c2.shape[3]
        
        if not self.training and num_clips!=self.num_clips:
            if self.type_inference == "patch":
                outputs =self.predictor(img,_c_f,c4)
                outputs = outputs.sigmoid()
                
                height,width = (480,480)
                
                output = sem_seg_postprocess(outputs, img.size()[-2:], height, width)
                processed_results = output[-1].unsqueeze(0)
                return processed_results
                
            
            output = sem_seg_postprocess(outputs, img.size()[-2:], height, width)
            processed_results = output[-1].unsqueeze(0)
            return processed_results
        
        full_c1 = c1.reshape(batch_size*ratio, num_clips, -1, c1.shape[-2], c1.shape[-1])
        full_c2 = c2.reshape(batch_size*ratio, num_clips, -1, c2.shape[-2], c2.shape[-1])
        full_c3 = c3.reshape(batch_size*ratio, num_clips, -1, c3.shape[-2], c3.shape[-1])
        full_c4 = c4.reshape(batch_size*ratio, num_clips, -1, c4.shape[-2], c4.shape[-1])
        
        ref_c1 = full_c1[:,:1]
        ref_c2 = full_c2[:,:1]
        ref_c3 = full_c3[:,:1]
        ref_c4 = full_c4[:,:1]
        
        
        ref_c1 = self.pooling_mhsa_c1(ref_c1,[8,16,24,32])
        ref_c2 = self.pooling_mhsa_c2(ref_c2,[4,8,12,16])
        ref_c3 = self.pooling_mhsa_c3(ref_c3,[2,4,6,8])
        ref_c4 = self.pooling_mhsa_c4(ref_c4,[1,2,3,4])

        
        ref_frame = torch.cat([ref_c1,ref_c2,ref_c3,ref_c4],dim=-1)
        ref_frame = ref_frame.squeeze(1)
        
        ref_frame = self.pooling_linear(ref_frame).unsqueeze(0).permute(0,3,1,2)
            
        c1=full_c1[-4:]
        c2=full_c2[-4:]
        c3=full_c3[-4:]
        c4=full_c4[-4:]
        
        
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        # # remove last frame

        query_frame=[query_c1, query_c2, query_c3, query_c4]

        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]

        final_feature = self.hypercorre_module(query_frame,supp_frame)  
        
        final_feature = [i.reshape(batch_size,h2,w2,self.embeding).permute(0,3,1,2) for i in final_feature]
        f_c1 = F.interpolate(fuse_f[0],size=(h2,w2),mode='bilinear',align_corners=False) 
        final_feature.insert(0,f_c1)
        
        feature_cat = torch.cat(final_feature,dim=0)
        
        aux_query_out = self.aux_head(feature_cat[-1].clone().unsqueeze(0))
        aux_query_out = F.interpolate(
            aux_query_out, size=(120, 120), mode="bilinear", align_corners=False
        )

        
        b0,c0,h0,w0 = feature_cat.shape
        
        ref_frame_dsn = self.dsn_head(ref_frame)
        ref_frame_ocr = self.ocrn(ref_frame,ref_frame_dsn)

        
        feature_cat_last= self.mul_attention(feature_cat[-1].clone().unsqueeze(0),ref_frame_ocr,ref_frame_ocr)
        
        feature_cat[-1] = feature_cat_last.reshape(1,c0,h0,w0)
        
        outputs =self.predictor(img[-4:],feature_cat,c4[:,-4:])
        # print(prediction.shape)

        if not self.training:
            if self.type_inference == "patch":
                outputs = outputs.sigmoid()
                
                height,width = (480,480)
                
                output = sem_seg_postprocess(outputs, img.size()[-2:], height, width)
                processed_results = output[-1].unsqueeze(0)
                return processed_results
            
            outputs = prediction.sigmoid()
            height,width = (480,480)

            output = sem_seg_postprocess(outputs, img.size()[-2:], height, width)

            processed_results = output[-1].unsqueeze(0)
            return processed_results
            
        return torch.cat([outputs.unsqueeze(0),aux_query_out.unsqueeze(0)],dim=1)
        
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg    
