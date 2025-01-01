import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from memory import *

class CenterPivotConv4d_half(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d_half, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
        #                        bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        ## x should be size of bsz*s, inch, hb, wb

        # if self.stride[2:][-1] > 1:
        #     out1 = self.prune(x)
        # else:
        #     out1 = x
        # bsz, inch, ha, wa, hb, wb = out1.size()
        # out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        # out1 = self.conv1(out1)
        # outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        # out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        # bsz, inch, ha, wa, hb, wb = x.size()
        bsz_s, inch, hb, wb = x.size()
        out2 = self.conv2(x)

        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        # out2 = self.conv2(out2)
        # outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        # out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        # if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
        #     out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
        #     out2 = out2.squeeze()

        # y = out1 + out2
        return out2

# [torch.Size([1, 1, 330, 512]), torch.Size([1, 1, 330, 320]), torch.Size([1, 1, 330, 128]), torch.Size([1, 1, 330, 64])]


# class Memory(nn.Module):
#     def __init__(self):
#         super(Memory, self).__init__()
#         self.memory_st = nn.Parameter(torch.zeros(1,4,330,512,dtype=torch.float), requires_grad=False)
#         # self.memory_st = []
#         # self.memory_st.append(nn.Parameter(torch.zeros(1,4,330,512,dtype=torch.float), requires_grad=False)) 
#         # self.memory_st.append(nn.Parameter(torch.zeros(1,4,330,320,dtype=torch.float), requires_grad=False)) 
#         # self.memory_st.append(nn.Parameter(torch.zeros(1,4,330,128,dtype=torch.float), requires_grad=False)) 
#         # self.memory_st.append(nn.Parameter(torch.zeros(1,4,330,64,dtype=torch.float), requires_grad=False)) 

#     def update(self,feature,ratio,step):
#         _,_,_,emd = feature.shape
#         self.memory_st[:,ratio:ratio+1,:,:emd] = ((step-1)/step) * self.memory_st[:,ratio:ratio+1,:,:emd] + (1/step) * feature
#         # self.memory_st[ratio] = ((step-1)/step) * self.memory_st[ratio] + (1/step) * feature

#     def get(self):
#         return self.memory_st

#     def forward(self, m_in,ratio,step):  
#         # in b,1,-1,h,w
#         # self.memory_st = self.update(m_in,)
#         self.memory_st[ratio] = ((step-1)/step) * self.memory_st[ratio] + (1/step) * m_in

#         return self.memory_st

class HPNLearner_topk2(nn.Module): 
    def __init__(self, inch, backbone):
        super(HPNLearner_topk2, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # ## new way for better trade-off between speed and performance
        if backbone=='b1':
            outch1, outch2, outch_final = 1,2,1
            self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
            self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
            self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

            # Mixing building blocks
            self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
            self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])
        else:
            outch1 = 1
            self.encoder_layer4 = make_building_block(inch[0], [outch1], [3], [1])
            self.encoder_layer3 = make_building_block(inch[1], [outch1], [5], [1])
            self.encoder_layer2 = make_building_block(inch[2], [outch1], [5], [1])

            # # Mixing building blocks
            self.encoder_layer4to3 = make_building_block(outch1, [1], [3], [1])
            self.encoder_layer3to2 = make_building_block(outch1, [1], [3], [1])


    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr


    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        return hypercorr_mix432



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


class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used
    Args:
    num_feats: number of features being used
    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4, backbone='b1'):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        # self.num_qkv=2
        # self.qkv_bias=qkv_bias
        # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        # self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        # self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.q1 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.q2 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.q3 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.q4 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.k1 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.k2 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.k3 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.k4 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.i4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.i3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.i2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.i1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.iv4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.iv3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.iv2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.iv1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.pooling_mhsa_fk4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_fk3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_fk2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_fk1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_fv4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_fv3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_fv2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_fv1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_sk4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_sk3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_sk2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_sk1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_sv4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_sv3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_sv2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_sv1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_f4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_f3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_f2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_f1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_fs4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_fs3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_fs2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_fs1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_fss4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_fss3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_fss2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_fss1 = pooling_mhsa(dim[3])

        self.pooling_mhsa_fsq4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_fsq3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_fsq2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_fsq1 = pooling_mhsa(dim[3])

        self.f4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.f3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.f2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.f1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.v4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.v3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.v2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.v1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.fs4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.fs3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.fs2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.fs1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.vs4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.vs3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.vs2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.vs1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.pk4 = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.pk3 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.pk2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.pk1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.f4p = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.f3p = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.f2p = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.f1p = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.f4pv = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.f3pv = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.f2pv = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.f1pv = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.f4psv = nn.Linear(dim[0], dim[0], bias=qkv_bias)
        self.f3psv = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.f2psv = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.f1psv = nn.Linear(dim[3], dim[3], bias=qkv_bias)
        
        self.pooling_proj_linear = nn.Linear(1024,256)
        self.pooling_proj_linear_1 = nn.Linear(1024,256)
        self.pooling_proj_linear_2 = nn.Linear(1024,256)
        self.pooling_proj_linear_3 = nn.Linear(1024,256)

        self.hpn=HPNLearner_topk2([1,1,1], backbone)
        self.hpn1=HPNLearner_topk2([1,1,1], backbone)
        self.hpn2=HPNLearner_topk2([1,1,1], backbone)

        self.memory = FeatureMemory()

        self.k_top=5
        if backbone=='b0':
            self.threh=0.8
        else:
            self.threh=0.5
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        [B*1*c*h/32*w/32, B*1*c*h/16*w/16, B*1*c*h/8*w/8, B*1*c*h/4*w/4]
        先pooling—ratio,ratio逐渐变大,生成s,
        4个features cancate,1/4 s*c
        Args:
            
        """

        start_time=time.time()
        query_frame=query_frame[::-1]
        supp_frame=supp_frame[::-1]
        query_qkv_all=[]
        query_shape_all=[]


        # print("ini_k",[i.shape for i in query_frame])
        for ii, query in enumerate(query_frame):
            if ii==0:
                # print('query cx1 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                # query=self.pooling_mhsa_k4(query,[1,2,3,4])
                query_qkv=self.k4(query.permute(0,1,3,4,2)) #1 3 512 15 15

                # query_key=self.k4(query.permute(0,1,3,4,2))
            elif ii==1:
                # print('query cx2 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                # query=self.pooling_mhsa_k3(query,[2,4,6,8])
                query_qkv=self.k3(query.permute(0,1,3,4,2)) #1 3 320 15 15

                # query_key=self.k3(query.permute(0,1,3,4,2))
            elif ii==2:
                # print('query cx3 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                # query=self.pooling_mhsa_k2(query,[4,8,12,16])
                query_qkv=self.k2(query.permute(0,1,3,4,2)) #1 3 128 15 15

                # query_key=self.k2(query.permute(0,1,3,4,2))
            elif ii==3:
                # query=self.pooling_mhsa_k1(query,[8,16,24,32])
                query_qkv=self.k1(query.permute(0,1,3,4,2)) #1 3 64 120 120

                # query_key=self.k1(query.permute(0,1,3,4,2))
                # print('query cx4 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                
                ## skip h/4*w/4 feature because it is too big
                # query_qkv_all.append(None)
                # query_shape_all.append([None,None])
                # continue
            B,num_ref_clips,cx,hx,wx=query.shape
            # B,num_ref_clips,cx,hx,wx=query.shape

            query_qkv_all.append(query_qkv)
            # query_qkv_all.append(query.reshape(B,num_ref_clips,cx,hx,wx))       ## B,num_ref_clips,hy*wy,cx
            query_shape_all.append([hx,wx])
            # query_shape_all.append(wy)

        # print("k",[i.shape for i in query_qkv_all])
        # [torch.Size([1, 3, 330, 512]), torch.Size([1, 3, 330, 320]), torch.Size([1, 3, 330, 128]), torch.Size([1, 3, 330, 64])] 
       
        supp_qkv_all=[]
        supp_shape_all=[]
        
        for ii, supp in enumerate(supp_frame): #B,N,-1,C
            if ii==0:
                # print('supp cx1 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                # supp=self.pooling_mhsa_k4(supp,[1/2,1/2,1/2,1/2])
                # supp=supp.permute(0,1,4,2,3)
                supp_qkv=self.q4(supp.permute(0,1,3,4,2)) #1 1 512 15 15
                # supp_qkv=self.q4(supp)
            elif ii==1: 
                # supp=self.pooling_mhsa_k3(supp,[1,1,1,1])
                # supp=supp.permute(0,1,4,2,3)
                # # print('supp cx2 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                supp_qkv=self.q3(supp.permute(0,1,3,4,2)) #1 1 320 30 30 
                # supp_qkv=self.q3(supp)
            elif ii==2:
                # supp=self.pooling_mhsa_k2(supp,[2,2,2,2])
                # supp=supp.permute(0,1,4,2,3)
                # # print('supp cx3 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                supp_qkv=self.q2(supp.permute(0,1,3,4,2)) #1 1 128 60 60
                # supp_qkv=self.q2(supp)
            elif ii==3:
                # print('supp cx4 %d %d %d %d %d'%(B,num_ref_clips,cx,hy,wy))
                # print("inner ",supp.shape)
                # supp=self.pooling_mhsa_k1(supp,[4,4,4,4])
                # supp=supp.permute(0,1,4,2,3)
                supp_qkv=self.q1(supp.permute(0,1,3,4,2)) #1 1 64 120 120
                # supp_qkv_all.append(None)
                # supp_shape_all.append([None,None])
                # continue
                # supp_qkv=self.q1(supp)

            B,num_ref_clips,cx,hx,wx=supp.shape
            # B,num_ref_clips,wx,cx=supp.shape


            # supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            # supp_shape_all.append([hx,wx])
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            supp_shape_all.append([hx,wx])
            # supp_shape_all.append([hx,wx])

        # print("q",[i.shape for i in supp_qkv_all])
        # [torch.Size([1, 1, 225, 512]), torch.Size([1, 1, 900, 320]), torch.Size([1, 1, 3600, 128]), torch.Size([1, 1, 14400, 64])]
        # a=input()

        # [torch.Size([1, 1, 330, 512]), torch.Size([1, 1, 330, 320]), torch.Size([1, 1, 330, 128]), torch.Size([1, 1, 330, 64])]

        loca_selection=[]
        indices_sele=[]
        # k_top=5
        # threh=0.5
        # print(threh,k_top)

        # query_qkv_all=query_qkv_all[::-1]
        # query_shape_all=query_shape_all[::-1]
        # supp_qkv_all=supp_qkv_all[::-1]
        # supp_shape_all=supp_shape_all[::-1]
        atten_all=[]
        s_all=[]

        B=supp_qkv_all[0].shape[0]
        q_num_ref=query_qkv_all[0].shape[1]

        atten_weight_perframe = []

        # store the features according to channel
        atten_feature_c1 = []
        atten_feature_c2 = []
        atten_feature_c3 = []
        atten_feature_c4 = []

        # TODO:Share weight between weights

        # start from here

        from_t = []
        atten_store_1 = []
        atten_store_2 = []
        last_feature = None
        step_atten_1 = []
        step_atten_2 = []

        last_features_cat6 = []
        last_features_cat3 = []

        p_features = []
        p1_features = []

        # last1_features = []
        # last1_features.append(self.pooling_mhsa_fsq1(query_frame[0][:,0:1],[1,2,3,4]))
        # last1_features.append(self.pooling_mhsa_fsq2(query_frame[1][:,0:1],[2,4,6,8]))
        # last1_features.append(self.pooling_mhsa_fsq3(query_frame[2][:,0:1],[4,8,12,16]))
        # last1_features.append(self.pooling_mhsa_fsq4(query_frame[3][:,0:1],[8,16,24,32]))

        # last1_features_cat = torch.cat(last1_features,dim=3)
        
        p1_features.append(self.pooling_mhsa_fsq1(query_frame[0][:,2:3],[1,2,3,4]))
        p1_features.append(self.pooling_mhsa_fsq2(query_frame[1][:,2:3],[2,4,6,8]))
        p1_features.append(self.pooling_mhsa_fsq3(query_frame[2][:,2:3],[4,8,12,16]))
        p1_features.append(self.pooling_mhsa_fsq4(query_frame[3][:,2:3],[8,16,24,32]))

        last3_features_cat = torch.cat(p1_features,dim=3)

        for ii in range(0,4): #according to ratio
            for fi in range(1,len(query_frame)-1): #according to frame t-3 t-6 t-9 1 3 6 
                if ii==0 and fi==1:
                    last_feature = self.pooling_mhsa_fk1(query_frame[ii][:,fi:fi+1],[1,2,3,4])
                    last_feature_p = self.pk1(last_feature)
                    p_features.append(last_feature_p)
                    # last_feature_fv1=self.pooling_mhsa_fv1(query_frame[ii][:,fi:fi+1],[1,2,3,4])
                    last_feature_f1=self.f1p(last_feature)
                    last_features_cat6.append(last_feature)
                elif ii==1 and fi==1:
                    last_feature = self.pooling_mhsa_fk2(query_frame[ii][:,fi:fi+1],[2,4,6,8])
                    last_feature_p = self.pk2(last_feature)
                    p_features.append(last_feature_p)
                    # last_feature_fv2=self.pooling_mhsa_fv2(query_frame[ii][:,fi:fi+1],[2,4,6,8])
                    last_feature_f2=self.f2p(last_feature)
                    last_features_cat6.append(last_feature)
                elif ii==2 and fi==1:
                    last_feature = self.pooling_mhsa_fk3(query_frame[ii][:,fi:fi+1],[4,8,12,16])
                    last_feature_p = self.pk3(last_feature)
                    p_features.append(last_feature_p)
                    # last_feature_fv3=self.pooling_mhsa_fv3(query_frame[ii][:,fi:fi+1],[4,8,12,16])
                    last_feature_f3=self.f3p(last_feature)
                    last_features_cat6.append(last_feature)
                elif ii==3 and fi==1:
                    last_feature = self.pooling_mhsa_fk4(query_frame[ii][:,fi:fi+1],[8,16,24,32])
                    last_feature_p = self.pk4(last_feature)
                    p_features.append(last_feature_p)
                    # last_feature_fv4=self.pooling_mhsa_fv4(query_frame[ii][:,fi:fi+1],[8,16,24,32])
                    last_feature_f4=self.f4p(last_feature)
                    last_features_cat6.append(last_feature)

                # print('qqqkkk',ii,fi,query_qkv_all[ii][:,fi-1:fi].shape,last_feature_p.transpose(2,3).shape)
                step_atten = torch.matmul(query_qkv_all[ii][:,fi-1:fi],last_feature_p.transpose(2,3)) #qk

                # store intermediate atten for hpn
                if fi == 1:
                    step_atten_1.append(step_atten)
                elif fi == 2:
                    step_atten_2.append(step_atten)


                if fi == 1:
                    if  ii == 0:
                        qkv = torch.matmul(step_atten,last_feature_f1)
                    elif ii == 1:
                        qkv = torch.matmul(step_atten,last_feature_f2)
                    elif ii == 2:
                        qkv = torch.matmul(step_atten,last_feature_f3)
                    else:
                        qkv = torch.matmul(step_atten,last_feature_f4)
                else:
                    qkv = torch.matmul(step_atten,last_feature_v)
                    

                if fi == 1:
                    atten_store_1.append(qkv)
                else:
                    atten_store_2.append(qkv)

                
                if ii==0 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk1(qkv.permute(0,1,4,2,3),[1,2,3,4])
                    last_feature_p = self.i1(pooling_feature) 
                    # pooling_fv1=self.pooling_mhsa_sv1(qkv.permute(0,1,4,2,3),[1,2,3,4]) 
                    last_feature_v = self.iv1(pooling_feature)
                    last_features_cat3.append(last_feature)  
                elif ii==1 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk2(qkv.permute(0,1,4,2,3),[2,4,6,8])
                    last_feature_p = self.i2(pooling_feature)
                    # pooling_fv2=self.pooling_mhsa_sv2(qkv.permute(0,1,4,2,3),[2,4,6,8]) 
                    last_feature_v = self.iv2(pooling_feature)
                    last_features_cat3.append(last_feature)
                elif ii==2 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk3(qkv.permute(0,1,4,2,3),[4,8,12,16])
                    last_feature_p = self.i3(pooling_feature)
                    # pooling_fv3=self.pooling_mhsa_sv3(qkv.permute(0,1,4,2,3),[4,8,12,16])  
                    last_feature_v = self.iv3(pooling_feature)
                    last_features_cat3.append(last_feature)
                elif ii==3 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk4(qkv.permute(0,1,4,2,3),[8,16,24,32])
                    last_feature_p = self.i4(pooling_feature)
                    # pooling_fv4=self.pooling_mhsa_sv4(qkv.permute(0,1,4,2,3),[8,16,24,32])  
                    last_feature_v = self.iv4(pooling_feature)
                    last_features_cat3.append(last_feature)

            from_t.append(qkv)

        # print('kq1',[i.shape for i in atten_store_1])
        # print('kq2',[i.shape for i in atten_store_2])
        # [torch.Size([1, 1, 15, 15, 512]), torch.Size([1, 1, 30, 30, 320]), torch.Size([1, 1, 60, 60, 128]), torch.Size([1, 1, 120, 120, 64])]

        # print('k1',[i.shape for i in atten_feature_c1])
        # print('k2',[i.shape for i in atten_feature_c2])
        # print('k3',[i.shape for i in atten_feature_c3])
        # print('k4',[i.shape for i in atten_feature_c4])


        # 这里是t-6特征的融合
        # pooling_fs1 = self.pooling_mhsa_fs1(last_features_cat6[0].permute(0,1,4,2,3),[1,2,3,4])
        # pooling_fs2 = self.pooling_mhsa_fs2(last_features_cat6[1].permute(0,1,4,2,3),[2,4,6,8])
        # pooling_fs3 = self.pooling_mhsa_fs3(last_features_cat6[2].permute(0,1,4,2,3),[4,8,12,16])
        # pooling_fs4 = self.pooling_mhsa_fs4(last_features_cat6[3].permute(0,1,4,2,3),[8,16,24,32])

        # pooling_fs1 = self.pooling_mhsa_fs1(atten_store_1[0].permute(0,1,4,2,3),[1,2,3,4])
        # pooling_fs2 = self.pooling_mhsa_fs2(atten_store_1[1].permute(0,1,4,2,3),[2,4,6,8])
        # pooling_fs3 = self.pooling_mhsa_fs3(atten_store_1[2].permute(0,1,4,2,3),[4,8,12,16])
        # pooling_fs4 = self.pooling_mhsa_fs4(atten_store_1[3].permute(0,1,4,2,3),[8,16,24,32])

        # pooling_fs1 = self.f1pv(pooling_fs1)
        # pooling_fs2 = self.f2pv(pooling_fs2)
        # pooling_fs3 = self.f3pv(pooling_fs3)
        # pooling_fs4 = self.f4pv(pooling_fs4)
        
        # # 这里是t-3特征的融合
        # pooling_fss1 = self.pooling_mhsa_fs1(last_features_cat3[0].permute(0,1,4,2,3),[1,2,3,4])
        # pooling_fss2 = self.pooling_mhsa_fs2(last_features_cat3[1].permute(0,1,4,2,3),[2,4,6,8])
        # pooling_fss3 = self.pooling_mhsa_fs3(last_features_cat3[2].permute(0,1,4,2,3),[4,8,12,16])
        # pooling_fss4 = self.pooling_mhsa_fs4(last_features_cat3[3].permute(0,1,4,2,3),[8,16,24,32])

        # pooling_fss1 = self.pooling_mhsa_fss1(atten_store_2[0].permute(0,1,4,2,3),[1,2,3,4])
        # pooling_fss2 = self.pooling_mhsa_fss2(atten_store_2[1].permute(0,1,4,2,3),[2,4,6,8])
        # pooling_fss3 = self.pooling_mhsa_fss3(atten_store_2[2].permute(0,1,4,2,3),[4,8,12,16])
        # pooling_fss4 = self.pooling_mhsa_fss4(atten_store_2[3].permute(0,1,4,2,3),[8,16,24,32])

        # pooling_fss1 = self.f1psv(pooling_fss1)
        # pooling_fss2 = self.f2psv(pooling_fss2)
        # pooling_fss3 = self.f3psv(pooling_fss3)
        # pooling_fss4 = self.f4psv(pooling_fss4)

        # print(pooling_fs1.shape,pooling_fs2.shape,pooling_fs3.shape,pooling_fs4.shape)

        # atten_feature_f1 = torch.cat([pooling_fs1,pooling_fs2,pooling_fs3,pooling_fs4],dim=3)
        # atten_feature_f2 = torch.cat([pooling_fss1,pooling_fss2,pooling_fss3,pooling_fss4],dim=3)

        atten_feature_f1 = torch.cat([last_features_cat6[0],last_features_cat6[1],last_features_cat6[2],last_features_cat6[3]],dim=3)
        atten_feature_f2 = torch.cat([last_features_cat3[0],last_features_cat3[1],last_features_cat3[2],last_features_cat3[3]],dim=3)


        # 这里是t特征的融合
        pooling_f1 = self.pooling_mhsa_f1(from_t[0].permute(0,1,4,2,3),[1,2,3,4])
        pooling_f2 = self.pooling_mhsa_f2(from_t[1].permute(0,1,4,2,3),[2,4,6,8])
        pooling_f3 = self.pooling_mhsa_f3(from_t[2].permute(0,1,4,2,3),[4,8,12,16])
        pooling_f4 = self.pooling_mhsa_f4(from_t[3].permute(0,1,4,2,3),[8,16,24,32])

        # pooling_t = [pooling_f1,pooling_f2,pooling_f3,pooling_f4]

        # st_final = []
        # for ii in range(0,4):
        #     atten_w = torch.matmul(last1_features[ii],pooling_t[ii].transpose(-1,-2))
        #     st_final.append(torch.matmul(atten_w,pooling_t[ii]))

        # pooling_f = st_final

        # pooling_f1=pooling_f[0]
        # pooling_f2=pooling_f[1]
        # pooling_f3=pooling_f[2]
        # pooling_f4=pooling_f[3]

        p_f1 = self.f1(pooling_f1)
        p_f2 = self.f2(pooling_f2)
        p_f3 = self.f3(pooling_f3)
        p_f4 = self.f4(pooling_f4)

        p_v1 = self.v1(pooling_f1)
        p_v2 = self.v2(pooling_f2)
        p_v3 = self.v3(pooling_f3)
        p_v4 = self.v4(pooling_f4)

        from_t_m = [p_f1,p_f2,p_f3,p_f4]

        from_t_v = [p_v1,p_v2,p_v3,p_v4]


        former_t_feature = torch.cat(from_t_v,dim=3)

        # from here

        # t = []

        # for i in range(0,4):
        #     qk = torch.matmul(from_t[i],from_t_m[i].transpose(2,3))
        #     qkv = torch.matmul(qk,from_t_v[i])
        #     t.append(qkv)

        # pooling_qk1 = self.pooling_mhsa_fss1(t[0].permute(0,1,4,2,3),[1,2,3,4])
        # pooling_qk2 = self.pooling_mhsa_fss2(t[1].permute(0,1,4,2,3),[2,4,6,8])
        # pooling_qk3 = self.pooling_mhsa_fss3(t[2].permute(0,1,4,2,3),[4,8,12,16])
        # pooling_qk4 = self.pooling_mhsa_fss4(t[3].permute(0,1,4,2,3),[8,16,24,32])

        # atten_feature_fs3 = torch.cat([pooling_qk1,pooling_qk2,pooling_qk3,pooling_qk4],dim=3)

        # atten_weight_1 = self.pooling_proj_linear(atten_feature_fs1)
        # atten_weight_2 = self.pooling_proj_linear_1(atten_feature_fs2)
        # atten_weight = self.pooling_proj_linear_1(atten_feature_fs3)

        # ends here

        # atten_all from here!!!

        atten_all=[]
        atten_all1=[]
        atten_all2=[]

        # print('ft',former_t_feature.shape)

        for ii in range(0,4):
            hx,wx=supp_shape_all[ii]

            # print("check",supp_qkv_all[ii].shape,from_t_m[ii].transpose(2,3).shape)
            # check torch.Size([1, 1, 225, 512]) torch.Size([1, 1, 512, 330])
            # check torch.Size([1, 1, 900, 320]) torch.Size([1, 1, 320, 330])
            # check torch.Size([1, 1, 3600, 128]) torch.Size([1, 1, 128, 330])
            # check torch.Size([1, 1, 14400, 64]) torch.Size([1, 1, 64, 330]) 14400*330 330*64


            # t帧和t-9做融合
            atten = torch.matmul(supp_qkv_all[ii],from_t_m[ii].transpose(2,3)) #q,t0
            # atten_add = torch.matmul(p_features[ii],)

            # atten = torch.matmul(from_t_m[ii],supp_qkv_all[ii].transpose(2,3)) #q,t0
            # atten1 = torch.matmul(pooling_fs[ii],supp_qkv_all[ii].transpose(2,3))
            # print('atten',atten.shape)
            # print('atten1',step_atten_1[ii].shape)

            hs = atten.shape[-1]

            final_feature_1 = step_atten_1[ii].reshape(B,1,-1,hs).permute(0,1,3,2).reshape(B*hs,hx,wx)
            final_feature_2 = step_atten_2[ii].reshape(B,1,-1,hs).permute(0,1,3,2).reshape(B*hs,hx,wx)
            final_feature = atten.permute(0,1,3,2).reshape(B*hs,hx,wx) # B*(num_clips-1)*s)*1*hx*wx

            atten_all.append(final_feature.unsqueeze(1))
            atten_all1.append(final_feature_1.unsqueeze(1))
            atten_all2.append(final_feature_2.unsqueeze(1))
            # atten_all1.append(final_feature1.unsqueeze(1))
            # final_m_feature = torch.matmul(atten,from_t[ii])
            # # with torch.no_grad():
            # #     self.memory.update(last_feature,ii,4)
            # after_t.append(final_m_feature)
        
        # print('kk',[a.shape for a in atten_all])
        # [torch.Size([330, 1, 15, 15]), torch.Size([330, 1, 30, 30]), torch.Size([330, 1, 60, 60]), torch.Size([330, 1, 120, 120])]

        atten_all = self.hpn(atten_all)
        atten_all1 = self.hpn1(atten_all1)
        atten_all2 = self.hpn2(atten_all2)

        atten_all = atten_all.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)
        atten_all1 = atten_all1.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)
        atten_all2 = atten_all2.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)

        former_t_feature = self.pooling_proj_linear(former_t_feature)
        atten_feature_f1 = self.pooling_proj_linear_1(atten_feature_f1)
        atten_feature_f2 = self.pooling_proj_linear_2(atten_feature_f2)

        # last3_features_cat = self.pooling_proj_linear_3(last3_features_cat) 

        # print('qq',atten_all.shape,former_t_feature.shape)
        # torch.Size([1, 1, 3600, 330]) torch.Size([1, 1, 330, 256]) --> [1,1,3600,256]

        atten_weight = torch.matmul(atten_all,former_t_feature)
        atten_weight_1 = torch.matmul(atten_all1,atten_feature_f1)
        atten_weight_2 = torch.matmul(atten_all2,atten_feature_f2)

        # print('a1 ',atten_weight.shape)

        # atten_weight_inter = torch.matmul(atten_weight,last3_features_cat.transpose(-1,-2))  
        # atten_weight = torch.matmul(atten_weight_inter,last3_features_cat)

        # print('a2 ',atten_weight.shape)

        # st_final = []
        # for ii in range(0,4):
        #     atten_w = torch.matmul(p_features[ii],pooling_t[ii].transpose(-1,-2))
        #     st_final.append(torch.matmul(atten_w,pooling_t[ii]))

        # pooling_f = st_final

        # atten ends here!!!
        

        # for ii in range(0,len(supp_frame)):
        #     fs=query_shape_all[ii]
        #     hx,wx=supp_shape_all[ii]
        #     # B,num_ref_clips,hx*wx,cx   B,num_ref_clips,cx,fs
        #     # print('supp-query',supp_qkv_all[ii].shape,query_qkv_all[ii].transpose(2,3).shape)
        #     atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3)) # B*(num_clips-1)*(hx*wx)*fs
        #     # s=atten.shape[3]
        #     # print("here: ", atten.shape, atten.max(), atten.min())

        #     atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*fs,hx,wx)   ## (B*(num_clips-1)*s)*hy*wy
        #     atten_all.append(atten.unsqueeze(1)) ## (B*(num_clips-1)*s)*1*hx*wx
        #     s_all.append(fs)

        # print('atten_all',[i.shape for i in atten_all])
        # # [torch.Size([990, 1, 15, 15]), torch.Size([990, 1, 30, 30]), torch.Size([990, 1, 60, 60]), torch.Size([990, 1, 120, 120])]


        # for ii in range(0,len(supp_frame)-1):
        #     hy,wy=query_shape_all[ii]
        #     hx,wx=supp_shape_all[ii]
        #     if ii==0:
        #         atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hy*wy)
        #         atten_fullmatrix=atten #1,3,225,225
        #         # print('inital atten',atten_fullmatrix.shape)
        #     else:
        #         cx=query_qkv_all[ii].shape[-1]
        #         query_selected=query_qkv_all[ii]  ## B*(num_clips-1)*(hy*wy)*c
        #         assert query_selected.shape[:-1]==loca_selection[ii-1].shape     
                
        #         num_selection=torch.unique((loca_selection[ii-1]).sum(2))
        #         assert num_selection.shape[0]==1 and num_selection.dim()==1
        #         # query_selected=torch.masked_select(querry_selected, loca_selection[ii-2].unsqueeze(-1))
        #         query_selected=query_selected[loca_selection[ii-1]>0.5].reshape(B, q_num_ref, int(num_selection[0]),cx)     ##  B*(num_clips-1)*s*c

        #         atten=torch.matmul(supp_qkv_all[ii], query_selected.transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(s)
        #         # atten_fullmatrix=-100*torch.ones(B,q_num_ref,atten.shape[2],(query_shape_all[ii][0]*query_shape_all[ii][1])).cuda()
        #         # indices=indices_sele[ii-1]
        #         # assert indices.shape[-1]==num_selection[0]
        #         # indices=indices.unsqueeze(2).expand(B,q_num_ref,atten.shape[2],indices.shape[-1])    ## B*(num_clips-1)*(hx*wx)*(s)
        #         # atten_fullmatrix=atten_fullmatrix.scatter(3,indices,atten)
        #     if ii==0:
        #         # atten_temp=atten.reshape(B*atten.shape[1],hx*wx,query_shape_all[ii][0],query_shape_all[ii][1])
        #         # atten_topk=F.interpolate()
        #         atten_topk=torch.topk(atten_fullmatrix,self.k_top,dim=2)[0]    # B*(num_clips-1)*(k)*(hy*wy)
        #         atten_topk=atten_topk.sum(2)   # B*(num_clips-1)*(hy*wy)
        #         # atten_kthvalue=torch.kthvalue(atten_topk,atten_topk.shape[-1]*threh,dim=2)[0]   # 
        #         # topk_mask=atten_topk>atten_kthvalue   # B*(num_clips-1)*(hy*wy)
        #         # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy, wy)  # B*(num_clips-1)*hy*wy
        #         # s=int(hy*wy*threh)
        #         # indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
        #         # topk_mask=torch.zeros_like(atten_topk)
        #         # topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
        #         # atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

        #         # topk_mask=topk_mask.reshape(B, q_num_ref, hy, wy)
                
        #         hy_next, wy_next=query_shape_all[ii+1]

        #         # print("hy next",hy_next,wy_next,hy,wy)

        #         if hy_next==hy and wy_next==wy:
        #             s=int(hy*wy*self.threh)
        #             indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
        #             topk_mask=torch.zeros_like(atten_topk)
        #             topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
        #             atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

        #         else:   # hy_next!=hy or wy_next!=wy
        #             atten=atten.reshape(B*q_num_ref,hx*wx,hy,wy)
        #             atten=F.interpolate(atten, (hy_next, wy_next), mode='bilinear', align_corners=False).reshape(B,q_num_ref,hx*wx,hy_next*wy_next)

        #             atten_topk=atten_topk.reshape(B,q_num_ref,hy,wy)    # B*(num_clips-1)*hy*wy
        #             atten_topk=F.interpolate(atten_topk, (hy_next, wy_next), mode='bilinear', align_corners=False)    # B*(num_clips-1)*hy_next*wy_next
        #             atten_topk=atten_topk.reshape(B, q_num_ref, hy_next*wy_next)   # B*(num_clips-1)*(hy_next*wy_next)

        #             s=int(hy_next*wy_next*self.threh)
        #             indices=torch.topk(atten_topk,s,dim=2)[1]    # # B*(num_clips-1)*s
        #             topk_mask=torch.zeros_like(atten_topk)
        #             topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy_next*wy_next)

        #             atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)
        #             # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy_next, wy_next)
        #         # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], query_shape_all[ii][0], query_shape_all[ii][1])  # B*(num_clips-1)*hy*wy
        #         # loca_selection[ii-1]=F.interpolate(topk_mask, query_shape_all[ii], mode='nearest', align_corners=False)>0.5
        #         loca_selection.append(topk_mask) ## B*(num_clips-1)*(hy*wy)
        #         # indices_sele[0]=indices
        #     elif ii<=len(supp_frame)-3:
        #         # print("iiiiiiii ",len(supp_frame)-3)
        #         loca_selection.append(loca_selection[-1])

        #     # print("len",len(loca_selection))
        #     # print("loca[0]",loca_selection[0].shape)

        #     s=atten.shape[3]
        #     # print("here: ", atten.shape, atten.max(), atten.min())
        #     atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*s,hx,wx)   ## (B*(num_clips-1)*s)*hx*wx
        #     atten_all.append(atten.unsqueeze(1))
        #     s_all.append(s)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        # print("atten",len(atten_all))
        # print("atten size",atten_all[0].shape)
        # print("s_all",len(s_all))
        # print("s_all size",s_all[0])


        # atten_all=self.hpn(atten_all)
        # print('atten',[i.shape for i in atten_all])
        # start_time2=time.time()
        # # B,num_ref_clips,_,_,_=query_frame[0].shape
        # # atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        # # atten_all=[atten_one.squeeze(1).reshape(B,q_num_ref,s_all[i],supp_shape_all.shape[0],supp_shape_all.shape[1]).permute(0,1,3,4,2) for i,atten_one in enumerate(atten_all)]
        # atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*fs
        
        # # return atten_all, loca_selection[-1]>0.5
        # return atten_all

        # according to t-6  t-3  t
        return [atten_weight_1,atten_weight_2,atten_weight]