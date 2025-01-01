import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# from memory import *

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
        bsz_s, inch, hb, wb = x.size()
        out2 = self.conv2(x)

        return out2


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

    def forward(self,x,redu_ratios):

        B, N, C, hy, wy = x.shape

        pools = []

        x = x.reshape(-1,C,hy,wy)
        
        redu_ratios = [2,4,6,8]
        

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


        self.pooling_mhsa_sk4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_sk3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_sk2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_sk1 = pooling_mhsa(dim[3])


        self.pooling_mhsa_f4 = pooling_mhsa(dim[0])
        self.pooling_mhsa_f3 = pooling_mhsa(dim[1])
        self.pooling_mhsa_f2 = pooling_mhsa(dim[2])
        self.pooling_mhsa_f1 = pooling_mhsa(dim[3])
        

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

        
        dim_c = 1024
        
        self.pooling_proj_linear = nn.Linear(dim_c,256)
        self.pooling_proj_linear_1 = nn.Linear(dim_c,256)
        self.pooling_proj_linear_2 = nn.Linear(dim_c,256)
        self.pooling_proj_linear_3 = nn.Linear(dim_c,256)

        self.hpn=HPNLearner_topk2([1,1,1], backbone)
        self.hpn1=HPNLearner_topk2([1,1,1], backbone)
        self.hpn2=HPNLearner_topk2([1,1,1], backbone)

        # self.memory = FeatureMemory()

        self.linear1 = nn.Linear(dim[3], dim[3], bias=qkv_bias)
        self.linear2 = nn.Linear(512, 512, bias=qkv_bias)

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
        

        for ii, query in enumerate(query_frame):
            if ii==0:
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

            B,num_ref_clips,cx,hx,wx=query.shape
            # B,num_ref_clips,cx,hx,wx=query.shape

            query_qkv_all.append(query_qkv)
            # query_qkv_all.append(query.reshape(B,num_ref_clips,cx,hx,wx))       ## B,num_ref_clips,hy*wy,cx
            query_shape_all.append([hx,wx])
            # query_shape_all.append(wy)

        supp_qkv_all=[]
        supp_shape_all=[]
        
        for ii, supp in enumerate(supp_frame): #B,N,-1,C
            if ii==0:
                supp_qkv=self.q4(supp.permute(0,1,3,4,2)) #1 1 512 15 15
                # supp_qkv=self.q4(supp)
            elif ii==1: 
                supp_qkv=self.q3(supp.permute(0,1,3,4,2)) #1 1 320 30 30 
                # supp_qkv=self.q3(supp)
            elif ii==2:
                supp_qkv=self.q2(supp.permute(0,1,3,4,2)) #1 1 128 60 60
                # supp_qkv=self.q2(supp)
            elif ii==3:
                supp_qkv=self.q1(supp.permute(0,1,3,4,2)) #1 1 64 120 120


            B,num_ref_clips,cx,hx,wx=supp.shape

            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            supp_shape_all.append([hx,wx])
            # supp_shape_all.append([hx,wx])


        loca_selection=[]
        indices_sele=[]
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

        intermediate_weight_f1 = []
        intermediate_weight_f2 = []

        for ii in range(0,4): #according to ratio
            for fi in range(1,len(query_frame)-1):  #t-9 t-6 t-3 t
                if ii==0 and fi==1:
                    # print('q',query_frame[ii][:,fi-1:fi].shape)
                    last_feature = self.pooling_mhsa_fk1(query_frame[ii][:,fi-1:fi],[1,2,3,4])
                    last_feature_p = self.pk1(last_feature)
                    # p_features.append(last_feature_p)
                    # last_feature_fv1=self.pooling_mhsa_fv1(query_frame[ii][:,fi:fi+1],[1,2,3,4])
                    last_feature_v=self.f1p(last_feature)
                    last_features_cat6.append(last_feature_v)
                elif ii==1 and fi==1:
                    last_feature = self.pooling_mhsa_fk2(query_frame[ii][:,fi-1:fi],[2,4,6,8])
                    last_feature_p = self.pk2(last_feature)
                    # p_features.append(last_feature_p)
                    # last_feature_fv2=self.pooling_mhsa_fv2(query_frame[ii][:,fi:fi+1],[2,4,6,8])
                    last_feature_v=self.f2p(last_feature)
                    last_features_cat6.append(last_feature_v)
                elif ii==2 and fi==1:
                    last_feature = self.pooling_mhsa_fk3(query_frame[ii][:,fi-1:fi],[4,8,12,16])
                    last_feature_p = self.pk3(last_feature)
                    # p_features.append(last_feature_p)
                    # last_feature_fv3=self.pooling_mhsa_fv3(query_frame[ii][:,fi:fi+1],[4,8,12,16])
                    last_feature_v=self.f3p(last_feature)
                    last_features_cat6.append(last_feature_v)
                elif ii==3 and fi==1:
                    last_feature = self.pooling_mhsa_fk4(query_frame[ii][:,fi-1:fi],[8,16,24,32])
                    last_feature_p = self.pk4(last_feature)
                    # p_features.append(last_feature_p)
                    # last_feature_fv4=self.pooling_mhsa_fv4(query_frame[ii][:,fi:fi+1],[8,16,24,32])
                    last_feature_v=self.f4p(last_feature)
                    last_features_cat6.append(last_feature_v)

                step_atten = torch.matmul(query_qkv_all[ii][:,fi:fi+1],last_feature_p.transpose(2,3).unsqueeze(1)) #qk


                if fi == 1:
                    step_atten_1.append(step_atten)
                elif fi == 2:
                    step_atten_2.append(step_atten)


                if fi == 1:
                    qkv = torch.matmul(step_atten,last_feature_v.unsqueeze(1))
                else:
                    qkv = torch.matmul(step_atten,last_feature_v.unsqueeze(1))
                    # intermediate_weight_f2.append(qkv)
                
                if ii==0 and fi==1:
                    # print('qkv',qkv.permute(0,1,4,2,3).shape)
                    pooling_feature = self.pooling_mhsa_sk1(qkv.permute(0,1,4,2,3),[1,2,3,4])
                    last_feature_p = self.i1(pooling_feature) 
                    # pooling_fv1=self.pooling_mhsa_sv1(qkv.permute(0,1,4,2,3),[1,2,3,4]) 
                    last_feature_v = self.iv1(pooling_feature)
                    last_features_cat3.append(last_feature_v)  
                elif ii==1 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk2(qkv.permute(0,1,4,2,3),[2,4,6,8])
                    last_feature_p = self.i2(pooling_feature)
                    # pooling_fv2=self.pooling_mhsa_sv2(qkv.permute(0,1,4,2,3),[2,4,6,8]) 
                    last_feature_v = self.iv2(pooling_feature)
                    last_features_cat3.append(last_feature_v)
                elif ii==2 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk3(qkv.permute(0,1,4,2,3),[4,8,12,16])
                    last_feature_p = self.i3(pooling_feature)
                    # pooling_fv3=self.pooling_mhsa_sv3(qkv.permute(0,1,4,2,3),[4,8,12,16])  
                    last_feature_v = self.iv3(pooling_feature)
                    last_features_cat3.append(last_feature_v)
                elif ii==3 and fi==1:
                    pooling_feature = self.pooling_mhsa_sk4(qkv.permute(0,1,4,2,3),[8,16,24,32])
                    last_feature_p = self.i4(pooling_feature)
                    # pooling_fv4=self.pooling_mhsa_sv4(qkv.permute(0,1,4,2,3),[8,16,24,32])  
                    last_feature_v = self.iv4(pooling_feature)
                    last_features_cat3.append(last_feature_v)

            from_t.append(qkv)


        atten_feature_f1 = torch.cat([last_features_cat6[0],last_features_cat6[1],last_features_cat6[2],last_features_cat6[3]],dim=3)
        atten_feature_f2 = torch.cat([last_features_cat3[0],last_features_cat3[1],last_features_cat3[2],last_features_cat3[3]],dim=3)


        # from_t = [query_qkv_all[i][:,0:1] for i in range(4)]
        # 这里是t特征的融合
        pooling_f1 = self.pooling_mhsa_f1(from_t[0].permute(0,1,4,2,3),[1,2,3,4])
        pooling_f2 = self.pooling_mhsa_f2(from_t[1].permute(0,1,4,2,3),[2,4,6,8])
        pooling_f3 = self.pooling_mhsa_f3(from_t[2].permute(0,1,4,2,3),[4,8,12,16])
        pooling_f4 = self.pooling_mhsa_f4(from_t[3].permute(0,1,4,2,3),[8,16,24,32])

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

        # atten_all from here!!!

        atten_all=[]
        atten_all1=[]
        atten_all2=[]

        for ii in range(0,4):
            hx,wx=supp_shape_all[ii]

            # t帧和t-9做融合
            atten = torch.matmul(supp_qkv_all[ii],from_t_m[ii].transpose(2,3)) #q,t0

            hs = atten.shape[-1]

            final_feature_1 = step_atten_1[ii].reshape(B,1,-1,hs).permute(0,1,3,2).reshape(B*hs,hx,wx)
            final_feature_2 = step_atten_2[ii].reshape(B,1,-1,hs).permute(0,1,3,2).reshape(B*hs,hx,wx)
            final_feature = atten.permute(0,1,3,2).reshape(B*hs,hx,wx) # B*(num_clips-1)*s)*1*hx*wx

            atten_all.append(final_feature.unsqueeze(1))
            atten_all1.append(final_feature_1.unsqueeze(1))
            atten_all2.append(final_feature_2.unsqueeze(1))

        atten_all = self.hpn(atten_all)
        atten_all1 = self.hpn1(atten_all1)
        atten_all2 = self.hpn2(atten_all2)

        atten_all = atten_all.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)
        atten_all1 = atten_all1.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)
        atten_all2 = atten_all2.reshape(B,-1,supp_shape_all[2][0]*supp_shape_all[2][1]).unsqueeze(1).permute(0,1,3,2)

        # print(former_t_feature.shape,self.pooling_proj_linear)
        former_t_feature = self.pooling_proj_linear(former_t_feature)
        atten_feature_f1 = self.pooling_proj_linear_1(atten_feature_f1)
        atten_feature_f2 = self.pooling_proj_linear_2(atten_feature_f2)


        atten_weight = torch.matmul(atten_all,former_t_feature)
        atten_weight_1 = torch.matmul(atten_all1,atten_feature_f1)
        atten_weight_2 = torch.matmul(atten_all2,atten_feature_f2)

        
        return [atten_weight_1,atten_weight_2,atten_weight]