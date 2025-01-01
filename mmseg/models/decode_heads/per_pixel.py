import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from .transformer.position_encoding import PositionEmbeddingSine
from .transformer.transformer import TransformerEncoder, TransformerEncoderLayer

from mmcv.cnn import ConvModule,xavier_init

class BasePixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        norm = "SyncBN"
        lateral_convs = []
        output_convs = []
        conv_dim = 256
        mask_dim = 256
        
        use_bias = norm == ""
        
        features_channels = [64,128,320,512] 
        self.in_features = features_channels
        
        for idx,in_channels in enumerate(features_channels):
            if idx == len(features_channels)-1:
                # output_norm = nn.SyncBatchNorm(256)
                output_conv = ConvModule(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm_cfg=dict(type='SyncBN'),
                    act_cfg=dict(type='ReLU'),
                )
                # weight_init.c2_xavier_fill(output_conv)
                # xavier_init(output_conv,distribution='normal')
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                # lateral_norm = nn.SyncBatchNorm(256)
                # output_norm = nn.SyncBatchNorm(256)
                lateral_conv = ConvModule(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm_cfg=dict(type='SyncBN'),
                )
                output_conv = ConvModule(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm_cfg=dict(type='SyncBN'),
                    act_cfg=dict(type='ReLU'),
                )
                # xavier_init(lateral_conv,distribution='normal')
                # xavier_init(output_conv,distribution='normal')
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)
                
                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
                
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)
    
    def forward_features(self,features):
        for idx, f in enumerate(features[::-1]):
            x = features[idx]
            # print('f1',x.shape)
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), None
    
    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        # logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)

class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoderPixelDecoder(BasePixelDecoder):
    def __init__(self,):
        super().__init__()
        
        conv_dim = 256
        transformer_dropout = 0.1
        transformer_nheads = 8
        transformer_dim_feedforward = 2048
        transformer_enc_layers = 0
        transformer_pre_norm = False
        
        feature_channels = [64,128,320,512]
        self.in_features = feature_channels
        
        in_channels = feature_channels[len(feature_channels) - 1]
        
        self.input_proj = nn.Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        use_bias = False
        
        output_conv = ConvModule(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='ReLU'),
        )
        # xavier_init(output_conv,distribution='normal')
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv
        
        # weight_init.c2_xavier_fill(output_conv)
        
    def forward_features(self,features):
        for idx, f in enumerate(features):
            x = features[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features
    
    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        # logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)