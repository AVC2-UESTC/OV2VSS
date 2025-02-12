import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

from .cffm import BasicLayer3d3

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5):
        super().__init__()
        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 4)
        self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance)
        x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4)) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size)
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, _, _, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ensemble(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear') -> None:
        super().__init__()
        self.decoder_focal=BasicLayer3d3(dim=80,
               input_resolution=(24,
                                 24),
               depth=2,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               expand_layer="all",                           
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               use_layerscale=False, 
               layerscale_value=1e-4,
               focal_l_clips=[1,2,3],
               focal_kernel_clips=[5,5,5])
            #    focal_kernel_clips=[7,5,3])
        # self.image_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        # self.text_block = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size)

        self.pooling_new = nn.AdaptiveAvgPool2d((24, 24))
        # self.pooling_new =nn.Conv2d(208, 208, kernel_size=3, stride=1, padding=1)

        self.downlinear = nn.Linear(208,80,bias=True)

    def forward(self,x,img,text):
        # print('0',x.shape,img.shape,text.shape)
        # 2 torch.Size([4, 80, 81, 24, 24]) torch.Size([4, 128, 24, 24]) torch.Size([4, 81, 128])
        
        # x = self.image_block(x,img)
        # print('1',x.shape)
        # torch.Size([4, 80, 111, 24, 24])
        
        # x = self.decoder_focal(x,img)
        # x = rearrange(x,'b d c h w-> b c d h w')
        
        B, C,T, H, W = x.shape
        # x = rearrange(x, 'b c t h w -> b t h w c')
        # B0, D0, H0, W0, C = x.shape
        
        x=x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        
        img = repeat(img,'b c h w->(b t) c h w',t=T)
        
        # print('1',x.shape,img.shape)
        
        x = torch.cat([x,img],dim=1)
        
        x = self.pooling_new(x).permute(0,2,3,1)
        
        # print('2',x.shape,img.shape)
        
        x = self.downlinear(x)
        
        # print('3',x.shape,img.shape)
        
        x = x.permute(0,3,1,2).reshape(B,C,T,H,W)
        
        # print('4',x.shape)
        
        # x = self.text_block(x,text)
        return x

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(mid_channels // 16, mid_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels-guidance_channels, kernel_size=3, stride=3)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x,guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            # #print('llll',x.shape,guidance.shape)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)

class decoder(nn.Module):
    def __init__(self,
                 hidden_dim = 80):
        super().__init__()
        decoder_dims = [64]
        self.decoder1 = Up(hidden_dim, decoder_dims[0],decoder_dims[0])
        self.head = nn.Conv2d(decoder_dims[0], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x,guidance=None):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        # #print('q',corr_embed.shape)
        corr_embed = self.decoder1(corr_embed,guidance)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed

class Corr(nn.Module):
    def __init__(self,
                 prompt_channel = 80,
                #  hidden_dim = 128,
                 hidden_dim = 80,
                 num_feature_scale = 4,
                 text_guidance_dim = 512,
                 text_guidance_proj_dim = 128,
                #  appearance_guidance_feature= 256,
                 appearance_guidance_feature= 128,
                #  appearance_guidance_dim = 512,
                appearance_guidance_dim = 256,
                 appearance_guidance_dim_decoder = 64,
                 nheads = 4,
                 attention_type='linear',
                 pooling_size=(6, 6),
                 feature_resolution=(24, 24),
                 window_size=12,
                 num_layers = 4,
                 ) -> None:
        super().__init__()
        
        self.text_linear = nn.Linear(text_guidance_dim,text_guidance_proj_dim)
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None
        
        self.num_feature_scale = num_feature_scale
        self.num_frames = 4
        
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        
        self.layers = nn.ModuleList([
            ensemble(hidden_dim=hidden_dim,text_guidance_dim=text_guidance_proj_dim,appearance_guidance=appearance_guidance_feature,\
                nheads=nheads, input_resolution=feature_resolution,attention_type=attention_type,pooling_size=pooling_size,window_size=window_size) 
            for _ in range(num_layers)
        ])
        
        proj_guidance_img = 256
        self.proj_img_1 = nn.Sequential(
            nn.Conv2d(proj_guidance_img, appearance_guidance_dim_decoder, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_feature > 0 else None
        
        self.proj_img_c4 = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_feature, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        # self.proj_
        
        # self.decoder_guidance_projection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        # ]) if decoder_guidance_dims[0] > 0 else None
    
        # self.decoder1 = Up(hidden_dim, decoder_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1])
        # self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        
        self.decodeh1 = decoder()
        # self.decodeh2 = decoder()
        # self.decodeh3 = decoder()
        # self.decodeh4 = decoder()
    
    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        # #print('qq',img_feats.shape,text_feats.shape) 
        # 1,256,60,60 * 1,111,80,256 = 1,80,111,60,60
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    
    def corr_embed(self, x):
        B = x.shape[0]
        # #print('x',x.shape)
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W') #111 80 60 60
        
        corr_embed = self.conv1(corr_embed) #111 128 60 60
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B) # 1 128 111 60 60 
        return corr_embed
    
    def fusion(self,x):
        B = x.shape[0]
        for i in range(B-1,1,-1):
            # #print(x[i].shape,x[i-1].shape)
            q = rearrange(x[i],'P T H W -> P T (H W)')
            k = rearrange(x[i-1],'P T H W -> P T (H W)')
            
            # q = x[i].contiguous().reshape(x[i].shape[0],x[i].shape[1],-1)
            # k = x[i-1].contiguous().reshape(x[i-1].shape[0],x[i-1].shape[1],-1)
            
            # #print('atten1',q.shape,k.shape)
            atten = torch.matmul(q,k.transpose(-1,-2))
            # #print('atten',atten.shape,k.shape)
            fused = torch.matmul(atten,k).squeeze(0)
            x[i] = rearrange(fused,'P T (H W) -> P T H W',H=24)
            
        return x
        
    
    # def conv_decoder(self, x):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     #print('q',corr_embed.shape)
    #     corr_embed = self.decoder1(corr_embed)
    #     corr_embed = self.decoder2(corr_embed)
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    
    def forward(self,fuse_img,img,c4,text):
        
        # fuse_img是融合信息
        # img是未融合的 c4也是未融合的
        
        # text = self.text_linear(text)
        
        h,w = img[0].shape[-2],img[0].shape[-1]
        
        # for i in range(self.num_feature_scale):
        #     for j in range(self.num_frames):
        #         #print(img[i][j].shape,text.shape)
        #         if i==0:
        #             correlation_c1.append(self.correlation(img[i][j],text[j,:]))
        #         elif i==1:
        #             correlation_c2.append(self.correlation(img[i][j],text[j,:]))
        #         elif i==2:
        #             correlation_c3.append(self.correlation(img[i][j],text[j,:]))
        #         elif i==3:
        #             correlation_c4.append(self.correlation(img[i][j],text[j,:]))
        
        # #print(img[0].unsqueeze(0).shape,text[0,:].shape)
        
        # for i in range(self.num_feature_scale):
        #     # #print(img[i].shape,text.shape)
        #     if i==0:
        #         correlation_c1.append(self.corr_embed(self.correlation(img[i].unsqueeze(0),text[i,:])))
        
        # #print([i.shape for i in img],text.shape)
        # torch.Size([5, 256, 48, 48]) torch.Size([5, 124, 80, 256])
        
        # img = self.fusion(img)
        
        corr_emd_map = self.correlation(img,text)
        
        # corr_emd_map = self.fusion(corr_emd_map)
        
        corr_emd_map = self.corr_embed(corr_emd_map)
        # 5,128,111,48,48
        
        if self.text_guidance_projection is not None:
            text = text.mean(dim=-2)
            text = text / text.norm(dim=-1, keepdim=True)
            text = self.text_guidance_projection(text)
        
        if len(c4.shape) > 4:
            c4 = c4.squeeze(0)
        
        # c4 = self.fusion(c4)
        
        proj_c4 = self.proj_img_c4(c4)
        proj_c4 =F.interpolate(proj_c4,size=(24,24),mode='bilinear',align_corners=False)
        
        proj_fuse_img = self.proj_img_1(fuse_img)
        proj_fuse_img =F.interpolate(proj_fuse_img,size=(72,72),mode='bilinear',align_corners=False)
        
        # #print('ppp',proj_fuse_img.shape,proj_c4.shape)
        
        # corr_embeds = torch.cat([correlation_c1[0]],dim=0)
        # res_corr_map = []
        
        # correlation_c1[0].permute(2,3,)
        # #print(corr_embeds.shape,text.shape,correlation_c1[0].shape)
        # exit()
        
        # b,h*w,c * b,c,h*w = b,h*w,h*w b,h*w,c = b,h*w,c
        # b,c,h,w * b,t,c = b,c,t,h,w (t,h*w,c)*(t,c) = t,h*w,c
        # corr_emd_map = rearrange(corr_emd_map,'B C T H W -> (H W) (B T) C')
        
        # #print(corr_emd_map.shape,text.shape,)
        
        # supp_feats_1 = torch.matmul(correlation_c1[0],text[0,:])
        # supp_feats_1 = torch.einsum('b t c, t c -> b t c',corr_emd_map,text)
        # supp_feats_1 = rearrange(supp_feats_1,'(H W) T C -> C T H W',H=h).unsqueeze(0)
        
        #print('1',corr_emd_map.shape,proj_c4.shape,text.shape)
        for layer in self.layers:
            corr_embed = layer(corr_emd_map,proj_c4,text)
            
        # #print('cor',corr_embed.shape)
        # corr_embed = self.fusion(corr_embed)
        
        logit = F.interpolate(self.decodeh1(corr_embed,proj_fuse_img),size=(120,120),mode='bilinear',align_corners=False) 
        
        # for layer in self.layers:
        #     corr_embed = layer(corr_embeds,text)
            
        # correlation_map = torch.split(corr_embed,split_size_or_sections=1,dim=0)
        # logit1 = F.interpolate(self.decodeh1(correlation_map[0]),size=(120,120),mode='bilinear',align_corners=False) 
        # logit2 = F.interpolate(self.decodeh2(correlation_map[1]),size=(120,120),mode='bilinear',align_corners=False)
        # logit3 = F.interpolate(self.decodeh3(correlation_map[2]),size=(120,120),mode='bilinear',align_corners=False)
        # logit4 = F.interpolate(self.decodeh4(correlation_map[3]),size=(120,120),mode='bilinear',align_corners=False)
        
        # logit = torch.cat([logit1,logit2,logit3,logit4],dim=0)
        # # #print(img.shape,text.shape,corr.shape,corr_embed.shape,logit.shape)

        return logit
        
        # elif len(img)==4:
        #     correlation_c1 = []
        #     correlation_c2 = []
        #     correlation_c3 = []
        #     correlation_c4 = []
            
        #     text = self.text_linear(text)
            
        #     h,w = img[0].shape[-2],img[0].shape[-1]
            
        #     # for i in range(self.num_feature_scale):
        #     #     for j in range(self.num_frames):
        #     #         #print(img[i][j].shape,text.shape)
        #     #         if i==0:
        #     #             correlation_c1.append(self.correlation(img[i][j],text[j,:]))
        #     #         elif i==1:
        #     #             correlation_c2.append(self.correlation(img[i][j],text[j,:]))
        #     #         elif i==2:
        #     #             correlation_c3.append(self.correlation(img[i][j],text[j,:]))
        #     #         elif i==3:
        #     #             correlation_c4.append(self.correlation(img[i][j],text[j,:]))
            
        #     # #print(img[0].shape,text[0,:].shape)
            
        #     for i in range(self.num_feature_scale):
        #         # #print(img[i].shape,text.shape)
        #         if i==0:
        #             correlation_c1.append(self.corr_embed(self.correlation(img[i],text[i,:])))
        #         elif i==1:
        #             correlation_c2.append(self.corr_embed(self.correlation(img[i],text[i,:])))
        #         elif i==2:
        #             correlation_c3.append(self.corr_embed(self.correlation(img[i],text[i,:])))
        #         elif i==3:
        #             correlation_c4.append(self.corr_embed(self.correlation(img[i],text[i,:])))
            
        #     if self.text_guidance_projection is not None:
        #         text = text.mean(dim=-2)
        #         text = text / text.norm(dim=-1, keepdim=True)
        #         text = self.text_guidance_projection(text)
            
        #     corr_embeds = torch.cat([correlation_c1[0],correlation_c2[0],correlation_c3[0],correlation_c4[0]],dim=0)
            
        #     res_corr_map = []
            
        #     # correlation_c1[0].permute(2,3,)
        #     # #print(corr_embeds.shape,text.shape,correlation_c1[0].shape)
        #     # exit()
            
        #     # b,h*w,c * b,c,h*w = b,h*w,h*w b,h*w,c = b,h*w,c
        #     # b,c,h,w * b,t,c = b,c,t,h,w (t,h*w,c)*(t,c) = t,h*w,c
        #     correlation_c1[0] = rearrange(correlation_c1[0],'B C T H W -> (H W) (B T) C')
        #     correlation_c2[0] = rearrange(correlation_c2[0],'B C T H W -> (H W) (B T) C')
        #     correlation_c3[0] = rearrange(correlation_c3[0],'B C T H W -> (H W) (B T) C')
        #     correlation_c4[0] = rearrange(correlation_c4[0],'B C T H W -> (H W) (B T) C')
            
        #     # #print(correlation_c1[0].shape,text[0,:].shape,)
        #     # supp_feats_1 = torch.matmul(correlation_c1[0],text[0,:])
            
        #     supp_feats_1 = torch.einsum('b t c, t c -> b t c',correlation_c1[0],text[0,:])
        #     supp_feats_2 = torch.einsum('b t c, t c -> b t c',correlation_c2[0],text[0,:])
        #     supp_feats_3 = torch.einsum('b t c, t c -> b t c',correlation_c3[0],text[0,:])
        #     supp_feats_4 = torch.einsum('b t c, t c -> b t c',correlation_c4[0],text[0,:])
            
        #     # #print(supp_feats_1.shape)
            
        #     supp_feats_1 = rearrange(supp_feats_1,'(H W) T C -> C T H W',H=h).unsqueeze(0)
        #     supp_feats_2 = rearrange(supp_feats_2,'(H W) T C -> C T H W',H=h).unsqueeze(0)
        #     supp_feats_3 = rearrange(supp_feats_3,'(H W) T C -> C T H W',H=h).unsqueeze(0)
        #     supp_feats_4 = rearrange(supp_feats_4,'(H W) T C -> C T H W',H=h).unsqueeze(0)
            
        #     logit1 = F.interpolate(self.decodeh1(supp_feats_1),size=(120,120),mode='bilinear',align_corners=False) 
        #     logit2 = F.interpolate(self.decodeh2(supp_feats_2),size=(120,120),mode='bilinear',align_corners=False)
        #     logit3 = F.interpolate(self.decodeh3(supp_feats_3),size=(120,120),mode='bilinear',align_corners=False)
        #     logit4 = F.interpolate(self.decodeh4(supp_feats_4),size=(120,120),mode='bilinear',align_corners=False)
            
        #     logit = torch.cat([logit1,logit2,logit3,logit4],dim=0)
            
        #     # for layer in self.layers:
        #     #     corr_embed = layer(corr_embeds,text)
                
        #     # correlation_map = torch.split(corr_embed,split_size_or_sections=1,dim=0)
        #     # logit1 = F.interpolate(self.decodeh1(correlation_map[0]),size=(120,120),mode='bilinear',align_corners=False) 
        #     # logit2 = F.interpolate(self.decodeh2(correlation_map[1]),size=(120,120),mode='bilinear',align_corners=False)
        #     # logit3 = F.interpolate(self.decodeh3(correlation_map[2]),size=(120,120),mode='bilinear',align_corners=False)
        #     # logit4 = F.interpolate(self.decodeh4(correlation_map[3]),size=(120,120),mode='bilinear',align_corners=False)
            
        #     # logit = torch.cat([logit1,logit2,logit3,logit4],dim=0)
        #     # # #print(img.shape,text.shape,corr.shape,corr_embed.shape,logit.shape)

        #     return logit