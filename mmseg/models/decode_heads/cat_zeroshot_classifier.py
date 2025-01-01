import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .transformer.position_encoding import PositionEmbeddingSine
from .transformer.transformer import Transformer
from .third_party import clip
from .third_party import imagenet_templates

from einops import rearrange,repeat
import yaml
from .transformer.models import  Aggregator
import copy
import numpy as np

# from .hyper_correlation import Corr
from .hyper_correlation_copy import Corr
# from .hyper_co import Corr

from .third_party.model import QuickGELU
from timm.models.layers import trunc_normal_


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

class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
    ):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class VideoSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1,):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim//64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        # print(text.shape,visual.shape)
        b,d,t,c = text.shape
        visual = rearrange(visual,'b c h w-> b (h w) c')
        visual = repeat(visual,'b q c -> b d q c',d=d)
        text = text.reshape(b*d,t,c)
        visual = visual.reshape(-1,visual.shape[-2],visual.shape[-1])
        # print(text.shape,visual.shape)
        B, N, C = visual.shape
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        
        return self.alpha * text

class CatClassifier(nn.Module):
    def __init__(self,):
        super().__init__()
        
        hidden_dim = 256
        dropout = 0.1
        nheads = 8
        dim_feedforward = 2048
        enc_layers = 0
        dec_layers = 6
        pre_norm = False
        deep_supervision = True
        self.mask_classification = True
        self.wordvec = False
        train_class_json = "/root/vss/mmseg/handle_data/seen_classnames.json"
        test_class_json = "/root/vss/mmseg/handle_data/unseen_classnames.json"

        clip_pretrained = "ViT-B/16"
        prompt_ensemble_type = "imagenet"
        train_class_indexes_json = "datasets/coco/coco_stuff/split/seen_indexes.json"
        test_class_indexes_json = "datasets/coco/coco_stuff/split/seen_indexes.json"
        clip_classification = True
        mask_classification =True
        
        #not certain
        num_queries = 100
        in_channels = 256
        enforce_input_project = False
        temperature = 0.01
        
        text_dim=512
        text_guidance_proj_dim = 128
        appearance_guidance_dim =256
        appearance_guidance_proj_dim = 128
        decoder_dims = [64,32]
        decoder_guidance_dims = [256,128]
        decoder_guidance_proj_dims = [32,16]
        num_layers = 2
        num_heads = 4
        hidden_dims =128
        pooling_sizes = [2,2]
        feature_resolution = [24,24]
        window_sizes = 12
        attention_type = "linear"
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        # print(self.test_class_texts)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
        
        self.prompt_ensemble_type = prompt_ensemble_type
        
        assert "A photo of" not in self.class_texts[0]
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        prompt_templates_clip = imagenet_templates.IMAGENET_TEMPLATES_SELECT_CLIP
        
        self.clip_classification = clip_classification
        if self.clip_classification:
            self.clip_model = clip_model.float()
            self.clip_preprocess = clip_preprocess
        
        clip_finetune = "attention1"

          
        for name, params in self.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        
        self.text_features = zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0,2).float()
        self.text_features_test = zeroshot_classifier(self.test_class_texts, prompt_templates, clip_model).permute(1, 0,2).float()

        transformer = Aggregator(
            text_guidance_dim=text_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=8
            )
        self.transformer = transformer
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        
        self.corr = Corr()
        
        self.prompts_generator = VideoSpecificPrompt(layers=1, embed_dim=512, alpha=1e-4,)
        
        cfg={'PROMPT_LEARNER':"learnable",
                "PROMPT_DIM":512,
                "PROMPT_SHAPE":(16, 0),
                "PROMPT_CHECKPOINT":"",
                "CLIP_MODEL_NAME":"ViT-B/16"}
        
        
    def forward(self, ori_images,fuse_f,c4, images_tensor=None, ori_sizes=None):
        assert images_tensor == None
        
        ori_images = torch.cat([ori_images],dim=0)
        ori_images = F.interpolate(ori_images, size=self.clip_resolution, mode='bilinear', align_corners=False)

        clip_features = self.clip_model.encode_image(ori_images, dense=True)


        feature_resolution = [24,24]
        img_feat = rearrange(clip_features[:, 1:, :], "b (h w) c->b c h w", h=feature_resolution[0], w=feature_resolution[1])
        
        
        text = self.text_features if self.training else self.text_features_test
        text = text.repeat(img_feat.shape[0], 1, 1, 1)
        
        o_text = self.prompts_generator(text, img_feat)
        
        o_text = rearrange(o_text,'(b t) d c->b t d c',b=img_feat.shape[0])
        
        text = text + o_text
        
        out = self.corr(fuse_f,img_feat,c4,text)
        
        return out

def zeroshot_classifier(classnames, templates, clip_modelp):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize, shape: [48, 77]
            class_embeddings = clip_modelp.encode_text(texts)  # embed with text encoder
            # print('p',texts.shape,class_embeddings.shape)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
            class_embeddings /= class_embeddings.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
        