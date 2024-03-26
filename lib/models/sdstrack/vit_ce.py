import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None, new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_multimodal = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.pos_embed_z_z_dte = nn.Parameter(torch.zeros(1, 2 * self.num_patches_template, embed_dim))
        self.pos_embed_x_x_dte = nn.Parameter(torch.zeros(1, 2 * self.num_patches_search, embed_dim))
        num_patches = self.num_patches_search + self.num_patches_template

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, infer=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        fine_rgb_x = x[:, :3, :, :]
        fine_rgb_z = z[:, :3, :, :]
        # depth thermal event images
        fine_dte_x = x[:, 3:, :, :]
        fine_dte_z = z[:, 3:, :, :]

        fine_rgb_z = self.patch_embed(fine_rgb_z)
        fine_rgb_x = self.patch_embed(fine_rgb_x)

        fine_dte_z = self.patch_embed_multimodal(fine_dte_z)
        fine_dte_x = self.patch_embed_multimodal(fine_dte_x)

        # introduce mask
        if not infer:
            masked_rgb_z, masked_dte_z = apply_mask(fine_rgb_z, fine_dte_z)
            masked_rgb_x, masked_dte_x = apply_mask(fine_rgb_x, fine_dte_x)
        else:
            masked_rgb_z = masked_dte_z = masked_rgb_x = masked_dte_x = None

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        fine_rgb_z += self.pos_embed_z
        fine_rgb_x += self.pos_embed_x
        fine_dte_z += self.pos_embed_z
        fine_dte_x += self.pos_embed_x

        masked_rgb_z = masked_rgb_z + self.pos_embed_z  if not infer else None
        masked_rgb_x = masked_rgb_x + self.pos_embed_x  if not infer else None
        masked_dte_z = masked_dte_z + self.pos_embed_z  if not infer else None
        masked_dte_x = masked_dte_x + self.pos_embed_x  if not infer else None

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        fine_rgb = combine_tokens(fine_rgb_z, fine_rgb_x, mode=self.cat_mode)
        fine_dte = combine_tokens(fine_dte_z, fine_dte_x, mode=self.cat_mode)
        masked_rgb = combine_tokens(masked_rgb_z, masked_rgb_x, mode=self.cat_mode) if not infer else None
        masked_dte = combine_tokens(masked_dte_z, masked_dte_x, mode=self.cat_mode) if not infer else None

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            x_dte = torch.cat([cls_tokens, x_dte], dim=1)

        fine_rgb = self.pos_drop(fine_rgb)
        fine_dte = self.pos_drop(fine_dte)
        masked_rgb = self.pos_drop(masked_rgb) if not infer else None
        masked_dte = self.pos_drop(masked_dte) if not infer else None

        x_dict = {'fine_rgb': fine_rgb, 'fine_dte': fine_dte, 'fine_rgb_dte': None,
                  'masked_rgb': masked_rgb, 'masked_dte': masked_dte, 'masked_rgb_dte': None}

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(fine_rgb.device)
        global_index_t_all = torch.linspace(0, 2*lens_z - 1, 2*lens_z, dtype=torch.int64).to(fine_rgb.device)
        global_index_t = global_index_t.repeat(B, 1)
        global_index_t_all = global_index_t_all.repeat(B, 1)
        global_index_t = {'fine_rgb': global_index_t, 'fine_dte': global_index_t, 'fine_rgb_dte': None,
                          'masked_rgb': global_index_t, 'masked_dte': global_index_t, 'masked_rgb_dte': None}

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(fine_rgb.device)
        global_index_s_all = torch.linspace(0, 2*lens_x - 1, 2*lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        global_index_s_all = global_index_s_all.repeat(B, 1)
        global_index_s = {'fine_rgb': global_index_s, 'fine_dte': global_index_s, 'fine_rgb_dte': None,
                          'masked_rgb': global_index_s, 'masked_dte': global_index_s, 'masked_rgb_dte': None}

        removed_indexes_s = {'fine_rgb': [], 'fine_dte': [], 'fine_rgb_dte': [],
                             'masked_rgb': [], 'masked_dte': [], 'masked_rgb_dte': []}
        
        distill_features = {'fine_condition':[], 'masked_condition':[]}
        interact = [3, 6, 9, 11]
        for i, blk in enumerate(self.blocks):
            for path in ['fine_', 'masked_']:
                if infer and path == 'masked_':
                    continue
                for sub_mode in ['rgb', 'dte', 'rgb_dte']:
                    mode = path + sub_mode
                    if 'rgb_dte' in mode:
                        if i not in interact:
                            continue
                        global_index_t[mode] = global_index_t_all
                        global_index_s[mode]  = global_index_s_all
                        x_dict[mode] = combine_tokens(x_dict[path + 'rgb'], x_dict[path + 'dte'], mode=self.cat_mode)
                        # avoid overfitting
                        mask_x = torch.zeros(x_dict[mode].shape[1], x_dict[mode].shape[1]).to(x_dict[mode])
                        mask_x[:x_dict[path + 'rgb'].shape[1], :x_dict[path + 'rgb'].shape[1]] = 1
                        mask_x[x_dict[path + 'rgb'].shape[1]:, x_dict[path + 'rgb'].shape[1]:] = 1
                        mask_x = mask_x == 1
                    else:
                        mask_x = None
                    x_dict[mode], global_index_t, global_index_s, removed_index_s, attn = \
                        blk(x_dict[mode], global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, path, sub_mode)

                    if 'rgb_dte' in mode:
                        # fusion results
                        len_x = global_index_t[path + 'rgb'].shape[1] + global_index_s[path + 'rgb'].shape[1]
                        len_x_dte = global_index_t[path + 'dte'].shape[1] + global_index_s[path + 'dte'].shape[1]
                        x_dict[path + 'rgb'] = x_dict[mode][:, :len_x, :]
                        x_dict[path + 'dte'] = x_dict[mode][:, len_x:, :]

                    if self.ce_loc is not None and i in self.ce_loc:
                        removed_indexes_s[mode].append(removed_index_s)

                if i in interact:
                    distill_features[path+'condition'].append(x_dict[path+'rgb_dte'])

        x_list = []
        aux_dict_list = []
        for path in ['fine_', 'masked_']:
            if infer and path == 'masked_':
                continue
            tmp_x = []
            for sub_mode in ['rgb', 'dte']:
                mode = path + sub_mode
                lens_x_new = global_index_s[mode].shape[1]
                lens_z_new = global_index_t[mode].shape[1]
                z = x_dict[mode][:, :lens_z_new]
                x = x_dict[mode][:, lens_z_new:]
                if removed_indexes_s[mode] and removed_indexes_s[mode][0] is not None:
                    removed_indexes_cat = torch.cat(removed_indexes_s[mode], dim=1)
                    pruned_lens_x = lens_x - lens_x_new
                    pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                    x = torch.cat([x, pad_x], dim=1)
                    index_all = torch.cat([global_index_s[mode], removed_indexes_cat], dim=1)
                    # recover original token order
                    C = x.shape[-1]
                    x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
                x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                # re-concatenate with the template, which may be further used by other modules
                x = torch.cat([z, x], dim=1)
                tmp_x.append(x)
            
            x = sum(tmp_x) / len(tmp_x)
            x = self.norm(x)

            aux_dict = {
                "attn": attn,
                "removed_indexes_s": removed_indexes_s[mode],
                "distill_features": distill_features
            }

            x_list.append(x)
            aux_dict_list.append(aux_dict)


        return x_list, aux_dict_list

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False, infer=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, infer=infer)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def apply_mask(rgb, dte):
    mask_rgb = 0.3
    mask_dte = 0.3

    rgb_prob = torch.rand(rgb.shape[:-1]).unsqueeze(-1).repeat(1, 1, rgb.shape[-1])
    rgb_mask = rgb_prob < mask_rgb

    masked_rgb = rgb.clone()
    masked_rgb[rgb_mask] = 0.0

    dte_prob = torch.rand(dte.shape[:-1]).unsqueeze(-1).repeat(1, 1, dte.shape[-1])
    dte_mask = ~rgb_mask & (dte_prob < mask_dte)

    masked_dte = dte.clone()
    masked_dte[dte_mask] = 0.0

    return masked_rgb, masked_dte

