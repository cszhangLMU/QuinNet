import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import matplotlib.pyplot as plt
import torchvision.models as models
import math
import numpy as np
import torch
from torch import nn
from utils import *
import torchvision.transforms.functional as TF
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
from PIL import Image


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


# def shift(dim):
#             x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
#             x_cat = torch.cat(x_shift, 1)
#             x_cat = torch.narrow(x_cat, 2, self.pad, H)
#             x_cat = torch.narrow(x_cat, 3, self.pad, W)
#             return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1) #切分  self.shift_size切分数   1为维度(通道)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))] # 进行位移 滚动移位
        x_cat = torch.cat(x_shift, 1) # 按照第1维度进行拼接（之前进行了切分(从通道维度切分的)，现在再拼接）
        x_cat = torch.narrow(x_cat, 2, self.pad, H) # 取出x_cat中第二维的数据从 self.pad 到 H
        x_s = torch.narrow(x_cat, 3, self.pad, W)# 取出x_cat中第3维的数据从 self.pad 到 H


        x_s = x_s.reshape(B,C,H*W).contiguous()#变形
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x




class FANBlock_SE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., sharpen_attn=False, use_se=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., sr_ratio=1., qk_scale=None, linear = False, downsample=None, c_head_num=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(dim, num_heads=num_heads, qkv_bias=qkv_bias, mlp_hidden_dim=int(dim * mlp_ratio), sharpen_attn=sharpen_attn,
                                    attn_drop=attn_drop, proj_drop=drop, drop=drop, drop_path=drop_path, sr_ratio=sr_ratio, linear = linear, emlp=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
        self.norm2 = norm_layer(dim)
        self.mlp = SEMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma1 = 1
        self.gamma2 = 1

    def forward(self, x, H: int, W: int, attn=None):
        x_new, _ = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)
        x_new, H, W= self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(self.gamma2 * x_new)
        return x
        
        # 通道注意力和mlp为串联结构
        #x_new, _ = self.attn(self.norm1(x), H, W)
        #x1 = x + self.drop_path(self.gamma1 * x_new)
        #x_new, H, W= self.mlp(self.norm2(x), H, W)
        #x2 = x + self.drop_path(self.gamma2 * x_new)
        #return x2
        # 让通道注意力和mlp 成为并列结构
        #x_new, _ = self.attn(self.norm1(x), H, W)
        #x3 = x + self.drop_path(self.gamma1 * x_new)
        #x_new, H, W = self.mlp(self.norm2(x), H, W)
        #x4 = x + self.drop_path(self.gamma2 * x_new)
        #x = torch.add(x3, x4)
        #x = torch.multiply(x, x2)
        #return x
##################################
#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
###################################
class TokenMixing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                    sr_ratio=1, linear=False, share_atten=False, drop_path=0., emlp=False, sharpen_attn=False,
                    mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.share_atten = share_atten
        self.emlp = emlp


        cha_sr = 1
        self.q = nn.Linear(dim, dim // cha_sr, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 // cha_sr, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear
        self.sr_ratio = sr_ratio
        #self.channelAttn = ChannelAttention(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, atten=None, return_attention=False):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # import pdb;pdb.set_trace()
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q * self.scale @ k.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn  @ v

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# twin attention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_heads=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                    sr_ratio=1, linear=False, share_atten=False, drop_path=0., emlp=False, sharpen_attn=False,
                    mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_heads = (dim // num_heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * num_heads

        self.heads = num_heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, H, W, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out, dots @ v


class SEMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False, use_se=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # import pdb; pdb.set_trace()
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = self.se(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, N).permute(0, 2, 1)
        return x, H, W
def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Pooling2D(nn.Module):
    def __init__(self, pool_size=3):
        super(Pooling2D, self).__init__()
        self.pool = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=pool_size,
                stride=1,
                padding=pool_size // 2,
                count_include_pad=False
            )

        )

    def forward(self, x):

        return self.pool(x)-x
################################################
#MLP_ASPP
# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class MLP_ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(MLP_ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.mlp =  Mlp(in_features=in_channels, hidden_features=int(out_channels * 1), act_layer=nn.GELU, drop=0.)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        res = []
        # for conv in self.convs:
        #     res.append(conv(x))

        for conv in self.convs:
            res.append(conv)
        res[4] = res[4](x)
        res[3] = res[3](x)
        res[2] = res[2](torch.add(x, res[3]))
        res[1] = res[1](torch.add(x, res[2]))
        res[0] = res[0](torch.add(x, res[1]))
        #print(len(res)) 一共五个分支

        res = torch.cat(res, dim=1)
        res = self.project(res)
        #print('1', res.shape) # 2  64 256 256
        B, C, H, W = res.shape
        #print(res.shape)
        res = torch.flatten(res, 2).transpose(1, 2)
        #print('2', res.shape) # 2 256*256  64
        #print(res.shape)
        t_res = self.mlp(self.norm(res), H, W)
        #print('3', t_res.shape) # 2 256*256  64
        res = torch.add(res, t_res)
        #print('4', res.shape) #
        #print(t_res.shape)
        res = res.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复之前的维度
        #print('5', res.shape)
        return res#self.project(res)


################################################
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
#MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x
############################################################################  
#空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)      
# 空洞卷积，设置输入通道，输出通道和膨胀率
class DilationConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilationConv, self).__init__()
        self.DilationConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.DilationConv(x)
##########################################
#ASPP
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
       
        
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
       
        
        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        #self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        #self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
       
        
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        ################################
        #guide_model
        #x_guide = self.guide_model(x)
        ################################
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out
        
        
        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)# 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        #print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out
        

        ### Bottleneck
        # 继续进行 MLP block
        out ,H,W= self.patch_embed4(out) #继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()# 恢复减半前的维度
        
        ###########################
        #out = self.bottom_aspp(out)
        #out = self.bottom_mlpaspp(out)
        #############################
        
        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))# 先改变通道数(从256->160)，再改变图像尺寸增大
        out = torch.add(out,t4) # 和对应大小的和通道的特征图 逐元素相加
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out) 
        
#################################################
#SACA-UNet
class SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.Attn_Block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.Attn_Block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.MIA = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        self.dAttn_Block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dAttn_Block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.Attn_Block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.Attn_Block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度

        ###########################
        # out = self.bottom_aspp(out)
        out = self.MIA(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大


        out = torch.add(out, t4)  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dAttn_Block1):
            out = blk(out, H, W)
        ### Stage 3
        #print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))



        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dAttn_Block2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
##################################################
#测试SACA_UNet其他方法
class Test_SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        ###################################################
        
        
        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####
        
        #使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0],16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])#第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])#第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])#第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])#第4个辅助编码器  
        
        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1],16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])#第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])#第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1]) #第4个辅助编码器
        
        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2],8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])#第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2]) #第4个辅助编码器
        
        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3],4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])#第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])
        self.Attdw2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is Test_SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        
        #################################没效果
        #self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        #)
        #########################################
        
        

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
    
        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
      
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        
        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0]) # 之前的
        #x11_0_b = self.dilationConv11_0(Images[0]) #修改的
        #x11_0 = torch.add(x11_0_a, x11_0_b)
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################
        
        
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        
        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1]) 
        x22_0 = self.conv22_0(x12_0)#之前的
        #x22_0_b = self.dilationConv22_0(x12_0)#修改的
        #x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################
        
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        
        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)
        x33_0 = self.conv33_0(x23_0)#原来
        #x33_0_b = self.dilationConv33_0(x23_0)# 新加
        #x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################
        
        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out
        
        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) # 原来
        #x44_0_b = self.dilationConv44_0(x34_0)# 新加
        #x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################

        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        #print('out.shape==========',out.shape)
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大


        out = torch.add(out, t4)  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        #print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))



        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
###############################################    
#对比不同分支数量 SACA_UNet其他方法
class MulEncode_SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        ###################################################
        
        
        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####
        
        #使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0],16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])#第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])#第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])#第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])#第4个辅助编码器  
        
        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1],16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])#第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])#第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1]) #第4个辅助编码器
        
        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2],8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])#第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2]) #第4个辅助编码器
        
        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3],4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])#第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])
        self.Attdw2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is MulEncode_SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        
        #################################没效果
        #self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        #)
        #########################################
        
        

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
    
        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
      
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        
        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0]) # 之前的
          #x11_0_b = self.dilationConv11_0(Images[0]) #修改的
          #x11_0 = torch.add(x11_0_a, x11_0_b)
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################
        
        
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        
        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1]) 
        x22_0_a = self.conv22_0(x12_0)#之前的
        x22_0_b = self.dilationConv22_0(x12_0)#修改的
        x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################
        
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        
        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)
        x33_0_a = self.conv33_0(x23_0)#原来
        x33_0_b = self.dilationConv33_0(x23_0)# 新加
        x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################
        
        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out
        
        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0_a = self.conv44_0(x34_0) # 原来
        x44_0_b = self.dilationConv44_0(x34_0)# 新加
        x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################

        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        #print('out.shape==========',out.shape)
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大


        out = torch.add(out, t4)  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        #print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))



        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
###############################################  
#skip 过滤器
class Filters(nn.Module):
    def __init__(self,channels):
        super(Filters, self).__init__()
        self.filters = nn.Sequential(
            nn.Conv2d(channels, int(channels), 1, bias=False),
            nn.BatchNorm2d(int(channels)),
            nn.ReLU()
            #nn.Conv2d(int(channels/2), channels, 1, bias=False),
            #nn.BatchNorm2d(channels),
            #nn.ReLU()
        )
    def forward(self, x):
        out = self.filters(x)
        return out#torch.add(x, out)
        
class DilationFilters(nn.Module):
    def __init__(self, channels):
        super(DilationFilters, self).__init__()
        self.dilationFilters = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.dilationFilters(x)
        return torch.add(x, out)
###############################################
# 第二个创新点（第二个小论文）多级并行结构使用的是卷积
class SkipConnect_SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################

        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####

        # 使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0], 16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])  # 第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])  # 第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])  # 第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  # 第4个辅助编码器

        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1], 16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第4个辅助编码器

        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2], 8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第4个辅助编码器

        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3], 4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])  # 第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])  # 可以在每个门控单元之后，再加一个Conv1×1的卷积
        self.Attdw2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        self.skipAttdw1 = AG_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0]) #没有使用
        self.skipAttdw2 = AG_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1]) #没有使用
        self.skipAttdw3 = AG_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2]) #没有使用
        self.skipAttdw4 = AG_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3]) #没有使用
        
        
        # AdvancedC3
        self.c3Filter1 = AdvancedC3(nb_filter[0], nb_filter[0])
        self.c3Filter2 = AdvancedC3(nb_filter[1], nb_filter[1])
        self.c3Filter3 = AdvancedC3(nb_filter[2], nb_filter[2])
        self.c3Filter4 = AdvancedC3(nb_filter[3], nb_filter[3])
        
        
        self.skipFilter1 = Filters(channels = nb_filter[0])
        self.skipFilter2 = Filters(channels = nb_filter[1])
        self.skipFilter3 = Filters(channels = nb_filter[2])
        self.skipFilter4 = Filters(channels = nb_filter[3])
        ####################################################

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is SkipConnect_SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #################################没效果
        # self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        # )
        #########################################

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        
        #这里修改了
        #for i in range(len(self.nb_filter) - 1):
        #    Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        ######################################
        crop = []
        #辅助编码器的图像进行数据增强
        for i in range(len(self.nb_filter) - 1):
            crop.append(transforms.RandomCrop(size=(int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i]))))
            Images.append(crop[i](x))
        ###########################################

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0]) # 之前的
          #x11_0_b = self.dilationConv11_0(Images[0]) #修改的
          #x11_0 = torch.add(x11_0_a, x11_0_b)
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1]) 
        x22_0 = self.conv22_0(x12_0)#之前的
        #x22_0_b = self.dilationConv22_0(x12_0)#修改的
        #x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)
        x33_0 = self.conv33_0(x23_0)#原来
        #x33_0_b = self.dilationConv33_0(x23_0)# 新加
        #x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) # 原来
        #x44_0_b = self.dilationConv44_0(x34_0)# 新加
        #x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
       
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大
        #print('t4================',t4.shape)
        out = torch.add(out, self.skipFilter4(t4))  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, self.skipFilter3(t3))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)


###############################################
#MPS的小的解码器，用来恢复多级并行结构上的特征，用来做监督
class MPSDecoder(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels):
        super().__init__()
        
        
        mid_channels = in_channels//2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
###############################################
###############################################
# 第二个创新点（第二个小论文）多级并行结构使用的是卷积，对多级并行结构的输入 使用PIL.resize(),不使用随机裁剪
class SkipConnect_SACA_UNet_resize(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################

        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####

        # 使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0], 16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])  # 第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])  # 第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])  # 第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  # 第4个辅助编码器

        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1], 16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第4个辅助编码器

        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2], 8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第4个辅助编码器

        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3], 4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])  # 第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])  # 可以在每个门口单元之后，再加一个Conv1×1的卷积
        self.Attdw2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        self.skipAttdw1 = AG_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0]) #没有使用
        self.skipAttdw2 = AG_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1]) #没有使用
        self.skipAttdw3 = AG_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2]) #没有使用
        self.skipAttdw4 = AG_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3]) #没有使用
        
        
        # AdvancedC3
        self.c3Filter1 = AdvancedC3(nb_filter[0], nb_filter[0])
        self.c3Filter2 = AdvancedC3(nb_filter[1], nb_filter[1])
        self.c3Filter3 = AdvancedC3(nb_filter[2], nb_filter[2])
        self.c3Filter4 = AdvancedC3(nb_filter[3], nb_filter[3])
        
        
        self.skipFilter1 = Filters(channels = nb_filter[0])
        self.skipFilter2 = Filters(channels = nb_filter[1])
        self.skipFilter3 = Filters(channels = nb_filter[2])
        self.skipFilter4 = Filters(channels = nb_filter[3])
        ####################################################

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is SkipConnect_SACA_UNet_resize")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #################################没效果
        # self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        # )
        #########################################

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        
        #这里修改了
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        ######################################
        #crop = []
        #辅助编码器的图像进行数据增强
        #for i in range(len(self.nb_filter) - 1):
        #    crop.append(transforms.RandomCrop(size=(int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i]))))
        #    Images.append(crop[i](x))
        ###########################################

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0]) # 之前的
          #x11_0_b = self.dilationConv11_0(Images[0]) #修改的
          #x11_0 = torch.add(x11_0_a, x11_0_b)
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1]) 
        x22_0 = self.conv22_0(x12_0)#之前的
        #x22_0_b = self.dilationConv22_0(x12_0)#修改的
        #x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)
        x33_0 = self.conv33_0(x23_0)#原来
        #x33_0_b = self.dilationConv33_0(x23_0)# 新加
        #x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) # 原来
        #x44_0_b = self.dilationConv44_0(x34_0)# 新加
        #x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
       
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大
        #print('t4================',t4.shape)
        out = torch.add(out, self.skipFilter4(t4))  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, self.skipFilter3(t3))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)


###############################################
###############################################
# 第二个创新点（第二个小论文）多级并行结构使用的是卷积，对多级并行结构的输入 先进行目标病灶区域裁剪，再resize
class SkipConnect_SACA_UNet_crop_resize(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################
        

        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####

        # 使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0], 16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])  # 第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])  # 第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])  # 第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  # 第4个辅助编码器

        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1], 16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第4个辅助编码器

        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2], 8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第4个辅助编码器

        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3], 4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])  # 第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])  # 可以在每个门口单元之后，再加一个Conv1×1的卷积
        self.Attdw2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        self.skipAttdw1 = AG_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0]) #没有使用
        self.skipAttdw2 = AG_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1]) #没有使用
        self.skipAttdw3 = AG_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2]) #没有使用
        self.skipAttdw4 = AG_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3]) #没有使用
        
        
        # AdvancedC3
        self.c3Filter1 = AdvancedC3(nb_filter[0], nb_filter[0])
        self.c3Filter2 = AdvancedC3(nb_filter[1], nb_filter[1])
        self.c3Filter3 = AdvancedC3(nb_filter[2], nb_filter[2])
        self.c3Filter4 = AdvancedC3(nb_filter[3], nb_filter[3])
        
        
        self.skipFilter1 = Filters(channels = nb_filter[0])
        self.skipFilter2 = Filters(channels = nb_filter[1])
        self.skipFilter3 = Filters(channels = nb_filter[2])
        self.skipFilter4 = Filters(channels = nb_filter[3])
        ####################################################

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is SkipConnect_SACA_UNet_crop_resize")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #################################没效果
        # self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        # )
        #########################################

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x,targetArea=None):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        ###################################################
        
        #对原图进行目标病灶区域的裁剪
        if targetArea != None:
            Images2 = []
            Images4 = []
            Images8 = []
            Images16 = []
            for i in range(image_size[0]):
                tempX = torchvision.transforms.functional.crop(x[i],targetArea[i][0],targetArea[i][1],targetArea[i][2]-targetArea[i][0],targetArea[i][3]-targetArea[i][1])
                Images2.append(TF.resize(tempX, size=[int(image_size[2] / 2), int(image_size[3] / 2)]))
                Images4.append(TF.resize(tempX, size=[int(image_size[2] / 4), int(image_size[3] / 4)]))
                Images8.append(TF.resize(tempX, size=[int(image_size[2] / 8), int(image_size[3] / 8)]))
                Images16.append(TF.resize(tempX, size=[int(image_size[2] / 16), int(image_size[3] / 16)]))
            Images2tensor = torch.stack(Images2,dim=0)
            Images4tensor = torch.stack(Images4,dim=0)
            Images8tensor = torch.stack(Images8,dim=0)
            Images16tensor = torch.stack(Images16,dim=0)
        else:
            Images2tensor = TF.resize(x, size=[int(image_size[2] / 2), int(image_size[3] / 2)])
            Images4tensor = TF.resize(x, size=[int(image_size[2] / 4), int(image_size[3] / 4)])
            Images8tensor = TF.resize(x, size=[int(image_size[2] / 8), int(image_size[3] / 8)])
            Images16tensor = TF.resize(x, size=[int(image_size[2] / 16), int(image_size[3] / 16)])
         ####################################################
        #这里修改了
        #for i in range(len(self.nb_filter) - 1):
        #    Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        
       
        
       
        
        
        
        ######################################
        #crop = []
        #辅助编码器的图像进行数据增强
        #for i in range(len(self.nb_filter) - 1):
        #    crop.append(transforms.RandomCrop(size=(int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i]))))
        #    Images.append(crop[i](x))
        ###########################################

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images2tensor) # 之前的
          #x11_0_b = self.dilationConv11_0(Images[0]) #修改的
          #x11_0 = torch.add(x11_0_a, x11_0_b)
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images4tensor) 
        x22_0 = self.conv22_0(x12_0)#之前的
        #x22_0_b = self.dilationConv22_0(x12_0)#修改的
        #x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images8tensor)
        x23_0 = self.conv23_0(x13_0)
        x33_0 = self.conv33_0(x23_0)#原来
        #x33_0_b = self.dilationConv33_0(x23_0)# 新加
        #x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images16tensor)
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) # 原来
        #x44_0_b = self.dilationConv44_0(x34_0)# 新加
        #x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
       
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大
        #print('t4================',t4.shape)
        out = torch.add(out, self.skipFilter4(t4))  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, self.skipFilter3(t3))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)


###############################################
################################################
#多级并行结构中使用的注意力  这个注意力用在了多级并行结构中
class MPSAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = 1
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(in_channels, out_channels, kernel_size, stride_q, padding_q, bias=attention_bias)
        self.layernorm_q = nn.LayerNorm(out_channels, eps=1e-5)
        self.conv_k = nn.Conv2d(in_channels, out_channels, kernel_size, stride_kv, stride_kv, bias=attention_bias)
        self.layernorm_k = nn.LayerNorm(out_channels, eps=1e-5)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size, stride_kv, stride_kv, bias=attention_bias)
        self.layernorm_v = nn.LayerNorm(out_channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2] * q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2] * k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2] * v.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1
###############################################
###############################################
# 第二个创新点（第二个小论文）多级并行结构使用的是注意力
class QuinNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################

        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####
        
        ############################
        
        #多级并行结构的四个小的解码器
        #第一个小解码器
        self.mpsdecoder1 = MPSDecoder(in_channels = nb_filter[0])#未使用
        self.mpsdecoder1_1 = DoubleConv(nb_filter[0],1)
        #第二个小解码器
        self.mpsdecoder2 = MPSDecoder(in_channels = nb_filter[1])#未使用
        self.mpsdecoder2_1 = DoubleConv(nb_filter[1],nb_filter[0])
        self.mpsdecoder2_2 = DoubleConv(nb_filter[0],1)
        #第三个小解码器
        self.mpsdecoder3 = MPSDecoder(in_channels = nb_filter[2])#未使用
        self.mpsdecoder3_1 = DoubleConv(nb_filter[2],nb_filter[1])
        self.mpsdecoder3_2 = DoubleConv(nb_filter[1],nb_filter[0])
        self.mpsdecoder3_3 = DoubleConv(nb_filter[0],1)
        #第四个小解码器
        self.mpsdecoder4 = MPSDecoder(in_channels = nb_filter[3])#未使用
        self.mpsdecoder4_1 = DoubleConv(nb_filter[3],nb_filter[2])
        self.mpsdecoder4_2 = DoubleConv(nb_filter[2],nb_filter[1])
        self.mpsdecoder4_3 = DoubleConv(nb_filter[1],nb_filter[0])
        self.mpsdecoder4_4 = DoubleConv(nb_filter[0],1)
        ##############################
        

        # 使用到不同尺度的图像的编码卷积
        #Wide_Focus   DoubleConv  MPSAttention
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0], 16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])  # 第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])  # 第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])  # 第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  # 第4个辅助编码器

        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1], 16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第4个辅助编码器

        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2], 8)
        self.conv33_0 = MPSAttention(nb_filter[1], nb_filter[2])  # 第3个辅助编码器
        self.conv34_0 = MPSAttention(nb_filter[1], nb_filter[2])  # 第4个辅助编码器

        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3], 4)
        self.conv44_0 = Wide_Focus(nb_filter[2], nb_filter[3])  # 第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.TSF1 = C3_MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])  # 可以在每个门口单元之后，再加一个Conv1×1的卷积
        self.TSF2 = C3_MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.TSF3 = C3_MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.TSF4 = C3_MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        self.skipAttdw1 = AG_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0]) #没有使用
        self.skipAttdw2 = AG_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1]) #没有使用
        self.skipAttdw3 = AG_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2]) #没有使用
        self.skipAttdw4 = AG_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3]) #没有使用
        
        
        # AdvancedC3
        self.c3Filter1 = AdvancedC3(nb_filter[0], nb_filter[0])
        self.c3Filter2 = AdvancedC3(nb_filter[1], nb_filter[1])
        self.c3Filter3 = AdvancedC3(nb_filter[2], nb_filter[2])
        self.c3Filter4 = AdvancedC3(nb_filter[3], nb_filter[3])
        
        
        self.skipFilter1 = Filters(channels = nb_filter[0])
        self.skipFilter2 = Filters(channels = nb_filter[1])
        self.skipFilter3 = Filters(channels = nb_filter[2])
        self.skipFilter4 = Filters(channels = nb_filter[3])
        ####################################################

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is QuinNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.Attn_block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.Attn_block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.MIA = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #################################没效果
        # self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        # )
        #########################################

        self.dAttn_block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dAttn_block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        
        #这里修改了
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        
        
        
        
        ######################################
        #crop = []
        #辅助编码器的图像进行数据增强
        #for i in range(len(self.nb_filter) - 1):
        #    crop.append(transforms.RandomCrop(size=(int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i]))))
        #    Images.append(crop[i](x))
        ###########################################

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0]) # 之前的
        x1_0att = self.TSF1(g=x11_0, x=t1)
        
        
        #&&&&用来做监督
        #mspOutput1 = self.mpsdecoder1(x1_0att)
        mspOutput1 = F.relu(F.interpolate(self.mpsdecoder1_1(x1_0att), scale_factor=(2, 2), mode='bilinear'))
        #&&&&
        
        
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1]) 
        x22_0 = self.conv22_0(x12_0)#之前的
        
        #x22_0_b = self.dilationConv22_0(x12_0)#修改的
        #x22_0 = torch.add(x22_0_a, x22_0_b)
        x2_0att = self.TSF2(g=x22_0, x=t2)
        
        
        #&&&&用来做监督
        #mspOutput2 = self.mpsdecoder2(x2_0att)
        mspOutput2 = F.relu(F.interpolate(self.mpsdecoder2_1(x2_0att), scale_factor=(2, 2), mode='bilinear'))
        mspOutput2 = torch.add(mspOutput2,F.interpolate(x12_0, scale_factor=(2, 2), mode='bilinear'))# 小的skip
        mspOutput2 = F.relu(F.interpolate(self.mpsdecoder2_2(mspOutput2), scale_factor=(2, 2), mode='bilinear'))
        #&&&&
        
        
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        
        x23_0 = self.conv23_0(x13_0)
        
        #print('=====================',x23_0.shape)#[8, 32, 16, 16]
        x33_0 = self.conv33_0(x23_0)#原来
        #x33_0_b = self.dilationConv33_0(x23_0)# 新加
        #x33_0 = torch.add(x33_0_a, x33_0_b)
        x3_0att = self.TSF3(g=x33_0, x=t3)
        
        
        #&&&&用来做监督
        #mspOutput3 = self.mpsdecoder3(x3_0att)
        
        mspOutput3 = F.relu(F.interpolate(self.mpsdecoder3_1(x3_0att), scale_factor=(2, 2), mode='bilinear'))
        mspOutput3 = torch.add(mspOutput3,F.interpolate(x23_0, scale_factor=(2, 2), mode='bilinear'))# 小的skip
        mspOutput3 = F.relu(F.interpolate(self.mpsdecoder3_2(mspOutput3), scale_factor=(2, 2), mode='bilinear'))
        mspOutput3 = F.relu(F.interpolate(self.mpsdecoder3_3(mspOutput3), scale_factor=(2, 2), mode='bilinear'))
        #&&&&
        
        
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.Attn_block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) # 原来
        #x44_0_b = self.dilationConv44_0(x34_0)# 新加
        #x44_0 = torch.add(x44_0_a, x44_0_b)
        x4_0att = self.TSF4(g=x44_0, x=t4)
        
        
        #&&&&用来做监督
        #mspOutput4 = self.mpsdecoder4(x4_0att)
        mspOutput4 = F.relu(F.interpolate(self.mpsdecoder4_1(x4_0att), scale_factor=(2, 2), mode='bilinear'))
        mspOutput4 = torch.add(mspOutput4,F.interpolate(x34_0, scale_factor=(2, 2), mode='bilinear'))# 小的skip
        mspOutput4 = F.relu(F.interpolate(self.mpsdecoder4_2(mspOutput4), scale_factor=(2, 2), mode='bilinear'))
        mspOutput4 = F.relu(F.interpolate(self.mpsdecoder4_3(mspOutput4), scale_factor=(2, 2), mode='bilinear'))
        mspOutput4 = F.relu(F.interpolate(self.mpsdecoder4_4(mspOutput4), scale_factor=(2, 2), mode='bilinear'))
        #&&&&
        
        
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
       
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.Attn_block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        ###########################
        # out = self.bottom_aspp(out)
        out = self.MIA(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大
        #print('t4================',t4.shape)
        out = torch.add(out, self.skipFilter4(t4))  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dAttn_block1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, self.skipFilter3(t3))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dAttn_block2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return mspOutput1,mspOutput2,mspOutput3,mspOutput4,self.final(out)


###############################################
###############################################
# （第二个小论文）  消融实验
class EncodeNum_SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################

        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter
        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####

        # 使用到不同尺度的图像的编码卷积
        self.dilationConv11_0 = DilationConv(input_channels, nb_filter[0], 16)
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])  # 第1个辅助编码器
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])  # 第2个辅助编码器
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])  # 第3个辅助编码器
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  # 第4个辅助编码器

        self.dilationConv22_0 = DilationConv(nb_filter[0], nb_filter[1], 16)
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第2个辅助编码器
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第3个辅助编码器
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])  # 第4个辅助编码器

        self.dilationConv33_0 = DilationConv(nb_filter[1], nb_filter[2], 8)
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第3个辅助编码器
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])  # 第4个辅助编码器

        self.dilationConv44_0 = DilationConv(nb_filter[2], nb_filter[3], 4)
        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])  # 第4个辅助编码器
        # SE_MX_block   MX_block  AG_block   C3_MX_block
        self.Attdw1 = MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])  # 可以在每个门口单元之后，再加一个Conv1×1的卷积
        self.Attdw2 = MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################
        self.skipAttdw1 = AG_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0]) #没有使用
        self.skipAttdw2 = AG_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1]) #没有使用
        self.skipAttdw3 = AG_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2]) #没有使用
        self.skipAttdw4 = AG_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3]) #没有使用
        
        
        # AdvancedC3
        self.c3Filter1 = AdvancedC3(nb_filter[0], nb_filter[0])
        self.c3Filter2 = AdvancedC3(nb_filter[1], nb_filter[1])
        self.c3Filter3 = AdvancedC3(nb_filter[2], nb_filter[2])
        self.c3Filter4 = AdvancedC3(nb_filter[3], nb_filter[3])
        
        
        self.skipFilter1 = Filters(channels = nb_filter[0])
        self.skipFilter2 = Filters(channels = nb_filter[1])
        self.skipFilter3 = Filters(channels = nb_filter[2])
        self.skipFilter4 = Filters(channels = nb_filter[3])
        ####################################################

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)
        print("===============================This is EncodeNum_SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #################################没效果
        # self.x_seq = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten(),
        #    nn.Linear(256, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Linear(512, 512)
        # )
        #########################################

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        
        #这里修改了
        #for i in range(len(self.nb_filter) - 1):
        #    Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #
        ######################################
        crop = []
        #辅助编码器的图像进行数据增强
        for i in range(len(self.nb_filter) - 1):
            crop.append(transforms.RandomCrop(size=(int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i]))))
            Images.append(crop[i](x))
        ###########################################

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        #######################################  金字塔图像1
        x11_0 = self.conv11_0(Images[0]) # 之前的
        x1_0att = self.skipAttdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))
        #########################################

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        #######################################  金字塔图像2
        x12_0 = self.conv12_0(Images[1]) 
        x22_0 = self.conv22_0(x12_0)
        x2_0att = self.skipAttdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################

        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        #######################################  金字塔图像3
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)
        x33_0 = self.conv33_0(x23_0)
        x3_0att = self.skipAttdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out

        #######################################  金字塔图像4
        x14_0 = self.conv14_0(Images[3])
        x24_0 = self.conv24_0(x14_0)
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0) 
        x4_0att = self.skipAttdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
       
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度
        ###########################
        # out = self.bottom_aspp(out)
        out = self.bottom_mlpaspp(out)
        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大
        #print('t4================',t4.shape)
        out = torch.add(out, self.skipFilter4(t4))  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, self.skipFilter3(t3))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)


###############################################
###############################################
#密集金字塔图像编码
class Dense_Test_SACA_UNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        ###################################################
        nb_filter = [16, 32, 128, 160, 256]
        self.nb_filter = nb_filter

        ####用于cat之后的回复通道
        self.conv1_0 = DoubleConv(nb_filter[0] * 2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1] * 2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2] * 2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3] * 2, nb_filter[3])
        ####

        # 使用到不同尺度的图像的编码卷积
        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])

        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])

        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])
        # MX_block  AG_block   SE_MX_block
        self.Attdw1 = MX_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=nb_filter[0])
        self.Attdw2 = MX_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[1])
        self.Attdw3 = MX_block(F_g=nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[2])
        self.Attdw4 = MX_block(F_g=nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[3])
        ##################################################

        print("===============================This is Dense_Test_SACA_UNet")
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # FANBlock_SE  shiftedBlock
        self.block1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        # self.bottom_aspp = ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])
        self.bottom_mlpaspp = MLP_ASPP(in_channels=embed_dims[2], atrous_rates=[6, 12, 18], out_channels=embed_dims[2])

        #############################
        # 没效果
        # self.x_seq = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(256, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512)
        # )
        #################################

        self.dblock1 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([FANBlock_SE(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

        # 将特征图尺寸变为1×1
        self.change_map = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Images  获取金字塔图像
        image_size = x.shape
        Images = []
        divsize = [2, 4, 8, 16]
        for i in range(len(self.nb_filter) - 1):
            Images.append(TF.resize(x, size=[int(image_size[2] / divsize[i]), int(image_size[3] / divsize[i])]))

        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        # print('t1------------',t1.shape)
        #######################################  金字塔图像
        x11_0 = self.conv11_0(Images[0])
        x1_0att = self.Attdw1(g=x11_0, x=t1)
        out = self.conv1_0(torch.cat((x1_0att, t1), dim=1))

        # print('out------------', out.shape)
        #########################################
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        #######################################  金字塔图像
        x12_0 = self.conv12_0(Images[1])
        x12_0 = x12_0 * self.change_map(x11_0)
        x22_0 = self.conv22_0(x12_0)
        x2_0att = self.Attdw2(g=x22_0, x=t2)
        out = self.conv2_0(torch.cat((x2_0att, t2), dim=1))
        #########################################
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        #######################################  金字塔图像
        x13_0 = self.conv13_0(Images[2])
        x13_0 = x13_0 * self.change_map(x12_0)
        x23_0 = self.conv23_0(x13_0)
        x23_0 = x23_0 * self.change_map(x22_0)
        x33_0 = self.conv33_0(x23_0)

        x3_0att = self.Attdw3(g=x33_0, x=t3)
        out = self.conv3_0(torch.cat((x3_0att, t3), dim=1))
        #########################################
        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W = self.patch_embed3(
            out)  # 其实就是一个卷积进行维度变化和尺寸变化，然后对将特征维度从B C H W 变为 B C H*W  # 图像H W减半，图像特征被压缩成了一条特征 out： C (H/2)*(W/2) B  （每一个MLP block之前都会做的处理 增加通道）
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        # print(out)
        out = self.norm3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复到原来的维度 B C H W(此时H W已经减半)
        t4 = out
        #######################################  金字塔图像
        x14_0 = self.conv14_0(Images[3])
        x14_0 = x14_0 * self.change_map(x13_0)
        x24_0 = self.conv24_0(x14_0)
        x24_0 = x24_0 * self.change_map(x23_0)
        x34_0 = self.conv34_0(x24_0)
        x34_0 = x34_0 * self.change_map(x33_0)
        x44_0 = self.conv44_0(x34_0)
        x4_0att = self.Attdw4(g=x44_0, x=t4)
        out = self.conv4_0(torch.cat((x4_0att, t4), dim=1))
        #########################################
        ### Bottleneck
        # 继续进行 MLP block
        out, H, W = self.patch_embed4(out)  # 继续首先进行长、宽减半，  通道数增加
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 恢复减半前的维度

        ###########################
        # out = self.bottom_aspp(out)

        out = self.bottom_mlpaspp(out)

        #############################

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2),
                                   mode='bilinear'))  # 先改变通道数(从256->160)，再改变图像尺寸增大

        out = torch.add(out, t4)  # 和对应大小的和通道的特征图 逐元素相加
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        ### Stage 3
        # print('index', out.shape)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        ###############
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        ###############
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
##################################################
###############################################
###############################################
#### Pyramid Att-UNet #########################################################
   
class PAttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(PAttUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                
        
        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter
        print('===========================This is PAttUNet')
        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0]*2, nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1]*2, nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2]*2, nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3]*2, nb_filter[4])
        

        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  
        
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1]) 

        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2]) 

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])
        # AG_block 
        self.Attdw1 = AG_block(F_g= nb_filter[0], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))
        self.Attdw2 = AG_block(F_g= nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Attdw3 = AG_block(F_g= nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])   
        self.Attdw4 = AG_block(F_g= nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])        
                      
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.Att4 = AG_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3 = AG_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2 = AG_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])        
        self.Att1 = AG_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2)) 
        

        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        
    def forward(self, input):
        # Images
        image_size = input.shape
        Images = []
        divsize = [2,4,8,16]
        for i in range(len(self.nb_filter)-1):
            Images.append(TF.resize(input, size=[int(image_size[2]/divsize[i]) , int(image_size[3]/divsize[i])]))
        
        # encoding path
        x0_0 = self.conv0_0(input)
        
        x11_0 = self.conv11_0(Images[0])
        x1_0att = self.Attdw1(g=x11_0, x=self.pool(x0_0))        
        x1_0 = self.conv1_0(torch.cat((x1_0att, self.pool(x0_0)),dim=1))

        x12_0 = self.conv12_0(Images[1])
        x22_0 = self.conv22_0(x12_0)         
        x2_0att = self.Attdw2(g=x22_0, x=self.pool(x1_0))  
        x2_0 = self.conv2_0(torch.cat((x2_0att, self.pool(x1_0)),dim=1))        
        
        
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)        
        x33_0 = self.conv33_0(x23_0) 
        x3_0att = self.Attdw3(g=x33_0, x=self.pool(x2_0))        
        x3_0 = self.conv3_0(torch.cat((x3_0att, self.pool(x2_0)),dim=1))
        
        x14_0 = self.conv14_0(Images[3])  
        x24_0 = self.conv24_0(x14_0)         
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0)
        x4_0att = self.Attdw4(g=x44_0, x=self.pool(x3_0))
        x4_0 = self.conv4_0(torch.cat((x4_0att, self.pool(x3_0)),dim=1)) 
       
                             
      
        # decoding + concat path        
        x3_1 = self.up(x4_0)
        x3_0 = self.Att4(g=x3_1, x=x3_0) 
        x3_1 = self.conv3_1(torch.cat((x3_0, x3_1),dim=1))
        

        x2_2 = self.up(x3_1)
        x2_0 = self.Att3(g=x2_2, x=x2_0) 
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_2),dim=1)) 
        
        x1_3 = self.up(x2_2)
        x1_0 = self.Att2(g=x1_3, x=x1_0) 
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_3),dim=1)) 

        x0_4 = self.up(x1_3)
        x0_0 = self.Att1(g=x0_4, x=x0_0) 
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_4),dim=1))                
        
        output = self.final(x0_4)         

        return output 
##############################################
###### MX Block
class MX_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(MX_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2*F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.ReLU(),
            nn.Conv2d(F_int, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(self.cat_conv(torch.cat((g1,x1),dim=1)))
        #psi = self.relu(g1 + x1)
        #psi = self.psi(psi)

        return x * psi  
#################################################
#C3_MX_block
###### C3_MX_block
class C3_MX_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(C3_MX_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        #self.psi = nn.Sequential(
        #    nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #    nn.BatchNorm2d(1),
        #    nn.Sigmoid()
        #)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2*F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.ReLU(),
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.c3 = AdvancedC3(F_int ,F_int)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.cat((g1,x1),dim=1)
        
        psi = self.relu(self.cat_conv(psi))
        #psi = self.c3(psi)
        #psi = self.relu(g1 + x1)
        #psi = self.psi(psi)
        psi = x * psi
        psi = self.c3(psi)
        return psi  
################################################
###### SE_MX Block
class SE_MX_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(SE_MX_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2*F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.ReLU(),
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.se_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(F_int, F_int,bias=False),
            nn.ReLU(),
            nn.Linear(F_int, F_int, bias=True),
           
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(self.cat_conv(torch.cat((g1,x1),dim=1)))
        
        psi = x * psi
        psi_linear = self.se_linear(psi)
        psi_linear = psi_linear.view(psi_linear.shape[0], -1, 1, 1)
        return psi * psi_linear
##################################################
#AG_block
class AG_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AG_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
##################################################
##################################################
#AdvancedC3
class AdvancedC3(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, ratio=[2,4,8]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 1, 1)

        self.d1 = C3block(n, n + n1, 3, 1, ratio[0])
        self.d2 = C3block(n, n, 3, 1, ratio[1])
        self.d3 = C3block(n, n, 3, 1, ratio[2])
        # self.d4 = Double_CDilated(n, n, 3, 1, 12)
        # self.conv =C(nOut, nOut, 1,1)

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = torch.cat([d1, d2, d3], 1)

        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output
class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
        # self.act = nn.ReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output
class C3block(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv =nn.Sequential(
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False)
            )
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.PReLU(nIn),
                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
####################################################
#UNet++
class UNetplus(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, deep_supervision=False):
        super(UNetplus, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]
        print("===============================This is UNetplus")
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


#### AttUNet #########################################################

class AttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, deep_supervision=False):
        super(AttUNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        nb_filter = [32, 64, 128, 256, 512]
        print("===============================This is AttUNet")
        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.Att4 = AG_block(F_g=nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])

        self.Att3 = AG_block(F_g=nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])

        self.Att2 = AG_block(F_g=nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])

        self.Att1 = AG_block(F_g=nb_filter[1], F_l=nb_filter[0], F_int=int(nb_filter[0] / 2))

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # decoding + concat path
        x3_1 = self.up(x4_0)
        x3_0 = self.Att4(g=x3_1, x=x3_0)
        x3_1 = self.conv3_1(torch.cat((x3_0, x3_1), dim=1))

        x2_2 = self.up(x3_1)
        x2_0 = self.Att3(g=x2_2, x=x2_0)
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_2), dim=1))

        x1_3 = self.up(x2_2)
        x1_0 = self.Att2(g=x1_3, x=x1_0)
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_3), dim=1))

        x0_4 = self.up(x1_3)
        x0_0 = self.Att1(g=x0_4, x=x0_0)
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_4), dim=1))

        output = self.final(x0_4)

        return output
###############################################
# 加入另外一个分支  知道SACA-UNet
class Guide_UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        # up-right
        # self.up_conv1 = nn.Conv2d(32, 16, 1, bias=False)
        # self.up_conv2 = nn.Conv2d(128, 32, 1, bias=False)
        # self.up_conv3 = nn.Conv2d(160, 128, 1, bias=False)

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        #self.norm3 = norm_layer(embed_dims[1])
        #self.norm4 = norm_layer(embed_dims[2])

        #self.dnorm3 = norm_layer(160)
        #self.dnorm4 = norm_layer(128)

        #self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
         #                                     embed_dim=embed_dims[1])
        #self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
         #                                     embed_dim=embed_dims[2])

        #self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        #self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        #self.dbn1 = nn.BatchNorm2d(160)
        #self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        ##
        # 每一个块执行之前，都会先改变  通道和图像大小 ->之后再经过每一个块
        # #

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        _, _, H, W = out.shape

        ###Decoder
        #out = self.dnorm4(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)

################################################
class UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, deep_supervision=False):
        super(UNet, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]
        print("===============================This is UNet")
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#######################################################
#######################################################
# MT-Unet
#!/usr/bin/env python
# -*- coding:utf-8 -*-




class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)

        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        return x


class MEAttention(nn.Module):
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1,
                                                     3)  #(1, 32, 225, 32)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, configs, axial=False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = int(self.dim / configs["head"])
        self.all_head_size = self.num_head * self.attention_head_size

        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)

        self.out = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        # first row and col attention
        if self.axial:
            # row attention (single head attention)
            b, h, w, c = x.shape
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer_x = mixed_query_layer.view(b * h, w, -1)
            key_layer_x = mixed_key_layer.view(b * h, w, -1).transpose(-1, -2)
            attention_scores_x = torch.matmul(query_layer_x,
                                              key_layer_x)  # (b*h, w, w, c)
            attention_scores_x = attention_scores_x.view(b, -1, w,
                                                         w)  # (b, h, w, w)

            # col attention  (single head attention)
            query_layer_y = mixed_query_layer.permute(0, 2, 1,
                                                      3).contiguous().view(
                                                          b * w, h, -1)
            key_layer_y = mixed_key_layer.permute(
                0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attention_scores_y = torch.matmul(query_layer_y,
                                              key_layer_y)  # (b*w, h, h, c)
            attention_scores_y = attention_scores_y.view(b, -1, h,
                                                         h)  # (b, w, h, h)

            return attention_scores_x, attention_scores_y, mixed_value_layer

        else:

            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer = self.transpose_for_scores(mixed_query_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()  # (b, p, p, head, n, c)
            key_layer = self.transpose_for_scores(mixed_key_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            value_layer = self.transpose_for_scores(mixed_value_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()

            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            atten_probs = self.softmax(attention_scores)

            context_layer = torch.matmul(
                atten_probs, value_layer)  # (b, p, p, head, win, h)
            context_layer = context_layer.permute(0, 1, 2, 4, 3,
                                                  5).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.all_head_size, )
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)

        return attention_output


class WinAttention(nn.Module):
    def __init__(self, configs, dim):
        super(WinAttention, self).__init__()
        self.window_size = configs["win_size"]
        self.attention = Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            right_size = h + self.window_size - h % self.window_size
            new_x = torch.zeros((b, c, right_size, right_size))
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:,
                  x.shape[3]:] = x[:, :, (x.shape[2] - right_size):,
                                   (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5,
                      1).contiguous().view(b, h // self.window_size,
                                           w // self.window_size,
                                           self.window_size * self.window_size,
                                           c).cuda()
        x = self.attention(x)  # (b, p, p, win, h)
        return x


class DlightConv(nn.Module):
    def __init__(self, dim, configs):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)  # (b, n, n, 1, h)
        x_prob = self.softmax(self.linear(avg_x))  # (b, n, n, win)

        x = torch.mul(h,
                      x_prob.unsqueeze(-1))  # (b, p, p, 16, h) (b, p, p, 16)
        x = torch.sum(x, dim=-2)  # (b, n, n, 1, h)
        return x


class GaussianTrans(nn.Module):
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x_full, atten_y_full, value_full = x  # atten_x_full(b, h, w, w, c)   atten_y_full(b, w, h, h, c) value_full(b, h, w, c)
        new_value_full = torch.zeros_like(value_full)

        for r in range(x.shape[1]):  # row
            for c in range(x.shape[2]):  # col
                atten_x = atten_x_full[:, r, c, :]  # (b, w)
                atten_y = atten_y_full[:, c, r, :]  # (b, h)

                dis_x = torch.tensor([(h - c)**2 for h in range(x.shape[2])
                                      ]).cuda()  # (b, w)
                dis_y = torch.tensor([(w - r)**2 for w in range(x.shape[1])
                                      ]).cuda()  # (b, h)

                dis_x = -(self.shift * dis_x + self.bias).cuda()
                dis_y = -(self.shift * dis_y + self.bias).cuda()

                atten_x = self.softmax(dis_x + atten_x)
                atten_y = self.softmax(dis_y + atten_y)

                new_value_full[:, r, c, :] = torch.sum(
                    atten_x.unsqueeze(dim=-1) * value_full[:, r, :, :] +
                    atten_y.unsqueeze(dim=-1) * value_full[:, :, c, :],
                    dim=-2)
        return new_value_full


class CSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.win_atten = WinAttention(configs, dim)
        self.dlightconv = DlightConv(dim, configs)
        self.global_atten = Attention(dim, configs)
        self.gaussiantrans = GaussianTrans()
        #self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        #self.maxpool = nn.MaxPool2d(2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        '''

        :param x: size(b, n, c)
        :return:
        '''
        origin_size = x.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
            origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        x = self.win_atten(x)  # (b, p, p, win, h)
        b, p, p, win, c = x.shape
        h = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                   c).permute(0, 1, 3, 2, 4, 5).contiguous()
        h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                   c).permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        x = self.dlightconv(x)  # (b, n, n, h)
        atten_x, atten_y, mixed_value = self.global_atten(
            x)  # (atten_x, atten_y, value)
        gaussian_input = (x, atten_x, atten_y, mixed_value)
        x = self.gaussiantrans(gaussian_input)  # (b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.up(x)
        x = self.queeze(torch.cat((x, h), dim=1)).permute(0, 2, 3,
                                                          1).contiguous()
        x = x[:, :origin_h, :origin_w, :].contiguous()
        x = x.view(b, -1, c)

        return x


class EAmodule(nn.Module):
    def __init__(self, dim):
        super(EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm(x)

        x = self.CSAttention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.ElayerNorm(x)

        x = self.EAttention(x)
        x = h + x

        return x


class DecoderStem(nn.Module):
    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):
        x = self.block(x, features)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)  #out_dim, model_dim
        #self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))

    def forward(self, x):

        x, features = self.model(x)  # (1, 512, 28, 28)
        x = self.trans_dim(x)  # (B, C, H, W) (1, 256, 28, 28)
        x = x.flatten(2)  # (B, H, N)  (1, 256, 28*28)
        x = x.transpose(-2, -1)  #  (B, N, H)
        #print(x.shape)
        #print(self.position_embedding.shape)
        x = x #+ self.position_embedding
        return x, features  #(B, N, H)


class encoder_block(nn.Module):
    def __init__(self, dim):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim),
            EAmodule(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1,
                                       2)  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)  # (14, 14, 256)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, dim, flag):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2),
                EAmodule(dim // 2)
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                EAmodule(dim),
                EAmodule(dim)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MTUNet, self).__init__()
        print('=======================This is MTUNet')
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(configs["bottleneck"]),
                                        EAmodule(configs["bottleneck"]))
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x


configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}
######################################################
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_Unet(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decode
        self.conv_decode4 = expansive_block(1024+512, 512, 512)
        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        final_layer = self.final_layer(decode_block0)
        final_layer = self.final_up(final_layer)
        return final_layer

###################################################

class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4) # 逐元素相加
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1): # 进行下一次的shiftedBlock 块
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)
# %% ######################################################################################
# create model

class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                          )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3


class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(3, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, scale_img="none"):
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # without skip
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # with skip
        return out


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        out = torch.sigmoid(self.conv3(x1))

        return out


class FCT(nn.Module):
    def __init__(self):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [8, 16, 32, 64, 128, 64, 32, 16, 8]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
       
        self.drp_out = 0.3

        # shape
        #init_sizes = torch.ones((2, 224, 224, 3))
        #init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_1 = Block_encoder_bottleneck("first", 3, filters[0], att_heads[0], dpr[0])
        self.block_2 = Block_encoder_bottleneck("second", filters[0], filters[1], att_heads[1], dpr[1])
        self.block_3 = Block_encoder_bottleneck("third", filters[1], filters[2], att_heads[2], dpr[2])
        self.block_4 = Block_encoder_bottleneck("fourth", filters[2], filters[3], att_heads[3], dpr[3])
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads[4], dpr[4])
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8])

        self.ds7 = DS_out(filters[6], 1)
        self.ds8 = DS_out(filters[7], 1)
        self.ds9 = DS_out(filters[8], 1)

    def forward(self, x):
        # Multi-scale input
        scale_img_2 = self.scale_img(x)
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)

        x = self.block_1(x)
        # print(f"Block 1 out -> {list(x.size())}")
        skip1 = x
        x = self.block_2(x, scale_img_2)
        # print(f"Block 2 out -> {list(x.size())}")
        skip2 = x
        x = self.block_3(x, scale_img_3)
        # print(f"Block 3 out -> {list(x.size())}")
        skip3 = x
        x = self.block_4(x, scale_img_4)
        # print(f"Block 4 out -> {list(x.size())}")
        skip4 = x
        x = self.block_5(x)
        # print(f"Block 5 out -> {list(x.size())}")
        x = self.block_6(x, skip4)
        # print(f"Block 6 out -> {list(x.size())}")
        x = self.block_7(x, skip3)
        # print(f"Block 7 out -> {list(x.size())}")
        skip7 = x
        x = self.block_8(x, skip2)
        # print(f"Block 8 out -> {list(x.size())}")
        skip8 = x
        x = self.block_9(x, skip1)
        # print(f"Block 9 out -> {list(x.size())}")
        skip9 = x

        out7 = self.ds7(skip7)
        # print(f"DS 7 out -> {list(out7.size())}")
        out8 = self.ds8(skip8)
        # print(f"DS 8 out -> {list(out8.size())}")
        out9 = self.ds9(skip9)
        # print(f"DS 9 out -> {list(out9.size())}")

        return out9


def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

############################################################################
class Mlp_SWIM(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
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


def window_reverse(windows, window_size, H, W):
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
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
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
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_SWIM(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

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

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer_2SwinUnet(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up_2SwinUnet(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinUnet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, input_channels=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("===============================This is SwinUNet")

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=input_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_2SwinUnet(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up_2SwinUnet(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
############################################################################
class SwinUNetR(nn.Module):
    def __init__(
        self,
        img_size,
        input_channels: int,
        out_channels: int,
        depths = (2, 2, 2, 2),
        num_heads = (3, 6, 12, 24),
        feature_size = 24,
        norm_name = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """
        super().__init__()
        print("===============================This is SwinUNetR")
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=input_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
        # pdb.set_trace()
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        # pdb.set_trace()
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits
############################################################################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channely we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

class Mlp_nnFormer(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_nnFormer(x, window_size):
  
    B, S, H, W, C = x.shape
    x = x.view(B, S // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse_nnFormer(windows, window_size, S, H, W):
   
    B = int(windows.shape[0] / (S * H * W / window_size / window_size / window_size))
    x = windows.view(B, S // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x



class SwinTransformerBlock_kv(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
                dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        #self.window_size=to_3tuple(self.window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_nnFormer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       
    def forward(self, x, mask_matrix,skip=None,x_up=None):
    
        B, L, C = x.shape
        S, H, W = self.input_resolution
 
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        x = x.view(B, S, H, W, C)
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = skip.shape

       
        
        # cyclic shift
        if self.shift_size > 0:
            skip = torch.roll(skip, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            x_up = torch.roll(x_up, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            skip = skip
            x_up=x_up
            attn_mask = None
        # partition windows
        skip = window_partition_nnFormer(skip, self.window_size) 
        skip = skip.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        x_up = window_partition_nnFormer(x_up, self.window_size) 
        x_up = x_up.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  
        attn_windows=self.attn(skip,x_up,mask=attn_mask,pos_embed=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse_nnFormer(attn_windows, self.window_size, Sp, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        
class WindowAttention_kv(nn.Module):
   
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w])) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, skip,x_up,pos_embed=None, mask=None):

        B_, N, C = skip.shape
        
        kv = self.kv(skip)
        q = x_up

        kv=kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_,N,self.num_heads,C//self.num_heads).permute(0,2,1,3).contiguous()
        k,v = kv[0], kv[1]  
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention_nnFormer(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None,pos_embed=None):

        B_, N, C = x.shape
        
        qkv = self.qkv(x)
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x+pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock_nnFormer(nn.Module):
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
   
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        self.attn = WindowAttention_nnFormer(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
       
            

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_nnFormer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
       
    def forward(self, x, mask_matrix):

        B, L, C = x.shape
        S, H, W = self.input_resolution
   
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_g = (self.window_size - S % self.window_size) % self.window_size

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size,-self.shift_size), dims=(1, 2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
       
        # partition windows
        x_windows = window_partition_nnFormer(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size,
                                   C)  

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask,pos_embed=None)  

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse_nnFormer(attn_windows, self.window_size, Sp, Hp, Wp) 

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging_nnFormer(nn.Module):
  

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=3,stride=2,padding=1)
       
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,2*C)
      
        return x
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
       
        self.norm = norm_layer(dim)
        # self.up=nn.ConvTranspose3d(dim,dim//2,2,2)
        self.up = nn.ConvTranspose3d(dim, dim//4, 2, 2)  # 输出通道数与 skip_dim 一致

    def forward(self, x, S, H, W):
      
        
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

       
        
        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,C//2)
       
        return x
class BasicLayer(nn.Module):
   
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        # build blocks
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_nnFormer(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
      

        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_nnFormer(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
          
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            Ws, Wh, Ww = (S + 1) // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W

class BasicLayer_up(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        

        # build blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SwinTransformerBlock_kv(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 ,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                    )
        for i in range(depth-1):
            self.blocks.append(
                SwinTransformerBlock_nnFormer(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size // 2 ,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i+1] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                        )
        

        
        self.Upsample = upsample(dim=2*dim, norm_layer=norm_layer)
    def forward(self, x,skip, S, H, W):
        
      
        x_up = self.Upsample(x, S, H, W)  # torch.Size([8, 1152, 768])
       
        x = x_up + skip
        S, H, W = S * 2, H * 2, W * 2
        # calculate attention mask for SW-MSA
        Sp = int(np.ceil(S / self.window_size)) * self.window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Sp, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        s_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for s in s_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, s, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_nnFormer(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size * self.window_size)  # 3d��3��winds�˻�����Ŀ�Ǻܴ�ģ�����winds����̫��
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        x = self.blocks[0](x, attn_mask,skip=skip,x_up=x_up)
        for i in range(self.depth-1):
            x = self.blocks[i+1](x,attn_mask)
        
        return x, S, H, W
        
class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        #norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        

        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            #norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x
        
    

class PatchEmbed_nnFormer(nn.Module):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=[patch_size[0],patch_size[1]//2,patch_size[2]//2]
        stride2=[patch_size[0],patch_size[1]//2,patch_size[2]//2]
        self.proj1 = project(in_chans,embed_dim//2,stride1,1,nn.GELU,nn.LayerNorm,False)
        self.proj2 = project(embed_dim//2,embed_dim,stride2,1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x



class Encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1  ,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3)
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_nnFormer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

       

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
   
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    (pretrain_img_size[0] // patch_size[0] // 2 ** i_layer) if pretrain_img_size[0] // patch_size[0]  // 2 ** i_layer != 0 else 1, pretrain_img_size[1] // patch_size[1] // 2 ** i_layer,
                    pretrain_img_size[2] // patch_size[2] // 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging_nnFormer
                if (i_layer < self.num_layers - 1) else None
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def forward(self, x):
        """Forward function."""
        
        x = self.patch_embed(x)  # torch.Size([8, 192, 1, 96, 96])
        down=[]
       
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)
        
      
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
              
                down.append(out)
        return down

   

class Decoder(nn.Module):
    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2,2,2],
                 num_heads=[24,12,6],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths)-i_layer-1)),
                input_resolution=(
                    (pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1)) if (pretrain_img_size[0] // patch_size[0] // 2 ** (len(depths)-i_layer-1)) != 0 else 1, pretrain_img_size[1] // patch_size[1] // 2 ** (len(depths)-i_layer-1),
                    pretrain_img_size[2] // patch_size[2] // 2 ** (len(depths)-i_layer-1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(
                    depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding
                )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
    def forward(self,x,skips):
            
        outs=[]
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()  # torch.Size([8, 144, 1536])
        for index,i in enumerate(skips):
             i = i.flatten(2).transpose(1, 2).contiguous()
             skips[index]=i
        x = self.pos_drop(x)  # torch.Size([8, 144, 1536])
            
        for i in range(self.num_layers)[::-1]:
            
            layer = self.layers[i]
            
            x, S, H, W,  = layer(x,skips[i], S, H, W)
            out = x.view(-1, S, H, W, self.num_features[i])
            outs.append(out)
        return outs

      
class final_patch_expanding(nn.Module):
    def __init__(self,dim,num_class,patch_size):
        super().__init__()
        self.up=nn.ConvTranspose3d(dim,num_class,patch_size,patch_size)
      
    def forward(self,x):
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.up(x)
      
        
        return x    




                                         
class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[64,128,128],
                embedding_dim=192,
                input_channels=3, 
                num_classes=1, 
                conv_op=nn.Conv3d, 
                depths=[2,2,2,2],
                num_heads=[6, 12, 24, 48],
                patch_size=[1,4,4],
                window_size=[4,4,8,4],
                deep_supervision=True):
      
        super(nnFormer, self).__init__()
        
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes=num_classes
        self.conv_op=conv_op
       
        
        self.upscale_logits_ops = []
     
        
        self.upscale_logits_ops.append(lambda x: x)
        
        embed_dim=embedding_dim
        depths=depths
        num_heads=num_heads
        patch_size=patch_size
        window_size=window_size
        self.model_down=Encoder(pretrain_img_size=crop_size,window_size=window_size,embed_dim=embed_dim,patch_size=patch_size,depths=depths,num_heads=num_heads,in_chans=input_channels)
        self.decoder=Decoder(pretrain_img_size=crop_size,embed_dim=embed_dim,window_size=window_size[::-1][1:],patch_size=patch_size,num_heads=num_heads[::-1][1:],depths=depths[::-1][1:])
        
        self.final=[]
        if self.do_ds:
            
            for i in range(len(depths)-1):
                self.final.append(final_patch_expanding(embed_dim*2**i,num_classes,patch_size=patch_size))

        else:
            self.final.append(final_patch_expanding(embed_dim,num_classes,patch_size=patch_size))
    
        self.final=nn.ModuleList(self.final)
    

    def forward(self, x):
      
            
        seg_outputs=[]
        skips = self.model_down(x)
        neck=skips[-1]
       
        out=self.decoder(neck,skips)
        
       
            
        if self.do_ds:
            for i in range(len(out)):  
                seg_outputs.append(self.final[-(i+1)](out[i]))
        
          
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]
############################################################################

#EOF
if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    #unext = UNext(num_classes=1)
    unext = Resnet34_Unet(in_channel=3, out_channel=1)
    y = unext(x)
