from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from util.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding, reverse_patches
from util.position import PositionEmbeddingLearned, PositionEmbeddingSine
import pdb
import math

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=48, patch_size=2, in_chans=64, embed_dim=768):
        super().__init__()
        img_size = tuple((img_size,img_size))
        patch_size = tuple((patch_size,patch_size))
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #16*64*48*48
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # pdb.set_trace()
        x = self.proj(x).flatten(2).transpose(1, 2)#64*48*48->768*6*6->768*36->36*768
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//4
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




class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        print('scale', self.scale)
        print(dim)
        print(dim//num_heads)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        # pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv: 3*16*8*37*96
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # pdb.set_trace()
        
        q_all = torch.split(q, math.ceil(N//4), dim=-2)
        k_all = torch.split(k, math.ceil(N//4), dim=-2)
        v_all = torch.split(v, math.ceil(N//4), dim=-2)        

        output = []
        for q,k,v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale   #16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            trans_x = (attn @ v).transpose(1, 2)#.reshape(B, N, C)

            output.append(trans_x)
        # pdb.set_trace()
        # attn = torch.cat(att, dim=-2)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) #16*37*768
        x = torch.cat(output,dim=1)
        x = x.reshape(B,N,C)
        # pdb.set_trace()
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x




## Base block
class MLABlock(nn.Module):
    def __init__(
        self, n_feat = 64,dim=768, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(MLABlock, self).__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=8, qkv_bias=False, qk_scale=None, \
                             attn_drop=0., proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        # self.posi = PositionEmbeddingLearned(n_feat)
        self.mlp = Mlp(in_features=dim, hidden_features=dim//4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)
    def forward(self, x):
        # pdb.set_trace()
        B = x.shape[0]
        # posi = self.posi(x)
        # x = posi + x
        # x = self.patch_embed(x) # 16*36*768
        # print(x.shape)
        x = extract_image_patches(x, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same')#   16*2304*576
        x = x.permute(0,2,1)

        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#self.drop_path(self.mlp(self.norm2(x)))
        return x