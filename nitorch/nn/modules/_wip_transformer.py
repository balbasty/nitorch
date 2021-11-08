from collections import OrderedDict
import inspect
import torch
import torch.utils.checkpoint as checkpoint
import numpy as np
from nitorch.nn.base import Module, Sequential, ModuleList
from nitorch.core import py, utils, math, linalg
from nitorch.nn.activations.base import make_activation_from_name
from .norm import make_norm_from_name
from nitorch import spatial
from .linear import Linear, MLP
from .conv import ConvBlock, _get_dropout_class
from .dropout import DropPath
from .encode_decode import DownStep


def _build_dropout(dropout, dim, path=False):
        if path:
            dropout = DropPath(dropout)
        else:
            dropout = (dropout() if inspect.isclass(dropout)
                    else dropout if callable(dropout)
                    else _get_dropout_class(dim)(p=float(dropout)) if dropout
                    else None)
        return dropout


def window_partition(x, window_size, dim=3):
    if dim==2:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    elif dim==2:
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], 
                   D // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1],
                                                                      window_size[2], C)
    return windows


def window_reverse(windows, window_size, shape):
    dim = len(shape)
    B = int(windows.shape[0] / (np.prod(shape) / np.prod(window_size)))
    if dim==2:
        H, W = shape
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    elif dim==2:
        H, W, D = shape
        x = windows.view(B, H // window_size[0], W // window_size[1], D // window_size[2], 
                         window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, H, W, D, -1)
    return x


# class to implement attention-based models - ViT, TRUnet, SWin, UFormer


class PatchEmbed(Module):
    """
    Patch embedding for images/volumes in transformers
    """
    def __init__(self, dim, patch_size=16, in_channels=1, embed_dim=768, norm=None, flatten=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = ConvBlock(dim, in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size,
                              norm=norm, activation=None, order='cnda')

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            # [B, C, *Spatial] ---> [B, C, N] ---> [B, N, C]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
        return x


class PatchMerge(Module):
    def __init__(self, input_resolution, in_channels, dim=3, norm='layer'):
        self.input_resolution = input_resolution
        self.dim = dim
        # need to check if dim**2 vs 2**dim...
        self.reduction = Linear(2**dim * in_channels, 2 * in_channels, bias=False)
        if isinstance(norm, str):
            self.norm = make_norm_from_name(norm, dim, 2**dim * in_channels)
        elif norm:
            self.norm = norm(2**dim * in_channels)
        else:
            self.norm = None

    def forward(self, x):
        if self.dim==2:
            H, W = self.input_resolution
        elif self.dim==3:
            H, W, D = self.input_resolution
        B, L, C = x.shape

        assert L == np.prod(self.input_resolution), 'input feature has wrong size'
        for d in self.input_resolution:
            assert d%2==0,  f"x size ({self.input_resolution}) not even." 

        if self.dim==2:
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]  
            x1 = x[:, 1::2, 0::2, :]  
            x2 = x[:, 0::2, 1::2, :]  
            x3 = x[:, 1::2, 1::2, :]  

        elif self.dim==3:
            # TODO: fix x0:x3 for 3D
            x = x.view(B, H, W, D, C)
            x0 = x[:, 0::2, 0::2, :]  
            x1 = x[:, 1::2, 0::2, :]  
            x2 = x[:, 0::2, 1::2, :]  
            x3 = x[:, 1::2, 1::2, :] 
       
        x = torch.cat([x0, x1, x2, x3], -1)  
        x = x.view(B, -1, 4 * C) 

        if self.norm:
            x = self.norm(x)
        x = self.reduction(x)

        return x


class Attention(Module):
    """
    Code adapted from rwightman ViT implementation
    """
    def __init__(self, dim, in_channels,
                 nb_heads=8, qkv_bias=False,
                 attn_dropout=None, proj_dropout=None
                 ):
        super().__init__()
        self.nb_heads = nb_heads
        head_dim = in_channels // nb_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(in_channels, 3 * in_channels, bias=qkv_bias)
        self.attn_dropout = _build_dropout(attn_dropout, dim)
        self.proj = Linear(in_channels, in_channels)
        self.proj_dropout = _build_dropout(proj_dropout, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nb_heads, C // self.nb_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        if self.attn_dropout:
            attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_dropout:
            x = self.proj_dropout

        return x


class ViTBlock(Module):
    """
    Block of Norm-Attention-Norm-MLP used for ViT.
    """
    def __init__(self, dim, in_channels, nb_heads, mlp_ratio=4, qkv_bias=False, dropout=None,
                 attn_dropout=None, path_dropout=None, activation=None, norm='layer'):
        super().__init__()

        if isinstance(norm, str):
            self.norm1 = make_norm_from_name(norm, dim, in_channels)
            self.norm2 = make_norm_from_name(norm, dim, in_channels)
        elif norm:
            self.norm1 = norm(in_channels)
            self.norm2 = norm(in_channels)
        else:
            self.norm1 = self.norm2 = None

        self.attn = Attention(dim, in_channels, nb_heads, qkv_bias, attn_dropout, dropout)
        self.path_dropout = _build_dropout(path_dropout, dim, path=True)

        mlp_hidden_dim = int(mlp_ratio * in_channels)
        # note - MLP dim may need correcting
        self.mlp = MLP(in_channels, in_channels, mlp_hidden_dim, 
                       activation=activation, dropout=dropout, dim=-1)

    def forward(self, x):
        xi = x # need to check if this actually works - maybe need deepcopy?
        if self.norm1:
            xi = self.norm1(x)
        xi = self.attn(xi)
        xi = self.path_dropout(xi)
        x += xi

        xi = x
        if self.norm2:
            xi = self.norm2(x)
        xi = self.mlp(xi)
        xi = self.path_dropout(xi)
        x += xi

        return x


class ViT(Module):
    """
    Vision transformer architecture, based on Ross Wightman's implementation
    (https://github.com/rwightman/pytorch-image-models/).

    References
    ----------
    ..[1] "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
           Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, 
           Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, 
           Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
           https://arxiv.org/abs/2010.11929
    """
    def __init__(self, dim, in_channels, out_channels, img_size, patch_size=16, embed_dim=786,
                 depth=12, nb_heads=12,  mlp_ratio=4, qkv_bias=True, 
                 representation_size=None, distilled=False,
                 dropout=None, attn_dropout=None, path_dropout=None, 
                 embed_layer=PatchEmbed, norm='layer', activation='GELU'):
        super().__init__()
        if isinstance(img_size, int):
            img_size = [img_size]
        if isinstance(patch_size, int):
            patch_size = [patch_size]

        self.out_channels = out_channels
        self.nb_features = self.embed_dim = embed_dim
        self.nb_tokens = 2 if distilled else 1

        self.patch_embed = embed_layer(dim, patch_size, in_channels, embed_dim)

        nb_patches = np.prod([img_size[i] // patch_size[i] for i in range(dim)])

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, nb_patches + self.nb_tokens, embed_dim))
        self.pos_dropout = _build_dropout(dropout, dim)

        path_dropout_list = [x.item() for x in torch.linspace(0, path_dropout, depth)] \
             if isinstance(path_dropout, (int, float)) else [None] * depth

        self.blocks = Sequential(*[
            ViTBlock(dim=embed_dim, nb_heads=nb_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout,
                     path_dropout=path_dropout_list[i], norm=norm, activation=activation)
        for i in range(depth)])
        self.norm = make_norm_from_name(norm, 1, embed_dim) # check dim is correct...

        if representation_size and not distilled:
            self.nb_features = representation_size
            self.pre_logits = Sequential(OrderedDict(
                fc = Linear(embed_dim, representation_size),
                act = make_activation_from_name('Tanh')
            ))
        else:
            self.pre_logits = None

        self.head = Linear(self.nb_features, out_channels) if out_channels > 0 else None
        if distilled:
            self.head_dist = Linear(self.embed_dim, self.out_channels) if self.out_channels > 0 else None

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat([cls_token, x], dim=1)
        else:
            x = torch.cat([cls_token, self.dist_token.expand(x.shape[0], -1, -1), x], dim=1)
        x += self.pos_embed
        if self.pos_dropout:
            x = self.pos_dropout(x)
        x = self.blocks(x)
        if self.norm:
            x = self.norm(x)
        if self.dist_token is None:
            if self.pre_logits:
                x = self.pre_logits(x[:, 0])
            else:
                x = x[:, 0]
        else:
            x = (x[:, 0], x[:, 1])
        
        if self.head_dist is not None:
            x, x_dist = x[0], x[1]
            if self.head:
                x = self.head(x)
                self.x_dist = self.head_dist(x_dist)
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        elif self.head:
            x = self.head(x)
        return x
        

class WindowAttention(Attention):
    def __init__(self, dim, window_size,
                 nb_heads=8, qkv_bias=False,
                 attn_dropout=None, proj_dropout=None):
        super().__init__(dim, nb_heads, qkv_bias, 
                         attn_dropout, proj_dropout)
        self.window_size = window_size

        coords = torch.stack(torch.meshgrid([torch.arange(w) for w in self.window_size]))
        coords_flat = coords.flatten(1) # should be of shape [dim, prod(*Spatial)]

        self.pos_bias_table = torch.nn.Parameter(
            torch.zeros(tuple([2*w-1 for w in self.window_size]) + (nb_heads,)))

        # TODO: figure out how to generalise coordinates to 3D...
        if dim==2:
            coords_rel = coords_flat[:, :, None] - coords_flat[:, None, :]
            coords_rel = coords_rel.permute(1, 2, 0).contiguous()
            coords_rel[:, :, 0] += self.window_size[0] - 1
            coords_rel[:, :, 1] += self.window_size[1] - 1
            coords_rel[:, :, 0] *= 2 * self.window_size[1] - 1
            pos_index = coords_rel.sum(-1)
        self.register_buffer('pos_index', pos_index)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nb_heads, C // self.nb_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        q *= self.scale
        attn = (q @ k.transpose(-2,-1))

        pos_bias = self.pos_bias_table[self.pos_index.view(-1)].view(tuple([np.prod(self.window_size) \
             for i in self.window_size]) + (-1,)) 
        if self.dim==2:
            pos_bias = pos_bias.permute(2, 0, 1).contiguous() 
        attn = attn + pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N).softmax(-1)
        else:
            attn = attn.softmax(-1)
        
        if self.attn_dropout:
            attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_dropout:
            x = self.proj_dropout

        return x


class SwinBlock(Module):
    def __init__(self, dim, in_channels, input_resolution, nb_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, dropout=None, attn_dropout=None, 
                 path_dropout=None, activation='GELU', norm='layer'):
        # TODO: make compatible with anisotropic window
        super().__init__()
        self.dim = dim
        if isinstance(input_resolution, int):
            input_resolution = [input_resolution] * dim
        if isinstance(window_size, int):
            window_size = [window_size] * dim
        if isinstance(shift_size, int):
            shift_size = [shift_size] * dim
        self.input_resolution = input_resolution
        self.nb_heads = nb_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= max(self.window_size):
            self.shift_size = [0] * dim
            self.window_size = [min(self.input_resolution)] * dim
        for i in range(dim):
            assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in range [0, window_size]"

        if isinstance(norm, str):
            self.norm1 = make_norm_from_name(norm, dim, in_channels)
            self.norm2 = make_norm_from_name(norm, dim, in_channels)
        elif norm:
            self.norm1 = norm(in_channels)
            self.norm2 = norm(in_channels)
        else:
            self.norm1 = self.norm2 = None

        self.attn = WindowAttention(dim, self.window_size, nb_heads, qkv_bias,
                                    attn_dropout, dropout)
        self.path_dropout = _build_dropout(path_dropout, dim, path=True)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = MLP(in_channels, in_channels, mlp_hidden_dim, 
                       activation=activation, dropout=dropout, dim=-1)

        if self.shift_size > 0:
            img_mask = torch.zeros((1,) + tuple(self.input_resolution) + tuple(1,))
            slices = [(slice(0, -self.window_size[i]),
                      slice(-self.window_size[i], -self.shift_size[i]),
                      slice(-self.shift_size[i], None)) for i in range(dim)]
            cnt = 0
            if dim==2:
                for h in slices[0]:
                    for w in slices[1]:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            elif dim==3:
                for h in slices[0]:
                    for w in slices[1]:
                        for d in slices[2]:
                            img_mask[:, h, w, d, :] = cnt
                            cnt += 1

            mask_windows = window_partition(img_mask, self.window_size, dim)
            mask_windows = mask_windows.view(-1, np.prod(self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        if self.dim==2:
            H, W = self.input_resolution
        elif self.dim==3:
            H, W, D = self.input_resolution
        B, L, C = x.shape

        assert L == np.prod(self.input_resolution), 'input feature has wrong size'

        shortcut = x
        if self.norm1:
            x = self.norm1(x)
        if self.dim==2:
            x = x.view(B, H, W, C)
        elif self.dim==3:
            x = x.view(B, H, W, D, C)
        
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=tuple([-s for s in self.shift_size]),
                                   dims=tuple([i+1 for i in range(self.dim)]))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size, self.dim)
        x_windows = x_windows.view(-1, np.prod(self.window_size), C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view((-1,) + tuple(self.window_size) + (C,))
        shifted_x = window_reverse(attn_windows, self.window_size, self.input_resolution)

        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=tuple(self.shift_size), 
                           dims=tuple([i+1 for i in range(self.dim)]))
        else:
            x = shifted_x
        x = x.view(B, np.prod(self.input_resolution), C)

        x = shortcut + self.path_dropout(x)
        if self.norm2:
            x += self.path_dropout(self.mlp(self.norm2(x)))
        else:
            x += self.path_dropout(self.mlp(x))

        return x


class SwinLayer(Module):
    def __init__(self, dim, in_channels, input_resolution, depth, nb_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, dropout=None, attn_dropout=None,
                 path_dropout=None, norm='layer', downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if isinstance(window_size, int):
            window_size = [window_size] * dim

        self.blocks = ModuleList([
            SwinBlock(dim, in_channels, input_resolution, nb_heads, window_size, 
            shift_size = 0 if (i % 2 == 0) else [w // 2 for w in window_size], mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout, norm=norm,
            path_dropout = path_dropout if isinstance(path_dropout, (list, tuple)) else path_dropout)
        for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, in_channels=in_channels, dim=dim, norm=norm)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Swin(Module):
    """
    Sliding window transformer.

    References
    ----------
    ..[1] "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
           Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, 
           Stephen Lin, Baining Guo
           https://arxiv.org/abs/2103.14030
    """
    def __init__(self, dim, img_size, patch_size=4, in_channels=1, out_channels=1,
                 embed_dim=96, depths=(2, 2, 6, 2), nb_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 dropout=None, attn_dropout=None, path_dropout=None,
                 norm='layer', ape=False, patch_norm=True,
                 use_checkpoint=False, pool=True, **kwargs):
        super().__init__()
        if isinstance(img_size, int):
            img_size = [img_size] * dim
        if isinstance(patch_size, int):
            patch_size = [patch_size] * dim

        self.in_channels = in_channels
        self.nb_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.nb_features = int(embed_dim * 2 ** (self.nb_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(dim, patch_size, in_channels, embed_dim,
                                      norm = norm if patch_norm else None)
        nb_patches = np.prod([img_size[i] // patch_size[i] for i in range(dim)])
        self.patch_grid = [img_size[i] // patch_size[i] for i in range(dim)]

        if self.ape:
            self.abs_pos_embed = torch.nn.Parameter(torch.zeros(1, nb_patches, embed_dim))
            # TODO: implement truncated normal for init
        else:
            self.abs_pos_embed = None

        self.pos_dropout = _build_dropout(dropout, dim)

        path_dropout_list = [x.item() for x in torch.linspace(0, path_dropout, len(depths))] \
             if isinstance(path_dropout, (int, float)) else [None] * len(depths)

        self.layers = Sequential(*[
            SwinLayer(dim, in_channels,
                      input_resolution=[p // (2 ** i_layer) for p in self.patch_grid],
                      depth=depths[i_layer], nb_heads=nb_heads[i_layer],
                      window_size=window_size, mlp_ratio=self.mlp_ratio,
                      qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout,
                      path_dropout=path_dropout_list[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                      norm=norm, downsample=PatchMerge if (i_layer < self.nb_layers - 1) else None,
                      use_checkpoint=use_checkpoint) for i_layer in range(self.nb_layers)])

        self.norm = make_norm_from_name(norm, dim, self.nb_features)
        self.pool = torch.nn.AdaptiveAvgPool1d(1) if pool else None
        self.head = Linear(self.nb_features, out_channels) if out_channels > 0 else None

    def forward(self, x):
        x = self.patch_embed(x)
        if self.abs_pos_embed is not None:
            x += self.abs_pos_embed
        if self.pos_dropout:
            x = self.pos_dropout(x)
        x = self.layers(x)
        if self.norm:
            x = self.norm(x)
        if self.pool:
            x = self.pool(x.transpose(1, 2)).flatten(1)
        if self.head:
            x = self.head(x)
        return x


# class UFormer(Module):
#     """
#     UFormer model with ViT-based encoder and decoder.

#     References
#     ----------
#     ..[1] ""
#     """


# class LinFormer(Module):
#     """
#     Transformer model with linear computational complexity.

#     References
#     ----------
#     ..[1]
#     """


# class UNETR(Module):
#     """
#     UNEt-TRansformer model.

#     References
#     ----------
#     ..[1]
#     """
