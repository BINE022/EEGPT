import torch
import torch.nn as nn
import math
from .attention import Attention

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
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


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 use_rope  = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(
                            dim, 
                            num_heads=num_heads, 
                            qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, 
                            proj_drop=drop,
                            is_causal = False,
                            use_rope  = use_rope)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = MLP(in_features=dim, 
                             hidden_features=int(dim * mlp_ratio), 
                             act_layer=act_layer, drop=drop)

    def forward(self, x, t, attn_mask=None, return_attn_weights=None):
        
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.norm1(x.float())
        z = self.attn(z, t, attn_mask, return_attn_weights)
        x = x + self.drop_path(z)
        
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.norm2(x.float())
        z = self.mlp(z)
        x = x + self.drop_path(z)
        
        return x


