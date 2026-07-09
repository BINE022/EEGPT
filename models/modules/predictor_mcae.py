import torch
import torch.nn as nn
import math
import math
from functools import partial
try:
    from .attention import RotaryEmbedding
    from .block import Block
    from .utils import trunc_normal_, generate_attention_mask, LinearWithConstraint
except:
    from models.modules.attention import RotaryEmbedding
    from models.modules.block import Block
    from models.modules.utils import trunc_normal_, generate_attention_mask, LinearWithConstraint


class Predictor(nn.Module):
    """ Predictor """
    def __init__(
        self,
        patch_size    = 64,
        embed_dim     = 768,
        embed_num     = (1,1,1),
        depth         = 6,
        predictor_embed_dim     = None,
        predictor_embed_dim_inp = None,
        predictor_depth         = None,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_out_layer=partial(nn.LayerNorm, eps=1e-6, elementwise_affine=False),
        init_std=0.02,
        embed_max_norm = -1.0,
        interpolate_factor = 2.,
        **kwargs
    ):
        super().__init__()
        predictor_embed_dim     = embed_dim if predictor_embed_dim     is None else predictor_embed_dim    
        predictor_embed_dim_inp = embed_dim if predictor_embed_dim_inp is None else predictor_embed_dim_inp
        predictor_embed_dim_out = predictor_embed_dim_inp
        # predictor_embed_dim_out = embed_dim if predictor_embed_dim_out is None else predictor_embed_dim_out
        predictor_depth     = depth     if predictor_depth is None     else predictor_depth
        self.patch_size= patch_size
        self.num_heads = num_heads
        self.embed_num = (0, embed_num[1], 0)
        self.embed_num_chan, self.embed_num_time, self.embed_num_both = self.embed_num
        
        self.predictor_embed = LinearWithConstraint(predictor_embed_dim_inp, predictor_embed_dim, 
                                                  doWeightNorm=(embed_max_norm>0.01), 
                                                  max_norm=embed_max_norm, bias=True)
        
        self.mask_token    = nn.Parameter(torch.zeros(1, self.embed_num_time, predictor_embed_dim), requires_grad=True)
        
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, predictor_depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (predictor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(
                                        dim = self.time_embed_dim,
                                        theta=10000,
                                        interpolate_factor=interpolate_factor)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=0,#dpr[i], 
                norm_layer=norm_layer,
                use_rope=True)
            for i in range(predictor_depth)])
        self.predictor_norm = norm_out_layer(predictor_embed_dim)
        
        self.predictor_proj = nn.Linear(predictor_embed_dim, predictor_embed_dim_out, bias=False)
        #--
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.mlp.fc1.weight.data, layer_id + 1)
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, z, time_pos_z_x, time_pos_z_y):
        """
            z        float [B, Nts*eNt, D]
            time_pos long  [B, M]
        """
        z  = self.predictor_embed(z)         # [B, Nts*eNt, D]
        B, Nx, _= z.shape
        
        tz_x = self.time_embed(time_pos_z_x) # [B, Nx, d]
        tz_y = self.time_embed(time_pos_z_y) # [B, Ny, d]
        _, Ny, _= tz_y.shape
        
        mask_token = self.mask_token.repeat((B,Ny//self.embed_num_time,1))
        x = torch.cat([z   , mask_token], dim=1)
        t = torch.cat([tz_x, tz_y      ], dim=1)
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, t, 
                    attn_mask=None) # B, Nx+Ny, D
        
        x = x[:,-Ny:,:]
        
        with torch.autocast(device_type='cuda', enabled=False):
            x = self.predictor_norm(x)
        
        x = self.predictor_proj(x)
        
        return x
    
def main():
    model = Predictor(
        max_num_chan = 64,
        patch_size    = 4,
        chan_embed    = None,
        embed_dim     = 64,
        embed_num     = (2,2,2),
        depth=2,
        num_heads=4,
    )
    z = torch.zeros((1,8,64))
    time_pos_z = torch.tensor([[0,1,2,3,2,1,3,4]])
    
    pred = model(z, time_pos_z)
    print(pred.shape)

if __name__=="__main__":
    main()
    