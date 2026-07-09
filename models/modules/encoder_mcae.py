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

class Encoder(nn.Module):
    """ Encoder """
    def __init__(
        self,
        max_num_chan = 64,
        patch_size    = 64,
        chan_embed    = None,
        embed_dim     = 768,
        embed_num     = (1,1,1),
        depth=6,
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
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_num = (0, embed_num[1], 0)
        self.embed_num_chan, self.embed_num_time, self.embed_num_both = self.embed_num
        # --
        self.encoder_embed = LinearWithConstraint(patch_size, embed_dim, 
                                                  doWeightNorm=(embed_max_norm>0.01), 
                                                  max_norm=embed_max_norm, bias=True)
        
        self.summary_time  = nn.Parameter(torch.zeros(1, self.embed_num_time, embed_dim), requires_grad=True)
        
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # -- 
        if chan_embed is None:
            self.chan_embed = nn.Embedding(max_num_chan, embed_dim)
        else:
            self.chan_embed = chan_embed
        # --
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=0,#dpr[i], 
                norm_layer=norm_layer,
                use_rope=False)
            for i in range(depth)])
        self.encoder_norm = norm_out_layer(embed_dim)
        #--
        self.init_std = init_std
        trunc_normal_(self.summary_time, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.encoder_blocks):
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
    
    def forward(self, x, chan_pos, time_pos, time_pos_sum, use_time_indp_attn=False, return_attn_weights=None, verbose=False):
        """
            x        float [B, NC, D]
            chan_pos long  [B, NC]
            time_pos_sum long  [B, N]
        """
        
        x  = self.encoder_embed(x)     # [B, N, D]
        B, N, D= x.shape
        
        cp = self.chan_embed(chan_pos) # [B, N, D]
        
        x  = x + cp
        
        Nts = time_pos_sum.shape[1]
        
        attn_mask = None
        
        if use_time_indp_attn:
            C  = N // Nts
            x  = x.reshape((B, Nts, C, D)).flatten(0,1)
            
            summary_time = self.summary_time.repeat((B*Nts, 1, 1))
            
            x = torch.cat([x,  summary_time], dim=1)
            
            # -- fwd prop
            for blk in self.encoder_blocks:
                x = blk(x, None, 
                        attn_mask=None, 
                        return_attn_weights=return_attn_weights) # B, NC, D
            
            z = x[:,-self.embed_num_time:]
            
        else:
            summary_time = self.summary_time.repeat((B, Nts, 1))
            
            x = torch.cat([x,  summary_time], dim=1)
            # -- 
            attn_mask = generate_attention_mask(
                None, 
                time_pos, 
                None, 
                time_pos_sum,
                self.embed_num,
                num_heads        =self.num_heads,
                num_heads_chan   =0,
                num_heads_time   =self.num_heads,
                use_all_attn     =False,
                use_indp_chan_sum=True,
                use_indp_time_sum=True,)
            
            # -- fwd prop
            for blk in self.encoder_blocks:
                x = blk(x, None, 
                        attn_mask=attn_mask, 
                        return_attn_weights=return_attn_weights) # B, NC, D
            
            z = x[:,N:]
        
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.encoder_norm(z)
        
        z_time = z.reshape((B, Nts, self.embed_num_time, D))
        if verbose:print(z.shape)
        return z_time, attn_mask
    
def main():
    model = Encoder(
        max_num_chan = 64,
        patch_size    = 4,
        chan_embed    = None,
        embed_dim     = 64,
        embed_num     = (2,2,2),
        depth=2,
        num_heads=4,
    )
    x = torch.zeros((1,4,4))
    chan_pos = torch.tensor([[0,1,2,3]])
    time_pos = torch.tensor([[1,1,2,2]])
    time_pos_sum = torch.tensor([[1,2]])
    
    pred, attn_mask = model(x, chan_pos, time_pos, time_pos_sum, use_time_indp_attn=True, verbose=True)
    print(pred.shape)

if __name__=="__main__":
    main()
    