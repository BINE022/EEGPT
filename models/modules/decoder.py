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


class Decoder(nn.Module):
    """ Decoder """
    def __init__(
        self,
        max_num_chan = 64,
        patch_size    = 64,
        output_num    = 1,
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
        interpolate_factor = 2.,
        use_all_attn     =False,
        use_indp_chan_sum=True,
        use_indp_time_sum=True,
        use_cache_attn_mask=True,
        **kwargs
    ):
        super().__init__()
        self.patch_size= patch_size
        self.output_num= output_num
        self.num_heads = num_heads
        self.embed_num = embed_num
        self.embed_num_chan, self.embed_num_time, self.embed_num_both = embed_num
        
        self.use_all_attn      = use_all_attn
        self.use_indp_chan_sum = use_indp_chan_sum
        self.use_indp_time_sum = use_indp_time_sum
        self.use_cache_attn_mask = use_cache_attn_mask
        
        self.decoder_embed = LinearWithConstraint(embed_dim, embed_dim, 
                                                  doWeightNorm=(embed_max_norm>0.01), 
                                                  max_norm=embed_max_norm, bias=True)
        
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # -- 
        if chan_embed is None:
            self.chan_embed = nn.Embedding(max_num_chan, embed_dim)
        else:
            self.chan_embed = chan_embed
        # --
        self.time_embed_dim = (embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(
                                        dim = self.time_embed_dim,
                                        theta=10000,
                                        interpolate_factor=interpolate_factor)
        # --
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=0,#dpr[i], 
                norm_layer=norm_layer,)
            for i in range(depth)])
        self.decoder_norm = norm_out_layer(embed_dim)
        self.decoder_proj = nn.Linear(embed_dim, patch_size * max(1, output_num), bias=False)
        #--
        self.cache_attn_mask = []
        #--
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.decoder_blocks):
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
    
    def forward(self, z, t, chan_pos, time_pos, chan_pos_sum, time_pos_sum):
        """
            z        float [B, S, D]
            t        float [B, ...]
            chan_pos long  [B, N]
            time_pos long  [B, N]
        """
        
        z  = self.decoder_embed(z)     # [B, N, D]
        
        cp = self.chan_embed(chan_pos) # [B, N, D]
        tp = self.time_embed(time_pos) # [B, N, d]
        B, N, _= cp.shape
        
        x = self.mask_token.repeat((1,N,1)) + cp
        
        x = torch.cat([x,  z], dim=1)
        t = torch.cat([tp, t], dim=1)
        # -- 
        
        
        
        if self.use_cache_attn_mask:
            if len(self.cache_attn_mask)>=1:
                if self.cache_attn_mask[0] is None:
                    attn_mask = None
                else:
                    attn_mask = self.cache_attn_mask[0].repeat((B,1,1,1)).to(x.device)
            else:
                attn_mask = generate_attention_mask(
                    chan_pos, 
                    time_pos, 
                    chan_pos_sum, 
                    time_pos_sum,
                    self.embed_num,
                    self.num_heads,
                    use_all_attn     =self.use_all_attn,
                    use_indp_chan_sum=self.use_indp_chan_sum,
                    use_indp_time_sum=self.use_indp_time_sum,)
                if attn_mask is None:
                    self.cache_attn_mask.append(attn_mask)
                else:
                    self.cache_attn_mask.append(attn_mask[0:1])
        else:
            attn_mask = generate_attention_mask(
                    chan_pos, 
                    time_pos, 
                    chan_pos_sum, 
                    time_pos_sum,
                    self.embed_num,
                    self.num_heads,
                    use_all_attn     =self.use_all_attn,
                    use_indp_chan_sum=self.use_indp_chan_sum,
                    use_indp_time_sum=self.use_indp_time_sum,)
        # -- fwd prop
        for blk in self.decoder_blocks:
            x = blk(x, t, 
                    attn_mask=attn_mask) # B, NC, D
        
        z = x[:,:N,:] 
        
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.decoder_norm(z.float())
        z = self.decoder_proj(z)
        if self.output_num < 1: 
            zs = z
        else:
            zs= torch.split(z, self.patch_size, dim=-1)
        return zs, attn_mask
    
def main():
    model = Decoder(
        max_num_chan = 64,
        patch_size    = 4,
        chan_embed    = None,
        embed_dim     = 64,
        embed_num     = (2,2,2),
        depth=2,
        num_heads=4,
    )
    x = torch.zeros((1,4,4))
    chan_pos = torch.tensor([[1,0,5,6]])
    time_pos = torch.tensor([[2,1,3,4]])
    chan_pos_sum = torch.tensor([[1,0,3,4,5]])
    time_pos_sum = torch.tensor([[1,0,3]])
    
    pred = model(x, chan_pos, time_pos, chan_pos_sum, time_pos_sum)
    print(pred[0].shape, pred[1].shape, pred[2].shape)

if __name__=="__main__":
    main()
    