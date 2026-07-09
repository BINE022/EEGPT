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
    """ models/modules/predictor.py """
    def __init__(
        self,
        max_num_chan = 64,
        predictor_embed_dim_out    = 64,
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
        **kwargs
    ):
        super().__init__()
        self.predictor_embed_dim_out= predictor_embed_dim_out
        self.num_heads = num_heads
        self.embed_num = embed_num
        self.embed_num_chan, self.embed_num_time, self.embed_num_both = embed_num
        
        self.predictor_embed = LinearWithConstraint(embed_dim, embed_dim, 
                                                  doWeightNorm=(embed_max_norm>0.01), 
                                                  max_norm=embed_max_norm, bias=True)
        
        # self.mask_token    = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.summary_chan  = nn.Parameter(torch.zeros(1, self.embed_num_chan, embed_dim), requires_grad=True)
        self.summary_time  = nn.Parameter(torch.zeros(1, self.embed_num_time, embed_dim), requires_grad=True)
        
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
        self.predictor_blocks = nn.ModuleList([
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
        self.predictor_norm = norm_out_layer(embed_dim)
        self.predictor_proj = nn.Linear(embed_dim, predictor_embed_dim_out, bias=False)
        #--
        self.init_std = init_std
        trunc_normal_(self.summary_chan, std=self.init_std)
        trunc_normal_(self.summary_time, std=self.init_std)
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
    
    def forward(self, z, t, chan_pos_sum_y, time_pos_sum_y, verbose=False):
        """
            z        float [B, S, D]
            t        float [B, ...]
            chan_pos long  [B, N]
            time_pos long  [B, N]
        """
        
        z  = self.predictor_embed(z)     # [B, N, D]
        B, N, D = z.shape
        
        cps = self.chan_embed(chan_pos_sum_y) # [B, Ncs, D]
        tps = self.time_embed(time_pos_sum_y) # [B, Nts, d]
        Ncs, Nts = cps.shape[1], tps.shape[1]
        
        cps      = torch.repeat_interleave(cps, repeats=self.embed_num_chan, dim=1)
        tps_time = torch.repeat_interleave(tps, repeats=self.embed_num_time, dim=1)
        
        summary_chan = self.summary_chan.repeat((B, Ncs, 1)) + cps
        summary_time = self.summary_time.repeat((B, Nts, 1))
        
        tps_chan     = torch.zeros(summary_chan.shape[:2]+(self.time_embed_dim,), 
                                   dtype = tps_time.dtype, device=tps_time.device)
        
        x = torch.cat([z,  summary_chan, summary_time], dim=1)
        t = torch.cat([t,  tps_chan,     tps_time    ], dim=1)
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, t, 
                    attn_mask=None) # B, NC, D
        
        z = x[:,N:,:] 
        
        with torch.autocast(device_type='cuda', enabled=False):
            z = self.predictor_norm(z.float())
        z = self.predictor_proj(z)
        
        a,b    = 0,(Ncs * self.embed_num_chan)
        z_chan = z[:, a:b].reshape((B, Ncs, self.embed_num_chan, D))
        if verbose:print(z.shape, a,b)
        a,b    = b,((Nts * self.embed_num_time)+b)
        z_time = z[:, a:b].reshape((B, Nts, self.embed_num_time, D))
        if verbose:print(z.shape, a,b)
        # a,b    = b,((self.embed_num_both)+b)
        # z_both = z[:, a:b].reshape((B, 1,   self.embed_num_both, D))
        # if verbose:print(z.shape, a,b)
        return (z_chan, z_time, None), (z,t), None
    
def main():
    model = Predictor(
        max_num_chan = 64,
        predictor_embed_dim_out    = 4,
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
    