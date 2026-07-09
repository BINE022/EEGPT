import os
import torch
from torch import nn
import numpy as np
import copy
import random
from functools import partial
from .modules.encoder import Encoder, generate_attention_mask
from .modules.decoder import Decoder
from .modules.utils_mask import make_masks, apply_mask, make_target
from .modules.electrode import electrode_num
from .modules.projector import PredictionHead, ProjectionHead
from .modules.predictor import Predictor
from .modules.utils import deactivate_requires_grad

class MAE(nn.Module):
    """ MAE """
    def __init__(
        self,
        max_num_chan = 64,
        patch_size    = 64,
        output_num    = 2,
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
        
        enc_use_all_attn     =False,
        enc_use_indp_chan_sum=True,
        enc_use_indp_time_sum=True,
        
        dec_use_all_attn     =False,
        dec_use_indp_chan_sum=True,
        dec_use_indp_time_sum=True,
        
        use_cache_attn_mask  =True, # cache same attn_mask
        
        only_encoder = False,
        **kwargs
    ):
        super().__init__()
        # print(embed_dim, max_num_chan)
        
        self.chan_embed = nn.Embedding(max_num_chan, embed_dim)
        self.patch_size = patch_size
        args = {
            'chan_embed':self.chan_embed,
            'max_num_chan' : max_num_chan,
            'patch_size'    : patch_size,
            'output_num'    : output_num,
            'embed_dim'     : embed_dim,
            'embed_num'     : embed_num,
            'depth':depth,
            'num_heads':num_heads,
            'mlp_ratio':mlp_ratio,
            'qkv_bias':qkv_bias,
            'drop_rate':drop_rate,
            'attn_drop_rate':attn_drop_rate,
            'drop_path_rate':drop_path_rate,
            'norm_layer':norm_layer,
            'norm_out_layer':norm_out_layer,
            'init_std':init_std,
            'embed_max_norm':embed_max_norm,
            'interpolate_factor':interpolate_factor,
            'use_cache_attn_mask':use_cache_attn_mask,
        }
        
        args['use_all_attn']     =enc_use_all_attn
        args['use_indp_chan_sum']=enc_use_indp_chan_sum
        args['use_indp_time_sum']=enc_use_indp_time_sum
        self.encoder = Encoder(**args)
        if not only_encoder:
            args['use_all_attn']     =dec_use_all_attn
            args['use_indp_chan_sum']=dec_use_indp_chan_sum
            args['use_indp_time_sum']=dec_use_indp_time_sum
            self.decoder = Decoder(**args)

    def no_weight_decay(self,):
        return ['mask_token', 'summary_chan', 'summary_time', 'summary_both']
        
    def avg_reference(self, x):
        x          = x - x.mean(-2,keepdim = True)
        return x
    
    
    def prepare_data(self, x, 
                chan_ids, 
                x_type = None, 
                mNCxy = (3,8,3,8), 
                use_norm_t=True, 
                use_norm_c=True,
                mask_info=None,
                use_avg_ref=False, ):
        # x float  [B, C, T] (raw)
        # chan_ids [B, C]
        # x_type   [B, T]
        
        if x_type is not None:
            x_type   = x_type.to(x.device)
            x_type   = x_type.unfold(dimension= 1,size = self.patch_size,step = self.patch_size)
            x_type   = x_type.mean(-1)>0 # [B, N] bool
        else:
            x_type   = None
            
        mN_x, mC_x, mN_y, mC_y = mNCxy
        
        x        =  x.unfold(dimension=2,size=self.patch_size,step=self.patch_size).transpose(1,2) # [B, N, C, D]
        B, N, C, D = x.shape
        
        if mask_info is None:
            mask_info = \
                make_masks(chan_ids, x_type, [C,N], 
                           mN_x, mC_x, mN_y, mC_y)
        
        (mask_x    , rval_x    , mask_y    , rval_y    ),\
        (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
        (chan_pos_sum, time_pos_sum) = mask_info
        # # -- 仅截取 input_x 中用到的time_pos
        # time_pos_sum_x = time_pos_sum[:, :mN_x]
        # time_pos_sum_y = time_pos_sum[:, mN_x:]
        
        rval_x = rval_x.flatten(1,2).unsqueeze_(-1)
        rval_y = rval_y.flatten(1,2).unsqueeze_(-1)
                    
        input_x, target_y = apply_mask(mask_x, x), apply_mask(mask_y, x)
        
        if use_avg_ref:
            input_x  = self.avg_reference(input_x ).flatten(1,2)
            target_y = self.avg_reference(target_y).flatten(1,2)
        else:
            input_x  = input_x.flatten(1,2)
            target_y = target_y.flatten(1,2)
        
        target_y_c = target_y_t= target_y
        input_x_c  = input_x_t = input_x 
        
        if use_norm_t:
            with torch.autocast(device_type="cuda", enabled=False):
                target_y_t = torch.nn.functional.layer_norm(target_y, (target_y.size(-1),))
                input_x_t  = torch.nn.functional.layer_norm(input_x,  (input_x.size(-1),))

        if use_norm_c:
            target_y_c = target_y.transpose(-1,-2)
            input_x_c  = input_x.transpose(-1,-2)
            with torch.autocast(device_type="cuda", enabled=False):
                target_y_c = torch.nn.functional.layer_norm(target_y_c, (target_y_c.size(-1),))
                input_x_c  = torch.nn.functional.layer_norm(input_x_c,  (input_x_c.size(-1),))
            target_y_c = target_y_c.transpose(-1,-2)
            input_x_c  = input_x_c.transpose(-1,-2)
            
        target_y_c, target_y_t= target_y_c * rval_y, target_y_t * rval_y
        input_x_c , input_x_t = input_x_c  * rval_x, input_x_t  * rval_x
    
        return [(input_x, rval_x, chan_pos_x, time_pos_x, target_y_t, target_y_c), \
                (target_y, rval_y, chan_pos_y, time_pos_y, input_x_t, input_x_c)], \
                (chan_pos_sum, time_pos_sum), mask_info
               
    
    def classiy_forward(self, x, chan_ids, return_layers=0,
                use_avg_ref=True):
        
            
        x        =  x.unfold(dimension=2,size=self.patch_size,step=self.patch_size).transpose(1,2) # [B, N, C, D]
        B, N, C, D = x.shape
        
        chan_pos_x = chan_ids.to(x.device)
        
        time_pos_x = torch.arange(N).unsqueeze_(0).repeat((B,1)).to(dtype=torch.long, device=x.device)
        # print(time_pos_x.shape)
        
        chan_pos_sum = chan_pos_x.clone().detach()
        time_pos_sum = time_pos_x.clone().detach()
        
        chan_pos_x = chan_pos_x.unsqueeze_(1).repeat((1,N,1)).flatten(1)
        time_pos_x = time_pos_x.unsqueeze_(2).repeat((1,1,C)).flatten(1)
        
        
        input_x = x
        
        if use_avg_ref:
            input_x  = self.avg_reference(input_x ).flatten(1,2)
        else:
            input_x  = input_x.flatten(1,2)
        
        result   = self.encoder(input_x, 
                                chan_pos_x, 
                                time_pos_x, 
                                chan_pos_sum,
                                time_pos_sum)
        (z_chan, z_time, z_both) = result[0]
        if return_layers==1:
            r = z_chan
        if return_layers==2:
            r = z_time
        if return_layers==3:
            r = z_both
            
        return r
        
    def forward(self, x, 
                chan_ids, 
                x_type = None, 
                mNCxy = (3,8,3,8), 
                use_norm_t=True, 
                use_norm_c=True,
                mask_info=None,
                use_IdentityAE=False,
                use_avg_ref=False,
                use_siamese_forward=False,
                **argks):
        # x float  [B, C, T] (raw)
        # chan_ids [B, C]
        # x_type   [B, T]
        datas, (chan_pos_sum, time_pos_sum), mask_info =\
            self.prepare_data(x, chan_ids, x_type, mNCxy, 
                              use_norm_t, use_norm_c, mask_info, use_avg_ref)
            
        results = []
        for idx in range(2):#
            input_x, rval_x, chan_pos_x, time_pos_x, target_y_t, target_y_c = datas[idx]
            input_y, rval_y, chan_pos_y, time_pos_y, target_x_t, target_x_c = datas[1-idx]
        
            z_inter, (z,t), attn_mask_enc = self.encoder(input_x, 
                                                        chan_pos_x, 
                                                        time_pos_x, 
                                                        chan_pos_sum, 
                                                        time_pos_sum)
            # z,t = torch.zeros_like(z),torch.zeros_like(t)
            rs, attn_mask_dec = self.decoder(z, t, 
                                            chan_pos_y, 
                                            time_pos_y, 
                                            chan_pos_sum, 
                                            time_pos_sum)
            
            rs = (rs*rval_y) if (self.decoder.output_num < 1) else tuple([r*rval_y for r in rs])
            result = (target_y_t, target_y_c, rs, z_inter,
                      (mask_info, datas), (attn_mask_enc, attn_mask_dec), input_x)
            if not use_siamese_forward:
                return result
            results.append(result)


class CAE(nn.Module):
    """ CAE """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        embed_dim, predictor_embed_dim_inp = kwargs['embed_dim'], kwargs['predictor_embed_dim_inp']
        predictor_depth = kwargs['predictor_depth'] if 'predictor_depth' in kwargs else kwargs['depth']
        predictor_num_heads = kwargs['predictor_num_heads'] if 'predictor_num_heads' in kwargs else kwargs['num_heads']
        
        predictor_embed_dim_inp = embed_dim if predictor_embed_dim_inp is None else predictor_embed_dim_inp

        self.model = MAE(**kwargs)
        self.chan_embed = self.model.chan_embed
        self.encoder_momentum = copy.deepcopy(self.model.encoder)
        self.projection_head  = ProjectionHead(embed_dim,    embed_dim*2, predictor_embed_dim_inp)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        
        # self.prediction_head = PredictionHead(predictor_embed_dim_inp, embed_dim*2, predictor_embed_dim_inp)
        kwargs['predictor_embed_dim_out'] = kwargs['embed_dim']
        kwargs['embed_dim'] = predictor_embed_dim_inp
        kwargs['depth'] = predictor_depth
        kwargs['num_heads'] = predictor_num_heads
        kwargs['chan_embed']=self.model.chan_embed
        self.predictor = Predictor(**kwargs)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        
    def no_weight_decay(self,):
        return self.model.no_weight_decay()
    
    def forward(self, x, 
                chan_ids, 
                x_type = None, 
                mNCxy = (3,8,3,8), 
                use_norm_t=True, 
                use_norm_c=True,
                mask_info=None,
                use_IdentityAE=False,
                use_avg_ref=False, 
                use_siamese_forward=False,
                **kwargs):
        # x float  [B, C, T] (raw)
        # chan_ids [B, C]
        # x_type   [B, T]
        datas, (chan_pos_sum, time_pos_sum), mask_info =\
            self.model.prepare_data(x, chan_ids, x_type, mNCxy, 
                              use_norm_t, use_norm_c, mask_info, use_avg_ref)
        results = []

        mN_x, mC_x, mN_y, mC_y = mNCxy
        for idx in range(2):#
            input_x, rval_x, chan_pos_x, time_pos_x, target_y_t, target_y_c = datas[idx]
            input_y, rval_y, chan_pos_y, time_pos_y, target_x_t, target_x_c = datas[1-idx]
            if idx==0:
                time_pos_sum_x = time_pos_sum[:, :mN_x]
                time_pos_sum_y = time_pos_sum[:, mN_x:]
                chan_pos_sum_x = chan_pos_sum[:, :mC_x]
                chan_pos_sum_y = chan_pos_sum[:, mC_x:]
            else:
                time_pos_sum_y = time_pos_sum[:, :mN_x]
                time_pos_sum_x = time_pos_sum[:, mN_x:]
                chan_pos_sum_y = chan_pos_sum[:, :mC_x]
                chan_pos_sum_x = chan_pos_sum[:, mC_x:]
            
            (z1,t1), pz_y, attn_mask_enc1 = \
                            self.forward_encoder(   input_x, 
                                                    chan_pos_x, 
                                                    time_pos_x,
                                                    chan_pos_sum_x, 
                                                    time_pos_sum_x,
                                                    chan_pos_sum_y, 
                                                    time_pos_sum_y)
            
            pz_m =          self.forward_encoder_momentum(  input_y, 
                                                            chan_pos_y, 
                                                            time_pos_y,
                                                            chan_pos_sum_y, 
                                                            time_pos_sum_y)
                
            rs, attn_mask_dec = self.model.decoder( z1, t1, 
                                                    chan_pos_y, 
                                                    time_pos_y, 
                                                    chan_pos_sum_x, 
                                                    time_pos_sum_x)
            rs = (rs*rval_y) if (self.model.decoder.output_num < 1) else tuple([r*rval_y for r in rs])
            result = (target_y_t, target_y_c, rs, z1,
                      (mask_info, datas), (attn_mask_enc1, attn_mask_dec), 
                      (pz_y, pz_m), input_x)
            if not use_siamese_forward:
                return result
            results.append(result)
        return results
    
    def forward_encoder(self, 
                        input_x, 
                        chan_pos_x, 
                        time_pos_x,
                        chan_pos_sum_x, 
                        time_pos_sum_x,
                        chan_pos_sum_y, 
                        time_pos_sum_y,
                        ):
        
        z_inter, (z,t), attn_mask_enc = self.model.encoder(input_x, 
                                                    chan_pos_x, 
                                                    time_pos_x, 
                                                    chan_pos_sum_x, 
                                                    time_pos_sum_x)
        pz   = self.projection_head(z)
        pz_inter, (pz,pt), _ = self.predictor(pz, t, chan_pos_sum_y, time_pos_sum_y)
        pz_y = pz
        return (z,t), pz_y, attn_mask_enc

    def forward_encoder_momentum(self, 
                        input_x, 
                        chan_pos_x, 
                        time_pos_x,
                        chan_pos_sum, 
                        time_pos_sum):
        with torch.no_grad():
            z_inter, (z,t), attn_mask_enc = self.encoder_momentum(input_x, 
                                                        chan_pos_x, 
                                                        time_pos_x, 
                                                        chan_pos_sum, 
                                                        time_pos_sum)
            z = z[:, :(-self.encoder_momentum.embed_num_both)]
            pz_m = self.projection_head_momentum(z)
            pz_m = pz_m.detach()
        return pz_m
    

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
def main():
    
    B, C, T = 1, 6, 4
    mNCxy = (2,3,2,3) # mN_x, mC_x, mN_y, mC_y
    # mNCxy = (3,3,3,3)
    x =torch.arange(C*T).reshape((B, T, C)).transpose(1,2).float()
    print("x\n", x)
    device='cuda:0'
    model = MAE(
        max_num_chan  = C,
        patch_size    = 1,
        embed_dim     = 8,
        embed_num     = (2,2,2),
        depth         = 2,
        num_heads     = 2,
        enc_use_all_attn       = False,
        enc_use_indp_chan_sum  = False,
        enc_use_indp_time_sum  = False,
        dec_use_all_attn       = True,
        dec_use_indp_chan_sum  = False,
        dec_use_indp_time_sum  = False,
    ).to(device)
    chan_ids = torch.arange(C).unsqueeze(0).repeat((B,1))
    model.eval()
    with torch.no_grad():
        target_y_t1, target_y_c1, r1, z1, (mask_info1, datas), (attn_mask_enc, attn_mask_dec), x1 = \
            model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy,
                    use_norm_t=False, 
                    use_norm_c=False,)
        
    (mask_x    , rval_x    , mask_y    , rval_y    ),\
    (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
    (chan_pos_sum, time_pos_sum) = mask_info1
    
    with torch.no_grad():
        target_y_t2, target_y_c2, r2, z2, (mask_info1, datas), _, x2 = \
            model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy, 
                mask_info = mask_info1,
                use_norm_t=False, 
                use_norm_c=False,)
    print("mask\n", mask_x, '\n', mask_y)
    print("chan_pos\t\t", chan_pos_x, chan_pos_y)
    print("time_pos\t\t", time_pos_x, time_pos_y)
    print("chan_pos_sum\t\t", chan_pos_sum)
    print("time_pos_sum\t\t", time_pos_sum)
    from models.modules.utils import visualize_tensor
    if attn_mask_enc is not None:
        print("attn_mask_enc", attn_mask_enc.shape, '\n', visualize_tensor(attn_mask_enc[0,0].int()), \
            '\n\n', visualize_tensor(attn_mask_enc[0,1].int()), '\n')
    if attn_mask_dec is not None:
        print("attn_mask_dec", attn_mask_dec.shape, '\n', visualize_tensor(attn_mask_dec[0,0].int()), \
            '\n\n', visualize_tensor(attn_mask_dec[0,1].int()), '\n')
    print(torch.norm(r2[0]-r1[0]),
          torch.norm(r2[1]-r1[1]),)
    """
    x
    tensor([[[ 0.,  6., 12., 18.],
            [ 1.,  7., 13., 19.],
            [ 2.,  8., 14., 20.],
            [ 3.,  9., 15., 21.],
            [ 4., 10., 16., 22.],
            [ 5., 11., 17., 23.]]])
    mask
    tensor([[[12, 15, 16],
            [18, 21, 22]]], device='cuda:0') 
    tensor([[[ 1,  2,  5],
            [ 7,  8, 11]]], device='cuda:0')
    chan_pos                 tensor([[0, 3, 4, 0, 3, 4]], device='cuda:0') tensor([[1, 2, 5, 1, 2, 5]], device='cuda:0')
    time_pos                 tensor([[2, 2, 2, 3, 3, 3]], device='cuda:0') tensor([[0, 0, 0, 1, 1, 1]], device='cuda:0')
    chan_pos_sum             tensor([[0, 3, 4, 1, 2, 5]], device='cuda:0')
    time_pos_sum             tensor([[2, 3, 0, 1]], device='cuda:0')
    """

    # #########
    # enc_use_all_attn       = False,
    # enc_use_indp_chan_sum  = True,
    # enc_use_indp_time_sum  = True,
    # dec_use_all_attn       = False,
    # dec_use_indp_chan_sum  = True,
    # dec_use_indp_time_sum  = True,
    # ##########

    """
    attn_mask_enc torch.Size([1, 2, 28, 28]) 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  

    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  

    attn_mask_dec torch.Size([1, 2, 28, 28]) 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  

    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  
    """

    # #########
    # enc_use_all_attn       = False,
    # enc_use_indp_chan_sum  = False,
    # enc_use_indp_time_sum  = False,
    # ##########
    """
    attn_mask_enc torch.Size([1, 2, 28, 28]) 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  

    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  
    """
    # #########
    # dec_use_all_attn       = False,
    # dec_use_indp_chan_sum  = False,
    # dec_use_indp_time_sum  = False,
    # ##########
    """
    attn_mask_dec torch.Size([1, 2, 28, 28]) 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    ■, _, _, ■, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, ■, _, _, ■, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  

    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, _, _, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, _, _, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, ■, ■, ■, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    _, _, _, _, _, _, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■,  
    """

    # #########
    # dec_use_all_attn       = True,
    # dec_use_indp_chan_sum  = False,
    # dec_use_indp_time_sum  = False,
    # ##########

    """
    attn_mask_dec torch.Size([1, 2, 28, 28]) 
    ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    ...
    ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, ■, 
    """