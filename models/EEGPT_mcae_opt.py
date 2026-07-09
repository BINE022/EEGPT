import os
import torch
from torch import nn
import numpy as np
import random
import copy
from functools import partial
from .modules.encoder_mcae import Encoder, generate_attention_mask
from .modules.decoder_mcae import Decoder
from .modules.predictor_mcae import Predictor
from .modules.utils_mask import make_masks_indp_chan, apply_mask
from .modules.electrode import electrode_num
from .modules.projector import PredictionHead, ProjectionHead
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
        only_encoder = False,
        **kwargs
    ):
        super().__init__()
        # print(embed_dim, max_num_chan)
        self.chan_embed = nn.Embedding(max_num_chan, embed_dim)
        self.patch_size = patch_size
        self.embed_num = (0, embed_num[1], 0)
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
        }
        self.encoder = Encoder(**args)
        if not only_encoder:
            self.decoder = Decoder(**args)

    def no_weight_decay(self,):
        return ['mask_token', 'summary_time']
    
    def avg_reference(self, x):
        x          = x - x.mean(-2,keepdim = True)
        return x
    
    
    def classiy_forward(self, x, chan_ids, use_avg_ref=True, use_time_indp_attn=False, *args, **argks):
        x        =  x.unfold(dimension=2,size=self.patch_size,step=self.patch_size).transpose(1,2) # [B, N, C, D]
        B, N, C, D = x.shape
        
        chan_pos_x = chan_ids.to(x.device)
        
        time_pos_x = torch.arange(N).unsqueeze_(0).repeat((B,1)).to(dtype=torch.long, device=x.device)
  
        time_pos_sum = time_pos_x.clone().detach()
        
        chan_pos_x = chan_pos_x.unsqueeze_(1).repeat((1,N,1)).flatten(1)
        time_pos_x = time_pos_x.unsqueeze_(2).repeat((1,1,C)).flatten(1)
        
        input_x = x
        
        if use_avg_ref:
            input_x  = self.avg_reference(input_x ).flatten(1,2)
        else:
            input_x  = input_x.flatten(1,2)
        
        result = self.encoder(input_x, 
                         chan_pos_x, 
                         time_pos_x,
                         time_pos_sum,
                         use_time_indp_attn=use_time_indp_attn,
                         )
        z = result[0]
        return z
        
    
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
                make_masks_indp_chan(chan_ids, x_type, [C,N], 
                                     mN_x, mC_x, mN_y, mC_y)
        
        (mask_x    , rval_x    , mask_y    , rval_y    ),\
        (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
        (_,          time_pos_sum) = mask_info
        # -- 仅截取 input_x 中用到的time_pos
        time_pos_sum_x = time_pos_sum[:, :mN_x]
        time_pos_sum_y = time_pos_sum[:, mN_x:]
        
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
    
        return [(input_x, rval_x, chan_pos_x, time_pos_x, time_pos_sum_x, target_y_t, target_y_c), \
                (target_y, rval_y, chan_pos_y, time_pos_y, time_pos_sum_y, input_x_t, input_x_c)], mask_info
               
    
    def forward(self, x, 
                chan_ids, 
                x_type = None, 
                mNCxy = (3,8,3,8), 
                use_norm_t=True, 
                use_norm_c=True,
                mask_info=None,
                use_IdentityAE=False,
                use_avg_ref=False, 
                use_time_indp_attn=False,
                use_siamese_forward=False):
        # x float  [B, C, T] (raw)
        # chan_ids [B, C]
        # x_type   [B, T]
        datas, mask_info =\
            self.prepare_data(x, chan_ids, x_type, mNCxy, 
                              use_norm_t, use_norm_c, mask_info, use_avg_ref)
        results = []
        for idx in range(2):
            input_x, rval_x, chan_pos_x, time_pos_x, time_pos_sum_x, target_y_t, target_y_c = datas[idx]
            rval_y, chan_pos_y, time_pos_y, time_pos_sum_y = datas[1-idx][1:5]
            z, attn_mask_enc = self.encoder(input_x, 
                                            chan_pos_x, 
                                            time_pos_x,
                                            time_pos_sum_x,
                                            use_time_indp_attn=use_time_indp_attn,
                                            )
            z          = z.flatten(1,2)
            time_pos_z_x = torch.repeat_interleave(time_pos_sum_x, repeats=self.embed_num[1], dim=1)
            
            rs, attn_mask_dec= self.decoder(z, 
                                            time_pos_z_x, 
                                            chan_pos_y, 
                                            time_pos_y)
            rs = (rs*rval_y) if (self.decoder.output_num < 1)  else (tuple([r*rval_y for r in rs]))
            result = (target_y_t, target_y_c, rs, z, 
                      (mask_info, datas), (attn_mask_enc, attn_mask_dec), 
                      input_x)
            if not use_siamese_forward:
                return result
            results.append(result)
        return results

class CAE(nn.Module):
    """ CAE """
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
        
        
        predictor_embed_dim     = None,
        predictor_embed_dim_inp = None,
        predictor_depth         = None,
        use_predictor       = True,
        use_combine_z       = True,
        
        **kwargs
    ):
        super().__init__()
        predictor_embed_dim_inp = embed_dim if predictor_embed_dim_inp is None else predictor_embed_dim_inp
        args = {
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
            'predictor_embed_dim':predictor_embed_dim,
            'predictor_embed_dim_inp':predictor_embed_dim_inp,
            'predictor_depth':predictor_depth,
            'use_predictor':use_predictor,
            'use_combine_z':use_combine_z,
        }
        self.model = MAE(**args)
        self.chan_embed = self.model.chan_embed
        self.use_predictor = use_predictor
        self.use_combine_z = use_combine_z
        self.encoder_momentum = copy.deepcopy(self.model.encoder)
        self.projection_head  = ProjectionHead(embed_dim,    embed_dim*2, predictor_embed_dim_inp)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        
        if use_predictor:
            self.predictor       = Predictor(**args)
        else:
            self.prediction_head = PredictionHead(predictor_embed_dim_inp, embed_dim*2, predictor_embed_dim_inp)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        
    def no_weight_decay(self,):
        return ['mask_token', 'summary_time']
    
    def forward(self, x, 
                chan_ids, 
                x_type = None, 
                mNCxy = (3,8,3,8), 
                use_norm_t=True, 
                use_norm_c=True,
                mask_info=None,
                use_IdentityAE=False,
                use_avg_ref=False, 
                use_time_indp_attn=False,
                use_siamese_forward=False):
        # x float  [B, C, T] (raw)
        # chan_ids [B, C]
        # x_type   [B, T]
        datas, mask_info =\
            self.model.prepare_data(x, chan_ids, x_type, mNCxy, 
                              use_norm_t, use_norm_c, mask_info, use_avg_ref)
        results = []
        for idx in range(2):#
            input_x, rval_x, chan_pos_x, time_pos_x, time_pos_sum_x, target_y_t, target_y_c = datas[idx]
            input_y, rval_y, chan_pos_y, time_pos_y, time_pos_sum_y, target_x_t, target_x_c = datas[1-idx]
            
            z, pz_y, time_pos_z_x, attn_mask_enc = \
                              self.forward_encoder(         input_x, 
                                                            chan_pos_x, 
                                                            time_pos_x,
                                                            time_pos_sum_x,
                                                            time_pos_sum_y,
                                                            use_time_indp_attn=use_time_indp_attn,)
            if self.use_predictor:
                z_m, pz_m       = self.forward_encoder_momentum(input_y, 
                                                                chan_pos_y, 
                                                                time_pos_y,
                                                                time_pos_sum_y,
                                                                use_time_indp_attn=use_time_indp_attn,)
            else:
                z_m, pz_m       = self.forward_encoder_momentum(input_x, 
                                                                chan_pos_x, 
                                                                time_pos_x,
                                                                time_pos_sum_x,
                                                                use_time_indp_attn=use_time_indp_attn,)
                
            rs, attn_mask_dec= self.model.decoder(z, 
                                            time_pos_z_x, 
                                            chan_pos_y, 
                                            time_pos_y)
            rs = (rs*rval_y) if (self.model.decoder.output_num < 1) else tuple([r*rval_y for r in rs])
            result = (target_y_t, target_y_c, rs, z,
                      (mask_info, datas), (attn_mask_enc, attn_mask_dec), 
                      (pz_y, pz_m), input_x)
            if not use_siamese_forward:
                return result
            results.append(result)
        return results
    
    def forward_encoder(self, 
                        input_x, 
                        chan_pos_x, 
                        time_pos_x,
                        time_pos_sum_x,
                        time_pos_sum_y,
                        use_time_indp_attn):
        z, attn_mask_enc = self.model.encoder(  input_x, 
                                                chan_pos_x, 
                                                time_pos_x,
                                                time_pos_sum_x,
                                                use_time_indp_attn=use_time_indp_attn,)
        
        z    = z.flatten(1,2)
        pz   = self.projection_head(z)
        
        time_pos_z_x = torch.repeat_interleave(time_pos_sum_x, repeats=self.model.embed_num[1], dim=1)
        
        pz_y  = None
        if self.use_predictor:
            time_pos_z_y = torch.repeat_interleave(time_pos_sum_y, repeats=self.model.embed_num[1], dim=1)
            pz_y         = self.predictor(pz, time_pos_z_x, time_pos_z_y)
            if self.use_combine_z:
                z            = torch.cat([z, pz_y], dim=1)
                time_pos_z_x = torch.cat([time_pos_z_x, time_pos_z_y], dim=1)
        else:
            pz_y         = self.prediction_head(pz)
        return z, pz_y, time_pos_z_x, attn_mask_enc

    def forward_encoder_momentum(self, 
                        input_x, 
                        chan_pos_x, 
                        time_pos_x,
                        time_pos_sum,
                        use_time_indp_attn):
        with torch.no_grad():
            z, attn_mask_enc = self.encoder_momentum(  input_x, 
                                                    chan_pos_x, 
                                                    time_pos_x,
                                                    time_pos_sum,
                                                    use_time_indp_attn=use_time_indp_attn,)
            z_m  = z.flatten(1,2)
            pz_m = self.projection_head_momentum(z_m)
            pz_m = pz_m.detach()
        return z_m, pz_m
    
    




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
    use_time_indp_attn = True
    use_siamese_forward = True
    x =torch.arange(C*T).reshape((B, T, C)).transpose(1,2).float()
    chan_ids = torch.arange(C).unsqueeze(0).repeat((B,1))
    print("x\n", x)
    
    device='cuda:0'
    
    model = CAE(
        max_num_chan  = C,
        patch_size    = 1,
        embed_dim     = 64,
        embed_num     = (1,1,1),
        output_num    = 0,
        depth         = 2,
        num_heads     = 4,
    ).to(device)
    model.eval()
    with torch.no_grad():
        results1 = \
            model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy,
                use_norm_t=True, 
                use_norm_c=True,#False,
                use_time_indp_attn=use_time_indp_attn,
                use_siamese_forward=use_siamese_forward)
        
    if type(results1) == tuple:
        results1 = [results1]
    for target_y_t1, target_y_c1, r1, z1, (mask_info1, datas), _, (pr, pz_m), x1 in results1:
        print(pr.shape, pz_m.shape, pr.mean(), pz_m.mean())
        
        (mask_x    , rval_x    , mask_y    , rval_y    ),\
        (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
        (chan_pos_sum, time_pos_sum) = mask_info1
        
        print("mask\n", mask_x, '\n', mask_y)
        # print("chan_pos\t\t", chan_pos_x, chan_pos_y)
        # print("time_pos\t\t", time_pos_x, time_pos_y)
        # print("time_pos_sum\t\t", time_pos_sum)
        
        for i, (input_x, rval_x, chan_pos_x, time_pos_x, time_pos_sum_x, target_y_t, target_y_c) in enumerate(datas):
            print(i,"input_x\t\t", input_x, rval_x)
            print(i,"input_pos\t\t", chan_pos_x, time_pos_x)
            print(i,"target_y\t\t", target_y_t, target_y_c)
            print(i,"time_pos_sum\t\t", time_pos_sum_x)
            
        break
    # for target_y_t, target_y_c, rs, z, (mask_info, datas), (attn_mask_enc, attn_mask_dec), (pr, pz_m), input_x in results:
    """
    torch.Size([1, 2, 32]) torch.Size([1, 2, 32]) tensor(-0.0131, device='cuda:0') tensor(-0.0195, device='cuda:0')
    torch.Size([1, 2, 32]) torch.Size([1, 2, 32]) tensor(-0.0124, device='cuda:0') tensor(-0.0144, device='cuda:0')
    """
    ######################################################################
    print("MAE=============================================")
    model = MAE(
        max_num_chan  = C,
        patch_size    = 1,
        embed_dim     = 64,
        embed_num     = (1,1,1),
        output_num    = 0,
        depth         = 2,
        num_heads     = 4,
    ).to(device)
    model.eval()
    with torch.no_grad():
        results1 = \
            model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy,
                use_norm_t=False, 
                use_norm_c=False,
                use_time_indp_attn=use_time_indp_attn,
                use_siamese_forward=use_siamese_forward)
        
    if type(results1) == tuple:
        results1 = [results1]
    for target_y_t1, target_y_c1, r1, z1, (mask_info1, datas), _, x1 in results1:
        
        (mask_x    , rval_x    , mask_y    , rval_y    ),\
        (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
        (chan_pos_sum, time_pos_sum) = mask_info1
        
        print("mask\n", mask_x, '\n', mask_y)
        # print("chan_pos\t\t", chan_pos_x, chan_pos_y)
        # print("time_pos\t\t", time_pos_x, time_pos_y)
        # print("time_pos_sum\t\t", time_pos_sum)
        
        for i, (input_x, rval_x, chan_pos_x, time_pos_x, time_pos_sum_x, target_y_t, target_y_c) in enumerate(datas):
            print(i,"input_x\t\t", input_x, rval_x)
            print(i,"input_pos\t\t", chan_pos_x, time_pos_x)
            print(i,"target_y\t\t", target_y_t, target_y_c)
            print(i,"time_pos_sum\t\t", time_pos_sum_x)
            
        break
    """
    x
    tensor([[[ 0.,  6., 12., 18.],
            [ 1.,  7., 13., 19.],
            [ 2.,  8., 14., 20.],
            [ 3.,  9., 15., 21.],
            [ 4., 10., 16., 22.],
            [ 5., 11., 17., 23.]]])
    mask
    tensor([[[ 0,  3,  5],
            [13, 16, 17]]], device='cuda:0') 
    tensor([[[ 7,  9, 10],
            [18, 22, 23]]], device='cuda:0')
    0 input_x                tensor([[[ 0.],
            [ 3.],
            [ 5.],
            [13.],
            [16.],
            [17.]]], device='cuda:0')
    0 input_pos              tensor([[0, 3, 5, 1, 4, 5]], device='cuda:0') tensor([[0, 0, 0, 2, 2, 2]], device='cuda:0')
    0 target_y               tensor([[[ 7.],
            [ 9.],
            [10.],
            [18.],
            [22.],
            [23.]]], device='cuda:0') tensor([[[ 7.],
            [ 9.],
            [10.],
            [18.],
            [22.],
            [23.]]], device='cuda:0')
    0 time_pos_sum           tensor([[0, 2]], device='cuda:0')
    1 input_x                tensor([[[ 7.],
            [ 9.],
            [10.],
            [18.],
            [22.],
            [23.]]], device='cuda:0')
    1 input_pos              tensor([[1, 3, 4, 0, 4, 5]], device='cuda:0') tensor([[1, 1, 1, 3, 3, 3]], device='cuda:0')
    1 target_y               tensor([[[ 0.],
            [ 3.],
            [ 5.],
            [13.],
            [16.],
            [17.]]], device='cuda:0') tensor([[[ 0.],
            [ 3.],
            [ 5.],
            [13.],
            [16.],
            [17.]]], device='cuda:0')
    1 time_pos_sum           tensor([[1, 3]], device='cuda:0')
    """
    
    if 1:
        for a in mask_y.cpu().flatten().tolist():
            x[x==a] = torch.rand((1,))
        print("x\n", x)
        
        with torch.no_grad():
            results = \
                model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy, 
                    mask_info = mask_info1,
                    use_norm_t=False, 
                    use_norm_c=False,
                    use_time_indp_attn=use_time_indp_attn,
                    use_siamese_forward=use_siamese_forward)
        
        if type(results) == tuple:
            results = [results]
        for (target_y_t1, target_y_c1, r1, z1, (mask_info1, datas), _, x1),\
            (target_y_t2, target_y_c2, r2, z2, _, _, x2) in zip(results1, results):
    
            print(torch.norm(z1-z2).item(),
                torch.norm(target_y_t1-target_y_t2).item(),
                torch.norm(target_y_c1-target_y_c2).item(),
                torch.norm(r1-r2).item(),
                torch.norm(x1-x2).item(),)
        """
        x
        tensor([[[ 0.0000,  6.0000, 12.0000, 18.0000],
                [ 1.0000,  0.9554, 13.0000, 19.0000],
                [ 0.6044,  8.0000, 14.0000, 20.0000],
                [ 0.5231,  0.1013, 15.0000, 21.0000],
                [ 0.6209,  0.5859, 16.0000, 22.0000],
                [ 5.0000, 11.0000, 17.0000, 23.0000]]])
        0.0 14.961575508117676 14.961575508117676 0.0 0.0
        3.526459217071533 0.0 0.0 0.015888266265392303 14.961575508117676
        """
    else:
        for a in mask_x[0,0].cpu().flatten().tolist():
            x[x==a] = torch.rand((1,))
        print("x\n", x)
        
        with torch.no_grad():
            results = \
                model(x.to(device), chan_ids.to(device) , mNCxy = mNCxy, 
                    mask_info = mask_info1,
                    use_norm_t=False, 
                    use_norm_c=False,
                    use_time_indp_attn=use_time_indp_attn,
                    use_siamese_forward=use_siamese_forward)
        
        if type(results) == tuple:
            results = [results]
            
        for (target_y_t1, target_y_c1, r1, z1, (mask_info1, datas), _, x1),\
            (target_y_t2, target_y_c2, r2, z2, _, _, x2) in zip(results1, results):
            print(torch.norm(z1-z2).item(),
                torch.norm(target_y_t1-target_y_t2).item(),
                torch.norm(target_y_c1-target_y_c2).item(),
                torch.norm(r1-r2).item(),
                torch.norm(x1-x2).item(),)
            
            print(torch.norm(z1-z2, dim=-1))
    
    
    
        """
        x
        tensor([[[0.0000e+00, 6.0000e+00, 1.2000e+01, 1.8000e+01],
                [1.5269e-02, 7.0000e+00, 1.3000e+01, 1.9000e+01],
                [4.8848e-01, 8.0000e+00, 1.4000e+01, 2.0000e+01],
                [3.0000e+00, 9.0000e+00, 1.5000e+01, 2.1000e+01],
                [4.0000e+00, 1.0000e+01, 1.6000e+01, 2.2000e+01],
                [6.7086e-01, 1.1000e+01, 1.7000e+01, 2.3000e+01]]])
        3.0070104598999023 0.0 0.0 0.012582574971020222 4.689976215362549
        tensor([[3.0070, 0.0000]], device='cuda:0')
        0.0 4.689976215362549 4.689976215362549 0.0 0.0
        tensor([[0., 0.]], device='cuda:0')
        """