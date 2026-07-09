import torch
from .electrode import electrode_none_idx

def make_masks(chs, 
               x_type     =None,
               num_patchs = (193,15), 
               mN_x=2, mC_x=12, mN_y=1, mC_y=10,
               ):
    """
    parameters:
    - chs: channel map (B, 193)
    - num_patchs: (193, N)
    - mC_x: number of channels to mask (same for each time patch) 
    - mN_x: number of time points to mask
    - mC_y: number of patches as target
    - mN_y: number of patches as target
    
    returns:
    - mask_x: (B, mN_x, mC_x)  the indices of masked patch & channel
    - mask_y: (B, mN_y, mC_y)  the indices of masked patch & channel
    
    example:
        
    ```python

        B, C, N = 1, 6, 7
        num_patchs = (6, 7)
        chs = torch.tensor([[0,1,2,3,4,192],])#[0,192,2],

        make_masks(chs, num_patchs, mC_x=2, mC_y=2, mN_x=3, mN_y=2)
        
        tensor([[[ 0,  6, 12, 18, 24, 30, 36],
                [ 1,  7, 13, 19, 25, 31, 37],
                [ 2,  8, 14, 20, 26, 32, 38],
                [ 3,  9, 15, 21, 27, 33, 39],
                [ 4, 10, 16, 22, 28, 34, 40],
                [ 5, 11, 17, 23, 29, 35, 41]]])
        tensor([[[ 7,  8],
                [19, 20],
                [37, 38]]])
        tensor([[[24, 28],
                [30, 34]]])
    ```
    """
    
    
    B, C = chs.shape
    
    C, N = num_patchs
    
    used_chan_mask = (chs != electrode_none_idx)  # B, C
    
    # -- mask time patches --
    rvalues = ( torch.rand((B, N), device=chs.device) + 0.1 ) * ( 1.0 if x_type is None else x_type.float() )
    
    indices = torch.topk(rvalues, k = mN_x+mN_y, dim=-1, sorted=False)[1] # B, mN_x+mN_y
    
    indices = indices[:, torch.randperm(indices.shape[-1])].view(indices.size())
    rvalues = (torch.gather(rvalues, dim=-1, index=indices)>=0.1)*1.0
    
    indices_t_x = indices[:, :mN_x] # B, mN_x
    rvalues_t_x = rvalues[:, :mN_x] # B, mN_x
    
    indices_t_y = indices[:, mN_x:mN_x+mN_y] # B, mN_y
    rvalues_t_y = rvalues[:, mN_x:mN_x+mN_y] # B, mN_y
    
    # indices = torch.sort(indices, dim=-1, descending=True).indices # B, N
    # indices_t_x = torch.sort(indices[:, :mN_x],  dim=-1, descending=False).values # B, mN_x
    # indices_t_y = torch.sort(indices[:, mN_x:mN_x+mN_y], dim=-1, descending=False).values # B, mN_y
    
    # -- mask channel patches --
    rvalues = ( torch.rand((B, C), device=chs.device) + 0.1 ) * ( used_chan_mask.float() )
    
    indices = torch.topk(rvalues, k = mC_x+mC_y, dim=-1, sorted=False)[1] # B, mN_x+mN_y
    
    indices = indices[:, torch.randperm(indices.shape[-1])].view(indices.size())
    rvalues = (torch.gather(rvalues, dim=-1, index=indices)>=0.1)*1.0
    
    indices_c_x = indices[:, :mC_x] # B, mC_x
    rvalues_c_x = rvalues[:, :mC_x] # B, mC_x
    
    indices_c_y = indices[:, mC_x:mC_x+mC_y] # B, mC_y
    rvalues_c_y = rvalues[:, mC_x:mC_x+mC_y] # B, mC_y
    
    # indices = torch.sort(indices, dim=-1, descending=True).indices # B, N
    # indices_c_x = torch.sort(indices[:, :mC_x],  dim=-1, descending=False).values # B, mC_x
    # indices_c_y = torch.sort(indices[:, mC_x:mC_x+mC_y], dim=-1, descending=False).values # B, mC_y
    
    
    mask_x = indices_c_x.unsqueeze(1) + C * indices_t_x.unsqueeze(2)
    mask_y = indices_c_y.unsqueeze(1) + C * indices_t_y.unsqueeze(2)
    rval_x = rvalues_c_x.unsqueeze(1)     * rvalues_t_x.unsqueeze(2)
    rval_y = rvalues_c_y.unsqueeze(1)     * rvalues_t_y.unsqueeze(2)
    
    chan_pos_x = torch.gather(chs, dim=1, index=indices_c_x)
    time_pos_x = indices_t_x
    chan_pos_y = torch.gather(chs, dim=1, index=indices_c_y)
    time_pos_y = indices_t_y
    chan_pos_sum = torch.cat([chan_pos_x, chan_pos_y], dim=1)
    time_pos_sum = torch.cat([time_pos_x, time_pos_y], dim=1)
    
    
    chan_pos_x = chan_pos_x.unsqueeze_(1).repeat((1,mN_x,1)).flatten(1)
    time_pos_x = time_pos_x.unsqueeze_(2).repeat((1,1,mC_x)).flatten(1)
    chan_pos_y = chan_pos_y.unsqueeze_(1).repeat((1,mN_y,1)).flatten(1)
    time_pos_y = time_pos_y.unsqueeze_(2).repeat((1,1,mC_y)).flatten(1)
    
    return (mask_x    , rval_x    , mask_y    , rval_y    ),\
           (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
           (chan_pos_sum, time_pos_sum)


def make_masks_indp_chan(chs, 
               x_type     =None,
               num_patchs = (193,15), 
               mN_x=2, mC_x=12, mN_y=1, mC_y=10,
               ):
    """
    parameters:
    - chs: channel map (B, 193)
    - num_patchs: (193, N)
    - mC_x: number of channels to mask (same for each time patch) 
    - mN_x: number of time points to mask
    - mC_y: number of patches as target
    - mN_y: number of patches as target
    
    returns:
    - mask_x: (B, mN_x, mC_x)  the indices of masked patch & channel
    - mask_y: (B, mN_y, mC_y)  the indices of masked patch & channel
    
    example:
        
    ```python

        B, C, N = 1, 6, 7
        num_patchs = (6, 7)
        chs = torch.tensor([[0,1,2,3,4,192],])#[0,192,2],

        make_masks(chs, num_patchs, mC_x=2, mC_y=2, mN_x=3, mN_y=2)
        
        tensor([[[ 0,  6, 12, 18, 24, 30, 36],
                [ 1,  7, 13, 19, 25, 31, 37],
                [ 2,  8, 14, 20, 26, 32, 38],
                [ 3,  9, 15, 21, 27, 33, 39],
                [ 4, 10, 16, 22, 28, 34, 40],
                [ 5, 11, 17, 23, 29, 35, 41]]])
        tensor([[[ 7,  8],
                [19, 20],
                [37, 38]]])
        tensor([[[24, 28],
                [30, 34]]])
    ```
    """
    
    
    B, C = chs.shape
    
    C, N = num_patchs
    
    used_chan_mask = (chs != electrode_none_idx)  # B, C
    
    
    # -- mask time patches --
    rvalues = ( torch.rand((B, N), device=chs.device) + 0.1 ) * ( 1.0 if x_type is None else x_type.float() )
    indices = torch.topk(rvalues, k = mN_x+mN_y, dim=-1, sorted=False)[1] # B, mN_x+mN_y
    
    indices = indices[:, torch.randperm(indices.shape[-1])].view(indices.size())
    # print(rvalues)
    # rvalues = (rvalues[torch.arange(B).to(indices),indices]>=0.1)*1.0
    rvalues = (torch.gather(rvalues, dim=-1, index=indices)>=0.1)*1.0
    
    indices_t_x = indices[:, :mN_x] # B, mN_x
    rvalues_t_x = rvalues[:, :mN_x] # B, mN_x
    indices_t_y = indices[:, mN_x:mN_x+mN_y] # B, mN_y
    rvalues_t_y = rvalues[:, mN_x:mN_x+mN_y] # B, mN_y
    # print(rvalues_t_x, rvalues_t_y)
    
    
    # -- mask channel patches --
    rvalues = ( torch.rand((B, mN_x+mN_y, C), device=chs.device) + 0.1 ) * ( used_chan_mask.unsqueeze(1).float() )
    indices = torch.topk(rvalues, k = mC_x+mC_y, dim=-1, sorted=False)[1] # B, mN_x+mN_y
    # print(rvalues)
    
    indices = indices[:, :, torch.randperm(indices.shape[-1])].view(indices.size())
    rvalues = (torch.gather(rvalues, dim=-1, index=indices)>=0.1)*1.0
    
    indices_c_x = indices[:, :mN_x, :mC_x] # B, mN_x
    rvalues_c_x = rvalues[:, :mN_x, :mC_x] # B, mN_x
    
    indices_c_y = indices[:, mN_x:mN_x+mN_y, mC_x:mC_x+mC_y] # B, mN_y
    rvalues_c_y = rvalues[:, mN_x:mN_x+mN_y, mC_x:mC_x+mC_y] # B, mN_y
    
    # indices = torch.sort(indices, dim=-1, descending=True).indices # B, N, C
    # indices_c_x = torch.sort(indices[:, :mN_x, :mC_x],  dim=-1, descending=False).values # B, N, mC_x
    # indices_c_y = torch.sort(indices[:, mN_x:mN_x+mN_y, mC_x:mC_x+mC_y], dim=-1, descending=False).values # B, N, mC_y
    
    mask_x = indices_c_x + C * indices_t_x.unsqueeze(2)
    mask_y = indices_c_y + C * indices_t_y.unsqueeze(2)
    rval_x = rvalues_c_x     * rvalues_t_x.unsqueeze(2)
    rval_y = rvalues_c_y     * rvalues_t_y.unsqueeze(2)
    
    chan_pos_x = torch.gather(chs.unsqueeze(1).repeat((1,mN_x,1)), dim=2, index=indices_c_x).flatten(1)
    chan_pos_y = torch.gather(chs.unsqueeze(1).repeat((1,mN_y,1)), dim=2, index=indices_c_y).flatten(1)
    
    # print(chan_pos_x.shape)
    
    time_pos_sum = torch.cat([indices_t_x, indices_t_y], dim=1)
    
    time_pos_x = indices_t_x.unsqueeze_(2).repeat((1,1,mC_x)).flatten(1)
    time_pos_y = indices_t_y.unsqueeze_(2).repeat((1,1,mC_y)).flatten(1)
    
    return (mask_x    , rval_x    , mask_y    , rval_y    ),\
           (chan_pos_x, time_pos_x, chan_pos_y, time_pos_y),\
           (None, time_pos_sum)


def apply_mask(mask_x, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask_x: tensor [B, mN, mC] containing indices of patches in [B, N, C] to keep 
    """    
    B, N, C, D = x.shape
    assert len(mask_x.shape)==3 # each sample in batch has its own mask
    mB, mN, mC = mask_x.shape
    assert mB==B
    mask_keep = mask_x.flatten(1).unsqueeze(-1).expand(-1, -1, D)
    
    masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
    masked_x = masked_x.contiguous().view((B,mN,mC,D))
    return masked_x

def make_target(x, mask_y, num_patches, use_norm=False, use_norm_c=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask_y: tensor [B, mN, mC] containing indices of patches in [B, N, C] to keep 
    :return y: tensor [B, mN, mC, D]
    """
    with torch.no_grad():
        C, N = num_patches
        assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
        block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
        x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
        # 将维度重新排列以使分块沿着通道轴和空间轴
        x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
        x = x.view(x.shape[0], N, C, block_size_c * block_size_n)
        y = apply_mask(mask_y.to(x.device), x)
        if use_norm:
            y = torch.nn.functional.layer_norm(y, (y.size(-1),))
        if use_norm_c:
            y = y.transpose(-1,-2)
            y = torch.nn.functional.layer_norm(y, (y.size(-1),))
            y = y.transpose(-1,-2)
        return y