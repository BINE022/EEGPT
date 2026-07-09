import torch
import torch.nn as nn
import math

def rotate_half(x):
    """Rotates half the feature dimensions."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    """Applies rotary embeddings to input tensor.
    : freqs float [B, N, head_dim//2]
    : t     float [B, num_head, N, head_dim]
    """
    freqs     = freqs.to(t)
    rot_dim   = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    freqs = freqs.unsqueeze(1) # [B, N, 1, head_dim//2]
    # Apply rotary embeddings with scaling
    t = ( (t * freqs.cos()) + (rotate_half(t) * freqs.sin()) ) * scale # [B, NC, num_head, head_dim]
    return torch.cat((t_left, t, t_right), dim=-1)

################################# RoPE ######################################
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta=10000,
        interpolate_factor=1.0,
        offset = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.offset = offset
        # Initialize frequency parameters
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(freqs.unsqueeze(0).unsqueeze(0), requires_grad=False) # [1,1,d]
        
        assert interpolate_factor >= 1.0, "Interpolation factor must be >= 1.0"
        self.interpolate_factor = interpolate_factor

    def forward(self, time_pos, device='cuda', dtype=torch.float):
        """ Prepares rotary frequencies
        : time_pos long [B, N]
        
        example usage:
        ```python
            time_embed = RotaryEmbedding(dim=4, interpolate_factor=1.0)
            time_embed.prepare_freqs((2,3), device='cpu').contiguous().view((3, 2, 4))
            # tensor([[[0.0000, 0.0000, 0.0000, 0.0000],
            #          [0.0000, 0.0000, 0.0000, 0.0000]],

            #         [[1.0000, 1.0000, 0.0100, 0.0100],
            #          [1.0000, 1.0000, 0.0100, 0.0100]],

            #         [[2.0000, 2.0000, 0.0200, 0.0200],
            #          [2.0000, 2.0000, 0.0200, 0.0200]]])
        ```
        """
            
        time_pos = (( time_pos + self.offset ) / self.interpolate_factor).to(dtype=dtype, device=device)
        freqs    = self.freqs.to(dtype=dtype, device=device)
        freqs    = time_pos.unsqueeze(-1) * freqs
        freqs    = freqs.repeat_interleave(repeats=2, dim=-1)    # (n_seq_pos, n_freqs*2)

        return freqs
    
################################# Attention ######################################
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        use_rope=False,
        return_attention=False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.use_rope  = use_rope
        self.is_causal = is_causal
        
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.return_attention = return_attention

    def forward(self, x, freqs=None, attn_mask=None, return_attn_weights=None):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0) # q,k,v: [B,num_heads,T,head_dim]
        
        if self.use_rope:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
            
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
        
        if type(return_attn_weights)==list:
            if attn_mask is not None:
                attn_maak = torch.zeros(q.size(-2), q.size(-2)).to(dtype=q.dtype, device=q.device)
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return_attn_weights.append(attn_weight)
        
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask, # [B, num_heads, T, T]
            dropout_p=self.attn_drop if self.training else 0,
            is_causal=self.is_causal)

        x = y.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__=="__main__":
    embed_dim = 8
    num_heads = 2
    head_dim  = (embed_dim//num_heads)
    N, C      = 3,2
    B         = 2
    
    time_embed = RotaryEmbedding(dim=head_dim//2, interpolate_factor=1.0)
    
    # time_pos = torch.arange(6).unsqueeze(0) # [1,6]
    time_pos = torch.tensor([0,0,1,1,2,2], dtype=torch.long).unsqueeze(0)
    q        = torch.ones((B,N*C,num_heads,head_dim))
    
    freqs = time_embed(time_pos, device='cpu')
    print(freqs)
    """
    tensor([[0., 0.],
        [0., 0.],
        [1., 1.],
        [1., 1.],
        [2., 2.],
        [2., 2.]])
    """
    
    q = apply_rotary_emb(freqs, q)
    print(q) 
    """
    tensor([
        [[[ 1.0000,  1.0000,  1.0000,  1.0000],
          [ 1.0000,  1.0000,  1.0000,  1.0000]],

         [[ 1.0000,  1.0000,  1.0000,  1.0000],
          [ 1.0000,  1.0000,  1.0000,  1.0000]],

         [[-0.3012,  1.3818,  1.0000,  1.0000],
          [-0.3012,  1.3818,  1.0000,  1.0000]],

         [[-0.3012,  1.3818,  1.0000,  1.0000],
          [-0.3012,  1.3818,  1.0000,  1.0000]],

         [[-1.3254,  0.4932,  1.0000,  1.0000],
          [-1.3254,  0.4932,  1.0000,  1.0000]],

         [[-1.3254,  0.4932,  1.0000,  1.0000],
          [-1.3254,  0.4932,  1.0000,  1.0000]]],


        [[[ 1.0000,  1.0000,  1.0000,  1.0000],
          [ 1.0000,  1.0000,  1.0000,  1.0000]],

         [[ 1.0000,  1.0000,  1.0000,  1.0000],
          [ 1.0000,  1.0000,  1.0000,  1.0000]],

         [[-0.3012,  1.3818,  1.0000,  1.0000],
          [-0.3012,  1.3818,  1.0000,  1.0000]],

         [[-0.3012,  1.3818,  1.0000,  1.0000],
          [-0.3012,  1.3818,  1.0000,  1.0000]],

         [[-1.3254,  0.4932,  1.0000,  1.0000],
          [-1.3254,  0.4932,  1.0000,  1.0000]],

         [[-1.3254,  0.4932,  1.0000,  1.0000],
          [-1.3254,  0.4932,  1.0000,  1.0000]]]])"""