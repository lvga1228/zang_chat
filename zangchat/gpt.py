"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

class GPT_Config:
    sequence_len :int =1024
    vocab_size :int = 50304
    n_head :int = 6
    n_kvhead: int = 6
    n_embd :int = 768
    n_layer :int = 12


def norm(x):
    return F.rms_norm(x,(x.size(-1),))

def apply_rotary_emb(x,cos,sin):
    assert x.ndim == 4
    #x.shape = (B, H, T, D)
    d = x.shape[-1]//2
    x1 = x[:,:,:,:d]
    x2 = x[:,:,:,d:]
    y1=x1*cos+x2*sin
    y2=x1*(-sin)+x2*cos
    out = torch.cat([y1,y2],dim=3)
    out.to(x.dtype)
    return out

class CasualSelfAttention(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kvhead = config.n_kvhead
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.head_dim == 0
        assert self.n_kvhead <= self.n_head and self.n_head % self.n_kvhead
        self.c_q = nn.Linear(self.n_embd,self.head_dim*self.n_head,bias=False)
        self.c_k = nn.Linear(self.n_embd,self.head_dim*self.n_kvhead,bias=False)
        self.c_v = nn.Linear(self.n_embd,self.head_dim*self.n_kvhead,bias=False)
        self.c_proj = nn.Linear(self.n_embd,self.n_embd,bias=False) 

    def forward(self,x,cos_sin:tuple,kv_cache):
        B,T,C = x.size()
        q = self.c_q(x).view(B,T,self.n_head,self.head_dim)
        k = self.c_k(x).view(B,T,self.n_kvhead,self.head_dim)
        v = self.c_v(x).view(B,T,self.n_kvhead,self.head_dim)

        cos,sin = cos_sin
        q = apply_rotary_emb(q,cos,sin)
        k = apply_rotary_emb(k,cos,sin)
        q,k= norm(q),norm(k)
        q,k,v= q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)    # # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
    
        if kv_cache is not None:
            k,v=kv_cache.insert_kv(self.layer_idx,k,v)

        Tq = q.size(2)
        Tk = k.size(2)

        #attention：q会按照自回归的机制来关注k/v\
        enable_gqa = self.n_head!=self.n_kvhead
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=enable_gqa)
        elif Tq==1:
            y = F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=enable_gqa)
        else:
            att_mask = torch.zeros((Tq,Tk),dtype=torch.bool,device=q.device) 
            prefix = Tk - Tq
            att_mask[:,:prefix] = True
            att_mask[:,prefix:] = torch.tril(torch.ones((Tq,Tq),dtype=torch.bool,device=q.device))
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=att_mask,enable_gqa=enable_gqa)
        
        y = y.transpose(1,2).contiguous().view(B,T,-1)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fn = nn.Linear(config.n_embd,4*config.n_embd,bias=False)
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd,bias=False)

    def forward (self,x):
        x = self.c_fn(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config,layer_idx):
        super().__init__()
        self.attn = CasualSelfAttention(config=config,layer_idx=layer_idx)
        self.mlp = MLP(config=config)

    def forward (self,x,cos_sin,kv_cache):
        x = x+self.attn(norm(x),cos_sin,kv_cache)
        x = x+self.mlp(norm(x))
        return x

        
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        #使用moduledic，像定义一个dic一样定义一个模型中的不同模组
        self.transformer = nn.ModuleDict({
            'wte':nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.n_embd),
            'h':nn.ModuleList([Block(config,layer_idx) for layer_idx in range(config.n_layer)])
            })
        #输出的预测结果
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        #
        self.rotary_seq_len = 10*config.sequence_len
        head_dim = config.n_embd//config.n_head
        cos,sin = 

    def _precompute_rotary_embeddings(self,seq_len,head_dim,base=10000,device=None):
        if device is None:
            device = self.transformer['wte'].weight.device
        
        channel_range = torch.arange(0,head_dim,2,dtype=torch.float32,device=device)     #深度学习一般使用float32
        inv_freq = 1.0/(base ** (channel_range/head_dim))
        t = torch.arange(seq_len,step=1,dtype=torch.float32,device=device)
        fres = torch.outer(t,inv_freq)
        cos=fres.cos()
        cos,

