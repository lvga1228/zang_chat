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

# from nanochat.common import get_dist_info, print0
# from nanochat.muon import Muon, DistMuon
# from nanochat.adamw import DistAdamW

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
    out = out.to(x.dtype)
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
        assert self.n_kvhead <= self.n_head and self.n_head % self.n_kvhead == 0
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
        #输出的预测结果，lm_head 是 GPT 模型的最后一层线性层，用来把 Transformer 输出的隐状态向量（embedding）映射到词表（vocab size），从而预测下一个 token 的概率。
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        #
        self.rotary_seq_len = 10*config.sequence_len
        head_dim = config.n_embd//config.n_head
        cos,sin = self._precompute_rotary_embeddings(self.rotary_seq_len,head_dim)
        self.register_buffer('cos',cos,persistent=False)
        self.register_buffer('sin',sin,persistent=False)
    
    def get_device(self):
        return self.transformer['wte'].weight.device

    def _precompute_rotary_embeddings(self,seq_len,head_dim,base=10000,device=None):
        if device is None:
            device = self.transformer['wte'].weight.device
        
        channel_range = torch.arange(0,head_dim,2,dtype=torch.float32,device=device)     #深度学习一般使用float32
        inv_freq = 1.0/(base ** (channel_range/head_dim))
        t = torch.arange(seq_len,dtype=torch.float32,device=device)
        fres = torch.outer(t,inv_freq)
        cos=fres.cos()
        sin =fres.sin()
        cos,sin = cos.bfloat16(),sin.bfloat16()
        cos,sin = cos[None,:,None,:] , sin[None,:,None,:]
        return cos,sin
    
    def _init_weights(self,module):# 初始化我们不通过模块的weights

        #这个 _init_weights 函数是 GPT 模型中的权重初始化函数，它的作用是：为所有 Linear 层 和 Embedding 层 设置初始参数（初始化权重）。让模型在训练一开始就处于一个“稳定、不容易崩溃”的状态。
        if isinstance(module,nn.Linear):
            fan_out = module.weight.size(0) #✔ Linear(in_features, out_features) ✔ weight.shape = (out_features, in_features)
            fan_in = module.weight.size(1)
            std = 1/math.sqrt(fan_in)*min(1.0,math.sqrt(fan_out/fan_in)) ## https://arxiv.org/pdf/2310.17813 初始论文
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=1.0)

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer['h']:
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        
        head_dim = self.config.n_embd//self.config.n_head
        self.cos,self.sin = self._precompute_rotary_embeddings(self.config.sequence_len,head_dim,device=self.get_device())

        if self.transformer['wte'].weight.device.type == "cuda": #word token embedding 
            self.transformer['wte'].to(dtype=torch.bfloat16)

    def estimate_flops(self): # 用来 估算 GPT 模型每生成 1 个 token 时大概需要多少 FLOPs（浮点运算次数）.ref:https://arxiv.org/abs/2204.02311
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer['wte'].weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self,idx,targets=None,kv_cache=None,loss_reduction='mean'):
        B,T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0=0 if kv_cache is None else kv_cache.get_pos()
        cos_sin  = self.cos[:,T0:T0+T],self.sin[:,T0:T0+T]
        x = self.transformer['wte'](idx)
        x = norm(x)
        for block in self.transformer['h']:
            x = block(x,cos_sin,kv_cache)
        x = norm(x)

        #compute the logits

        softcap = 15
        if targets is not None:
            logits= self.lm_head(x)
            logits = softcap * torch.tanh(logits/softcap)
            logits=logits.float() 
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1,reduction =loss_reduction)
            return loss
        else:
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits
        

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


    
