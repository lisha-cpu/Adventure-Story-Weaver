import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd,n_head,block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd,4*n_embd), nn.GELU(),
                               nn.Linear(4*n_embd,n_embd))
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=3):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size,n_embd)
        self.position_embedding = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd,n_head,block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        B,T = idx.shape
        if T > self.block_size:
            idx = idx[:,-self.block_size:]
            T = idx.shape[1]
            if targets is not None: targets = targets[:,-T:]
        pos = torch.arange(T,device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=300,temperature=0.9,top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:] / max(temperature,1e-5)
            if top_k:
                values,_ = torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits < values[:,-1,None]] = float("-inf")
            probs = F.softmax(logits,dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,next_id),dim=1)
        return idx
