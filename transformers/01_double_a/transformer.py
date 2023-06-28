import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

"""
B: Batch element dimension
T: Time dimension, range(BLACK_SZ), which is the context length
C: Channel dimension, which is the embedding length or in general, the number of output logits of a layer
"""

class TransformerModel(nn.Module):
    def __init__(self, tokenizer, embed_sz, num_layers, num_heads, block_sz, dropout):
        super().__init__()
        self.tok = tokenizer
        self.embed_sz = embed_sz
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_sz = block_sz
        self.dropout = dropout
        # For each char, store the probs of the next char
        self.token_embedding_table = nn.Embedding(tokenizer.vocab_sz, self.embed_sz)
        self.position_embedding_table = nn.Embedding(block_sz, self.embed_sz)
        head_sz = self.embed_sz//self.num_heads
        self.blocks = nn.Sequential(
            *[Block(num_heads, embed_sz , block_sz, head_sz, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_sz,elementwise_affine=True)
        self.lm_head = nn.Linear(self.embed_sz, tokenizer.vocab_sz)

    def forward(self, inp, targets=None):
        """ nn.Module.__call__() calls forward(). """
        B,T = inp.shape
        tok_emb = self.token_embedding_table( inp) # (B,T,C)
        pos_emb = self.position_embedding_table( torch.arange( T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
            return logits, loss
        else:
            # Pytorch wants the logits to be (B,C,T) for cross_entropy
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Targets is an index that is supposed to be largest in the logits vector,
            # for each token in each input of the batch.
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt, stoptoken=None, max_new_tokens=100):
        """ Generate from a prompt """
        # Add a fake batch dimension
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)

        for _ in range(max_new_tokens):
            # get the predictions
            prompt = prompt[:, -self.block_sz:] # (B,block_sz) limit input to block_sz
            logits, loss = self(prompt)
            logits = logits[:,-1,:] # B,C because we only take the last token
            probs = F.softmax(logits, dim=-1)
            #next = torch.multinomial(probs, num_samples=1)
            _,next = torch.max(probs, dim=1)
            next = next.unsqueeze(-1)
            prompt = torch.cat([prompt, next], dim=1)
            if next[0].tolist() == stoptoken:
                break
        out = self.tok.decode(prompt[0].tolist())
        return out

    
class Head(nn.Module):
    """ One head of self attention """
    def __init__(self,embed_sz,block_sz,head_sz, dropout):
        super().__init__()
        self.key = nn.Linear(embed_sz, head_sz, bias=False)
        self.query = nn.Linear(embed_sz, head_sz, bias=False)
        self.value = nn.Linear(embed_sz, head_sz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_sz,block_sz))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_sz)
        q = self.query(x) # (B,T,head_sz)
        v = self.value(x) # (B,T,head_sz)
        # compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        # Decoders only look into the past, so we mask the future
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T,T)

        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # maybe apply this to wei @ v instead?

        out = wei @ v # (B,T,T) @ (B,T,head_sz) --> (B,T,head_sz)
        return out

class MultiHead(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz, dropout):
        super().__init__()
        self.heads = nn.ModuleList( [Head(embed_sz, block_sz, head_sz, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear( embed_sz, embed_sz)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a nonlinearity """
    def __init__(self, n_inout, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inout, 4 * n_inout),
            nn.ReLU(),
            nn.Linear( 4 * n_inout, n_inout),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication (sa_heads) followed by computation (ffwd) """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz, dropout):
        super().__init__()
        self.sa_heads = MultiHead(num_heads, embed_sz, block_sz, head_sz, dropout)
        self.ffwd = FeedForward(embed_sz, dropout)    
        self.ln1 = nn.LayerNorm(embed_sz,elementwise_affine=True)
        self.ln2 = nn.LayerNorm(embed_sz,elementwise_affine=True)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    

