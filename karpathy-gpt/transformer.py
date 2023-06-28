import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

"""
B: Batch element dimension
T: Time dimension, range(BLACK_SZ), which is the context length
C: Channel dimension, which is the embedding length or in general, the number of output logits of a layer
"""
# Model hyperparameters
BATCH_SZ = 64

# context length; this is just an upper limit because the tril matrix needs allocating.
# You can feed in shorter sequences, see generate()
BLOCK_SZ = 256 

EMBED_SZ = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2

class Head(nn.Module):
    """ One head of self attention """
    def __init__(self,embed_sz,block_sz,head_sz):
        super().__init__()
        self.key = nn.Linear(embed_sz, head_sz, bias=False)
        self.query = nn.Linear(embed_sz, head_sz, bias=False)
        self.value = nn.Linear(embed_sz, head_sz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_sz,block_sz))))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_sz)
        q = self.query(x) # (B,T,head_sz)
        # compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,head_sz)
        out = wei @ v # (B,T,T) @ (B,T,head_sz) --> (B,T,head_sz)
        return out

class MultiHead(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz):
        super().__init__()
        self.heads = nn.ModuleList( [Head(embed_sz, block_sz, head_sz) for _ in range(num_heads)])
        self.proj = nn.Linear( embed_sz, embed_sz)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a nonlinearity """
    def __init__(self, n_inout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inout, 4 * n_inout),
            nn.ReLU(),
            nn.Linear( 4 * n_inout, n_inout),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication (sa_heads) followed by computation (ffwd) """
    def __init__(self, num_heads, embed_sz, block_sz, head_sz):
        super().__init__()
        self.sa_heads = MultiHead(num_heads, embed_sz, block_sz, head_sz)
        self.ffwd = FeedForward(embed_sz)    
        self.ln1 = nn.LayerNorm(embed_sz)
        self.ln2 = nn.LayerNorm(embed_sz)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    

class TransformerModel(nn.Module):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tok = tokenizer
        self.device = device
        self.block_sz = BLOCK_SZ
        # For each char, store the probs of the next char
        self.token_embedding_table = nn.Embedding(tokenizer.vocab_sz, EMBED_SZ)
        self.position_embedding_table = nn.Embedding(BLOCK_SZ, EMBED_SZ)
        #num_heads=4
        #num_layers = 6
        head_sz = EMBED_SZ//NUM_HEADS
        self.blocks = nn.Sequential(
            *[Block(NUM_HEADS, EMBED_SZ, BLOCK_SZ, head_sz) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_SZ)
        self.lm_head = nn.Linear(EMBED_SZ, tokenizer.vocab_sz)

    def forward(self, idx, targets=None):
        """ nn.Module.__call__() calls forward(). """
        B,T = idx.shape
        tok_emb = self.token_embedding_table( idx) # (B,T,C)
        pos_emb = self.position_embedding_table( torch.arange( T, device=self.device)) # (T,C)
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
            # Targets is an integer that is supposed to be largest in the logits vector,
            # for each token in each input of the batch.
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """ idx: (B,T) """
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond = idx[:, -self.block_sz:] # (B,block_sz) cut memory to block_sz
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] # B,C because we only take the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx    

    def generate_one(self, maxlen=100):
        """ Start by feeding a single zero, which happpens to be a newline """
        context = torch.zeros((1,1),dtype=torch.long, device=self.device)
        res = self.tok.decode(self.generate(context, maxlen)[0].tolist())
        return res