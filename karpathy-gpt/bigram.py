import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

"""
B: Batch element dimension
T: Time dimension, range(BLACK_SZ), which is the context length
C: Channel dimension, which is the embedding length or in general, the number of output logits of a layer
"""

class BigramLanguageModel(nn.Module):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tok = tokenizer
        self.device = device
        # For each char, store the probs of the next char
        self.token_embedding_table = nn.Embedding(tokenizer.vocab_sz, tokenizer.vocab_sz)
        #equal_dist = tensor([[1/vocab_size] * vocab_size] * vocab_size)
        #self.token_embedding_table = nn.Embedding.from_pretrained(equal_dist, freeze=False)

    def forward(self, idx, targets=None):
        """ nn.Module.__call__() calls forward(). """
        # idx: (B,T)
        # logits: (B,T,C), where C is the vocab size
        logits = self.token_embedding_table(idx)
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
            logits, _ = self(idx)
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