
import torch 
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

def main():
    torch.manual_seed(1337)
    B,T,C = 4,8,32 # batch, time, channel
    x = torch.randn(B,T,C)
    # bow means bag of words
    # Inefficient way to do a running average along time
    xbow = torch.zeros(B,T,C)
    for b in range(B):
        for t in range(T):
            xprev = x[b,:t+1] # (t,C)
            xbow[b,t] = torch.mean(xprev,0)

    # Efficent matrix way to do a running average along time
    wei = torch.tril(torch.ones(T,T)) 
    wei = wei / wei.sum(1, keepdim=True) 
    xbow2 = wei @ x # (B,T,C)
       
    # version 3: use Softmax, like later in the transformer affinities
    tril = torch.tril(torch.ones(T,T))
    wei = torch.zeros((T,T))
    wei = wei.masked_fill( tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    xbow3 = wei @ x # (B,T,C)    

    # lets see a single self-attention head
    head_sz = 16
    key = nn.Linear(C, head_sz, bias=False)
    query = nn.Linear(C, head_sz, bias=False)
    value = nn.Linear(C, head_sz, bias=False)
    k = key(x) # (B,T,head_sz)
    q = query(x) # (B,T,head_sz)
    # wei[r,c] is large if the key at time c is similar to the query at time r
    # Don't you want to norm k and q to be of length one?
    wei = q @ k.transpose(-2,-1) * head_sz**-0.5 # (B, T, head_sz) @ (B, head_sz, T) --> (B,T,T)
    # We don't look into the future, so mask out the upper triangle
    tril = torch.tril(torch.ones(T,T))
    wei = wei.masked_fill( tril == 0, float('-inf'))
    # make them sum to one
    wei = F.softmax(wei, dim=-1)
    v = value(x) # (B,T,head_sz)
    # Replace embedding at time t with the affinity weighted average of its predecessors in time
    xbow4 = wei @ v

    tt=42
main()