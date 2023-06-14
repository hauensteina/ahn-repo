import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch import tensor

WORDS = open('names.txt','r').read().splitlines()
_chars = sorted(list(set(''.join(WORDS))))
STOI = {s:i+1 for i,s in enumerate(_chars)}
STOI['.'] = 0
ITOS = { i:s for s,i in STOI.items()}
VSZ = len(STOI)
EMBED_SZ = 10
CONTEXT_SZ = 3
HIDDEN_SZ = 64

def build_dataset(words, context_size):
    X,Y = [],[]
    for w in words:
        #print(w)
        context = [0] * context_size
        for ch in w + '.':
            ix = STOI[ch]
            X.append(context) # left context
            Y.append(ix) # next char
            #print(''.join([ITOS[i] for i in context]), '--->', ITOS[ix])
            context = context[1:] + [ix] 

    X = tensor(X)
    Y = tensor(Y)   
    print(f'''Traing set size: {len(X)}''')
    return X,Y     

def cmp(s, dt, t):
    """ Compare manual gradients to pytorch gradients """
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approx: {str(app):5s} | maxdiff: {maxdiff:.4f}')

def main():
    import random
    random.seed(42)
    random.shuffle(WORDS)
    n1 = int(len(WORDS) * 0.8)
    n2 = int(len(WORDS) * 0.9)
    Xtr, Ytr = build_dataset(WORDS[:n1], CONTEXT_SZ) # 80% of the data
    Xdev, Ydev = build_dataset(WORDS[n1:n2], CONTEXT_SZ) # 10% of the data
    Xte, Yte = build_dataset(WORDS[n2:], CONTEXT_SZ) # 10% of the data
 
    g = torch.Generator().manual_seed(42)
    # Define the net
    C = torch.randn((VSZ, EMBED_SZ), generator=g) # embedding matrix
    # 5/3 / sqrt(fan_in) for a tanh nonlinearity 
    W1 = torch.randn((EMBED_SZ * CONTEXT_SZ, HIDDEN_SZ), generator=g) * (5/3) / ((EMBED_SZ * CONTEXT_SZ) ** 0.5)
    b1 = torch.randn((HIDDEN_SZ), generator=g) * 0.1
    W2 = torch.randn((HIDDEN_SZ, VSZ), generator=g) * 0.1
    b2 = torch.randn((VSZ), generator=g) * 0.1

    # Batchnorm parameters    
    bngain = torch.randn((1,HIDDEN_SZ )) * 0.1 + 1.0
    bnbias = torch.zeros((1,HIDDEN_SZ)) * 0.1

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    nparms = sum([p.numel() for p in parameters])
    print(f'Number of parameters: {nparms}')
    for p in parameters:
        p.requires_grad = True

    # Training
    BATCH_SIZE = 32
    MAX_STEPS = 200000
    lossi = []

    for i in range(MAX_STEPS):
        n = BATCH_SIZE
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        # forward pass
        emb = C[Xb] # (train_set_size, context_size, embedding_size)
        embcat = emb.view(emb.shape[0], -1) # (train_set_size, context_size * embedding_size)   
        hprebn = embcat @ W1 + b1 # (train_set_size, hidden_size)

        # BatchNorm layer
        #-----------------------------------------------------------
        bnmeani = hprebn.mean(0, keepdim=True)
        bndiff = hprebn - bnmeani   
        bndiff2 = bndiff ** 2
        bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5  # 1 / sigma
        bnraw = bndiff * bnvar_inv  #  (x - mu) / sigma
        hpreact = bngain * bnraw + bnbias

        # Non-linearity
        h = torch.tanh(hpreact) # (train_set_size, hidden_size)
        logits = h @ W2 + b2 # (train_set_size, vocab_size)
        #loss = F.cross_entropy(logits, Yb)
        # Explicit loss
        logit_maxes = logits.max(1, keepdim=True).values
        norm_logits = logits - logit_maxes # sub max for stability
        counts = norm_logits.exp()
        counts_sum = counts.sum(1, keepdim=True)
        counts_sum_inv = counts_sum ** -1
        probs = counts * counts_sum_inv
        logprobs = probs.log()
        loss = -logprobs[range(n), Yb].mean() # Loss at correct result letter, for each example in the batch

        # PyTorch backward pass
        for p in parameters:
            p.grad = None   
        for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, 
                  norm_logits, logit_maxes, logits, h, hpreact, bnraw, 
                  bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani, embcat, emb]:
            t.retain_grad()
        loss.backward()
        loss

        # Exercise 1: backprop through the whole thing manually
        #-----------------------------------------------------------
        dloss = 1
        dlogprobs = torch.zeros_like(logprobs)
        dlogprobs[range(n), Yb] = -1 / n
        cmp('logprobs', dlogprobs, logprobs)
        dprobs = dlogprobs / probs
        cmp('probs', dprobs, probs)
        dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
        cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
        dcounts = counts_sum_inv * dprobs 
        dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
        cmp('counts_sum', dcounts_sum, counts_sum)
        dcounts += torch.ones_like(counts) * dcounts_sum 
        cmp('counts', dcounts, counts)
        dnorm_logits = dcounts * counts
        cmp('norm_logits', dnorm_logits, norm_logits)
        dlogits = dnorm_logits.clone()
        dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
        cmp('logit_maxes', dlogit_maxes, logit_maxes)
        dlogits += F.one_hot(logits.max(1).indices, num_classes = logits.shape[1]) * dlogit_maxes
        cmp('logits', dlogits, logits)
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        cmp('h', dh, h)
        cmp('W2', dW2, W2)
        cmp('b2', db2, b2)
        dhpreact = dh * (1 - h ** 2)
        cmp('hpreact', dhpreact, hpreact)
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        cmp('bngain', dbngain, bngain)
        dbnraw = bngain * dhpreact
        cmp('bnraw', dbnraw, bnraw)
        dbnbias = dhpreact.sum(0, keepdim=True)
        cmp('bnbias', dbnbias, bnbias)
        dbndiff = bnvar_inv * dbnraw
        dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
        cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
        dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
        cmp('bnvar', dbnvar, bnvar)
        dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
        dbndiff += 2 * bndiff * dbndiff2
        cmp('bndiff', dbndiff, bndiff)
        dbnmeani = (-dbndiff).sum(0, keepdim=True)
        cmp('bnmeani', dbnmeani, bnmeani)
        dhprebn = dbndiff + dbnmeani / n
        cmp('hprebn', dhprebn, hprebn)
        dembcat = dhprebn @ W1.T
        cmp('embcat', dembcat, embcat)
        dW1 = embcat.T @ dhprebn
        cmp('W1', dW1, W1)
        db1 = dhprebn.sum(0)
        cmp('b1', db1, b1)
        demb = dembcat.view(emb.shape)
        cmp('emb', demb, emb)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[k, j]
                dC[ix] += demb[k, j]
        cmp

        tt= 42
        break


if __name__ == '__main__':
    main()
