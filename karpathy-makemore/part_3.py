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
HIDDEN_SZ = 200

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
    b1 = torch.randn((HIDDEN_SZ), generator=g) * 0.01
    W2 = torch.randn((HIDDEN_SZ, VSZ), generator=g) * 0.01
    b2 = torch.randn((VSZ), generator=g) * 0

    bngain = torch.ones((1,HIDDEN_SZ))
    bnbias = torch.zeros((1,HIDDEN_SZ))

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    nparms = sum([p.numel() for p in parameters])
    for p in parameters:
        p.requires_grad = True

    # Training
    BATCH_SIZE = 32
    MAX_STEPS = 200000
    lossi = []

    for i in range(MAX_STEPS):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        # forward pass
        emb = C[Xb] # (train_set_size, context_size, embedding_size)
        embcat = emb.view(emb.shape[0], -1) # (train_set_size, context_size * embedding_size)   
        hpreact = embcat @ W1 + b1 # (train_set_size, hidden_size)
        hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
        h = torch.tanh(hpreact) # (train_set_size, hidden_size)
        logits = h @ W2 + b2 # (train_set_size, vocab_size)
        loss = F.cross_entropy(logits, Yb)

        # backward pass
        for p in parameters:
            p.grad = None   
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.001
        for p in parameters:
            p.data += -lr * p.grad

        # Track stats
        if i % 10000 == 0:
            print(f'{i:7d}/{MAX_STEPS:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

        '''
        # Histogram of the hidden layer outputs
        plt.hist(h.view(-1).tolist(), bins=50)
        plt.show()

        plt.figure(figsize=(10,5))
        plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
        '''

    plt.plot(lossi)
    plt.show()

    @torch.no_grad()
    def bn_mean_std():
        """ Compute mean and standard deviation of hidden logits for X """
        emb = C[Xtr]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1 + b1
        bnmean = hpreact.mean(0, keepdim=True)
        bnstd = hpreact.std(0, keepdim=True)
        return bnmean, bnstd

    @torch.no_grad()
    def split_loss(split):
        """ Compute loss for any split in (train, val, test) """
        x,y = {'train':(Xtr,Ytr), 'val':(Xdev,Ydev), 'test':(Xte,Yte)}[split]
        bnmean, bnstd = bn_mean_std()
        emb = C[x] # (train_set_size, context_size, embedding_size)
        embcat = emb.view(emb.shape[0], -1) # (train_set_size, context_size * embedding_size)
        hpreact = embcat @ W1 + b1 # (train_set_size, hidden_size)
        hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
        h = torch.tanh(hpreact) # (train_set_size, hidden_size)
        logits = h @ W2 + b2 # (train_set_size, vocab_size)
        loss = F.cross_entropy(logits, y)
        print(split, loss.item())

    split_loss('train')    
    split_loss('val') 

    def generate():
        out = []
        context = [0] * CONTEXT_SZ
        while True:
            bnmean, bnstd = bn_mean_std()
            emb = C[tensor(context)]
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1 + b1
            hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0: break
        word = ''.join( [ ITOS[i] for i in out ] )
        return word
    
    for _ in range(20):
        word = generate()
        print(word)


    tt=42


if __name__ == '__main__':
    main()
