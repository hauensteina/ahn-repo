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
EMBED_SZ = 2
CONTEXT_SZ = 3
HIDDEN_SZ = 100

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
    X,Y = build_dataset(WORDS, CONTEXT_SZ)
    g = torch.Generator().manual_seed(42)

    # Define the net
    C = torch.randn((len(STOI), EMBED_SZ), generator=g) # embedding matrix
    W1 = torch.randn((EMBED_SZ * CONTEXT_SZ, HIDDEN_SZ), generator=g)
    b1 = torch.randn((HIDDEN_SZ), generator=g)
    W2 = torch.randn((HIDDEN_SZ, VSZ), generator=g)
    b2 = torch.randn((VSZ), generator=g)
    parameters = [C, W1, b1, W2, b2]
    nparms = sum([p.numel() for p in parameters])
    for p in parameters:
        p.requires_grad = True

    # for each context block, for each letter in the block, embedding as EMBED_SZ floats
    emb = C[X]  
    # emb is now a 3D tensor of shape (len(context_blocks), CONTEXT_SZ, EMBED_SZ)
    # we need to flatten it to 2D
    emb = emb.view(-1, EMBED_SZ * CONTEXT_SZ) # -1 means infer the size
    # Hidden layer
    '''
    Broadcast: Starting from the right, the dimensions must match. 
    When you run out of dimensions, use the ones from the other operand.
    The missing dimensions are filled with copies of the operand.
    So (32 x 100) + (100) -> (32 x 100) + (32 x 100) -> (32 x 100) .
    (100) + (32 x 100) -> (32 x 100) + (32 x 100) -> (32 x 100) .
    '''
    hidden_output = torch.tanh(emb @ W1 + b1)
    # Output layer
    logits = hidden_output @ W2 + b2
    # Think of logits as logs of counts
    #counts = logits.exp()
    # Normalize
    #probs = counts / counts.sum(dim=1, keepdim=True)
    # Or use softmax, which is exactly the same
    #probs = F.softmax(logits, dim=1)
    # Probs of training examples. We want to maximize this.
    # We want them close to 1.
    #ptrain = probs[torch.arange(len(X)), Y]
    # Loss. Negative mean log likelihood
    #loss = -ptrain.log().mean()

    # The neg mean log likelihood after a softmax is called cross entropy
    loss = F.cross_entropy(logits, Y)

    stepsizes_exp = torch.linspace(-3,0,1000)
    stepsizes = 10 ** stepsizes_exp


    BATCH_SIZE = 32
    while 1:
        lri = []
        lossi = []
        for i in range(40000):
            # minibatch construct
            ix = torch.randint(0, X.shape[0], (BATCH_SIZE,))

            # forward pass
            emb = C[X[ix]] # (train_set_size, context_size, embedding_size)
            hidden_output = torch.tanh(emb.view(-1, EMBED_SZ * CONTEXT_SZ) @ W1 + b1) # (train_set_size, hidden_size)
            logits = hidden_output @ W2 + b2 # (train_set_size, vocab_size)
            loss = F.cross_entropy(logits, Y[ix])
            # backward pass
            for p in parameters:
                p.grad = None   
            loss.backward()
            # update
            #lr = stepsizes[i]
            lr = 0.1
            if i > 20000: lr = 0.01
            for p in parameters:
                p.data += -lr * p.grad
            #lri.append(stepsizes_exp[i])
            #lossi.append(loss.item())

        #plt.plot(lri, lossi)
        print(f'''Batch loss:{loss.item()}''')
        emb = C[X]
        hidden_output = torch.tanh(emb.view(-1, EMBED_SZ * CONTEXT_SZ) @ W1 + b1) # (train_set_size, hidden_size)
        logits = hidden_output @ W2 + b2 # (train_set_size, vocab_size)
        loss = F.cross_entropy(logits, Y)
        print(f'''Total loss:{loss.item()}\n''')
        tt = 42




if __name__ == '__main__':
    main()
