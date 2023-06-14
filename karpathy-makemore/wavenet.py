import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch import tensor

WORDS = open('names.txt','r').read().splitlines()
print(len(WORDS))
print(max([len(w) for w in WORDS]))
print(WORDS[:8])

# Chars and their int mapping
CHARS = sorted(list(set(''.join(WORDS))))
STOI = {s:i+1 for i,s in enumerate(CHARS)}
STOI['.'] = 0
ITOS = {i:s for s,i in STOI.items()}
VOCAB_SZ = len(ITOS)
print(ITOS)
print(VOCAB_SZ)
CONTEXT_SZ = 8
EMBED_SZ = 24
HIDDEN_SZ = 128
BATCH_SIZE = 32
#BATCH_SIZE = 5
#MAX_STEPS = 200000
MAX_STEPS = 10000

torch.manual_seed(42)   

def build_dataset(words, context_size):
    X,Y = [],[]
    for w in words:
        #print(w)
        context = [0] * context_size
        for ch in w + '.':
            ix = STOI[ch]
            X.append(context) # left context
            Y.append(ix) # next char
            context = context[1:] + [ix] 

    X = tensor(X)
    Y = tensor(Y)   
    print(X.shape, Y.shape)
    return X,Y     

#------------------------------------------------------------
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ( [] if self.bias is None else [self.bias] )

#------------------------------------------------------------
class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # batchnorm trained variance and mean
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # batchnorm input variance and mean (estimated)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim=0
            elif x.ndim == 3:
                dim=(0,1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

#------------------------------------------------------------
class Tanh:
    def __call__(self,x):
        self.out = x.tanh()
        return self.out
    def parameters(self):
        return []   
    
class Embedding:
    def __init__(self, vocab_size, embedding_size):
        self.weight = torch.randn((vocab_size, embedding_size)) 
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B,T,C = x.shape # Batch, Time, Channels
        x = x.view( B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1) 
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        self.out = x
        return x
    
    def parameters(self):
        return [ p for l in self.layers for p in l.parameters()]

def main():
    import random
    random.seed(42)
    random.shuffle(WORDS)
    n1 = int(len(WORDS) * 0.8)
    n2 = int(len(WORDS) * 0.9)
    Xtr, Ytr = build_dataset(WORDS[:n1], CONTEXT_SZ) # 80% of the data
    Xdev, Ydev = build_dataset(WORDS[n1:n2], CONTEXT_SZ) # 10% of the data
    Xte, Yte = build_dataset(WORDS[n2:], CONTEXT_SZ) # 10% of the data

    for x,y in zip(Xtr[:10], Ytr[:10]):
        print(''.join( [ ITOS[ix.item()] for ix in x ] ), '-->', ITOS[y.item()] )
 
    # Define the net
    #------------------------------------------------------------
    model = Sequential([
        Embedding(VOCAB_SZ, EMBED_SZ),
        FlattenConsecutive(2), Linear(EMBED_SZ * 2, HIDDEN_SZ, bias=False), BatchNorm1d(HIDDEN_SZ), Tanh(),
        FlattenConsecutive(2), Linear(HIDDEN_SZ * 2, HIDDEN_SZ, bias=False), BatchNorm1d(HIDDEN_SZ), Tanh(),
        FlattenConsecutive(2), Linear(HIDDEN_SZ * 2, HIDDEN_SZ, bias=False), BatchNorm1d(HIDDEN_SZ), Tanh(),
        Linear(HIDDEN_SZ, VOCAB_SZ), 
    ])   
    with torch.no_grad():
        model.layers[-1].weight *= 0.1 # Make last layer less confident

    parameters = model.parameters()
    print('Number of parameters:', sum(p.numel() for p in parameters))
    for p in parameters:
        p.requires_grad = True  
    
    lossi = [] # loss at iteration i
    ud = [] # update steps relative to data standard deviation
    for i in range(MAX_STEPS):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        logits = model(Xb)
        #showshapes(Xb, Yb, model)
        show_all_shapes(model)
        loss = F.cross_entropy(logits, Yb)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update: simple SGD
        lr = 0.1 if i < 150000 else 0.01 # learn rate decay
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 10000 == 0:
            print(f'Step {i}: loss = {loss.item():.3f}')
        lossi.append(loss.log10().item())

        tt=42
        with torch.no_grad():
            ud.append( [ (lr*p.grad.std() / p.data.std()).log10().item() for p in parameters ] )  

        #if i > 1000:
        #    break       

    smoothed = torch.tensor(lossi).view(-1,1000).mean(1)
    plt.plot(smoothed)
    plt.show()

    for layer in model.layers:
        layer.training = False

    @torch.no_grad()
    def split_loss(split):
        """ Compute loss for any split in (train, val, test) """
        x,y = {'train':(Xtr,Ytr), 'val':(Xdev,Ydev), 'test':(Xte,Yte)}[split]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        print(split, loss.item())       
        
    split_loss('train')
    split_loss('val')

    def generate():
            out = []
            context = [0] * CONTEXT_SZ
            while True:
                logits = model(tensor([context]))
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0: break
            word = ''.join( [ ITOS[i] for i in out ] )
            return word

    for _ in range(20):
        print(generate())
    tt = 42


def showshapes(Xb,Yb,model):
    print(f'''Xb shape {Xb.shape}''')
    print(f'''Xb {Xb}''')
    print(f'''layer 0 embed out shape {model.layers[0].out.shape}''')
    print(f'''layer 1 flatten out shape {model.layers[1].out.shape}''')
    print(f'''layer 2 linear weight shape {model.layers[2].weight.shape}''')
    print(f'''layer 2 linear out shape {model.layers[2].out.shape}''')

def show_all_shapes(model):
    for i,layer in enumerate(model.layers):
        print(f'''layer {i} {layer.__class__.__name__} out shape {layer.out.shape}''')

if __name__ == '__main__':
    main()
