
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch import tensor

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=G) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ( [] if self.bias is None else [self.bias] )
    
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
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
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
    
class Tanh:
    def __call__(self,x):
        self.out = x.tanh()
        return self.out
    def parameters(self):
        return []   
    
EMBED_SZ = 10
HIDDEN_SZ = 100
CONTEXT_SZ = 3
WORDS = open('names.txt','r').read().splitlines()
_chars = sorted(list(set(''.join(WORDS))))
STOI = {s:i+1 for i,s in enumerate(_chars)}
STOI['.'] = 0
ITOS = { i:s for s,i in STOI.items()}
VOCAB_SZ = len(STOI)

G = torch.Generator().manual_seed(42)

C = torch.randn((VOCAB_SZ, EMBED_SZ), generator=G) # embedding matrix
layers = [
    Linear(EMBED_SZ * CONTEXT_SZ, HIDDEN_SZ), BatchNorm1d(HIDDEN_SZ),  Tanh(),
    Linear(HIDDEN_SZ, HIDDEN_SZ), BatchNorm1d(HIDDEN_SZ), Tanh(),
    Linear(HIDDEN_SZ, HIDDEN_SZ), BatchNorm1d(HIDDEN_SZ), Tanh(),
    Linear(HIDDEN_SZ, HIDDEN_SZ), BatchNorm1d(HIDDEN_SZ), Tanh(),
    Linear(HIDDEN_SZ, HIDDEN_SZ), BatchNorm1d(HIDDEN_SZ), Tanh(),
    Linear(HIDDEN_SZ, VOCAB_SZ), BatchNorm1d(VOCAB_SZ)
]

with torch.no_grad():
    # Prevent overconfidence in the softmax layer
    #layers[-1].weight *= 0.1
    layers[-1].gamma *= 0.1
    # All other layers get the magic 5/3 gain for tanh nonlinearity
    for l in layers[:-1]:
        if isinstance(l, Linear):
            l.weight *= 5/3 

parameters = [C] + [p for l in layers for p in l.parameters()]
print('Number of parameters:', sum(p.numel() for p in parameters))
for p in parameters:
    p.requires_grad = True  

# Training

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

import random
random.seed(42)
random.shuffle(WORDS)
n1 = int(len(WORDS) * 0.8)
n2 = int(len(WORDS) * 0.9)
Xtr, Ytr = build_dataset(WORDS[:n1], CONTEXT_SZ) # 80% of the data
Xdev, Ydev = build_dataset(WORDS[n1:n2], CONTEXT_SZ) # 10% of the data
Xte, Yte = build_dataset(WORDS[n2:], CONTEXT_SZ) # 10% of the data

BATCH_SIZE = 32
MAX_STEPS = 200000
lossi = []
ud = [] 

for i in range(MAX_STEPS):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=G)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb] # (train_set_size, context_size, embedding_size)
    x = emb.view(emb.shape[0], -1) # (train_set_size, context_size * embedding_size)   
    for l in layers:
        x = l(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    for layer in layers:
        layer.out.retain_grad() # keep gradients for debugging
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f'Step {i}: loss = {loss.item():.3f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])  

    if i > 1000:
        break   

# visualize activations by layer
plt.figure(figsize=(10,5))
legends = []
for i,layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % 
              (i, layer.__class__.__name__, t.mean(), t.std(), 100 * ((t.abs() > 0.97).float().mean())))
        hy, hx = torch.histogram(t, density=True)
        # Detach the tensors to avoid keeping the computation graph in memory
        hy = hy.detach()
        hx = hx.detach()
        # there is one more bin boundary than there are bins
        plt.plot(hx[:-1], hy) 
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')

# visualize layer output gradients
plt.figure(figsize=(10,5))
legends = []
for i,layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % 
              (i, layer.__class__.__name__, t.mean(), t.std(), 100 * ((t.abs() > 0.97).float().mean())))
        hy, hx = torch.histogram(t, density=True)
        # Detach the tensors to avoid keeping the computation graph in memory
        hy = hy.detach(); hx = hx.detach()
        # there is one more bin boundary than there are bins
        plt.plot(hx[:-1], hy) 
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('layer output gradient distribution')

# visualize layer weight gradients
plt.figure(figsize=(10,5))
legends = []
for i,p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2: # Skip the biases
        # A large data ratio tells you that your gradients are large relative to the data, so smaller learning rate needed.
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % 
              (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hy, hx = torch.histogram(t, density=True)
        # Detach the tensors to avoid keeping the computation graph in memory
        hy = hy.detach(); hx = hx.detach()
        plt.plot(hx[:-1], hy)
        legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('layer weight gradient distribution')

# For each layer, for ean iteration, plot the log ratio of the gradient standard deviation to the parameter standard deviation.
plt.figure(figsize=(10,5))
legends = []
for i,p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2: # Skip the biases
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append('param %d' % i)
plt.plot([0,len(ud)],[-3,-3],'k')  # Mark where we want to be ( 1 / 10000 of a weight value std should be an update step )   
plt.legend(legends)
plt.title('layer weight gradient distribution')

plt.show()

