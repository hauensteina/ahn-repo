
import re
import torch 
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
#from transformer import BLOCK_SZ, EMBED_SZ, BATCH_SZ
from tokenizer import Tokenizer

BLOCK_SZ = 16
EMBED_SZ = 4
BATCH_SZ = 2
NUM_LAYERS = 1
NUM_HEADS = 1
DROPOUT = 0.2
DEVICE = 'cpu'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
N_EPOCHS = 3000

def read_data():
    with open('input_0.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = ['{' + l + '}' for l in lines if len(l) > 0 and not l.startswith('#')]
    print('Number of samples: ', len(lines))
    print(lines[:3])
    return lines

def split_train_val(samps):
    train_frac = 0.9
    n = int(train_frac * len(samps))
    train_data = samps[:n]
    val_data = samps[n:] 
    return train_data, val_data

def get_batch(tok,samps):
    def get_y(x):
        """ 
        Replace anything between { up to and including , with 0 
        {AKA,AAKAA} -> {0000AAKAA}
        Then drop first char
        {0000AAKAA} -> 0000AAKAA}
        """
        ystr0 = tok.decode(x)
        makezero = lambda m: '{' + '0' * (len(m.group(0)) - 1)
        ystr1 = re.sub( r'{[^,]*', makezero, ystr0)
        ystr2 = ystr1.replace(',','0')[1:]
        return tok.encode(ystr2)

    batchx = [] # A list of lists
    batchy = [] 
    while len(batchx) < BATCH_SZ:
        batchelx = []
        while len(batchelx) < BLOCK_SZ + 1:
            idx = torch.randint(0, len(samps), (1,))
            batchelx += (samps[idx])
        batchely = get_y(batchelx[:BLOCK_SZ+1])
        batchelx = batchelx[:BLOCK_SZ]
        batchx.append(batchelx)
        batchy.append(batchely)

    batchx = torch.tensor(batchx)
    batchy = torch.tensor(batchy)
    return batchx, batchy # (B,T)

@torch.no_grad()
def estimate_loss(m,toksamps):
    """ Runs a few batches through the model and returns the average train and val loss"""
    trainsamps, valsamps = split_train_val(toksamps)
    n_batches = 100
    losses = [0.0, 0.0]
    m.eval()
    for split,samps in enumerate([trainsamps,valsamps]):
        losses = torch.zeros(n_batches)
        for k in range(n_batches):
            x,y = get_batch( m.tok, samps)
            logits, loss = m(x,y)
            losses[k] = loss.item()
        losses[split] = losses.mean()
    m.train()
    return losses

def generate(model,prompt):
    return model.generate( model.tok.encode(prompt), 
                            stoptoken=model.tok.encode('}'), 
                            max_new_tokens=20)

def main():
    torch.manual_seed(1337)
    samples = read_data()
    tok = Tokenizer(samples)
    toksamps = [tok.encode(s) for s in samples]
    print(toksamps[:3])   
    train_data, val_data = split_train_val(toksamps)

    xb, yb = get_batch(tok, train_data)

    model = TransformerModel(tok, embed_sz=EMBED_SZ, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, 
                             block_sz=BLOCK_SZ, dropout=DROPOUT, device=DEVICE)
    m = model.to(DEVICE)

    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print( generate(model,'{A,'))

    optimizer = torch.optim.Adam( m.parameters(), lr=LEARNING_RATE) # 3e-4 for larger nets
    
    for iter in range(N_EPOCHS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(m,toksamps)
            print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
        xb,yb = get_batch(tok,train_data)
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
    print(generate(model,'{AB,'))
    print(generate(model,'{ABCAB,'))


main()