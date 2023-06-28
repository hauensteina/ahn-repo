
import argparse
import os,re
import torch 
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
#from transformer import BLOCK_SZ, EMBED_SZ, BATCH_SZ
from tokenizer import Tokenizer

def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Train a transformer to rewrite character sequences

    Synopsis:
      {name} [--block_sz <int>] [--embed_sz <int>] [--batch_sz <int>] [--num_layers <int>] [--num_heads <int>] [--dropout <float>] [--device <cpu|cuda>] [--learning_rate <float>] [--eval_interval <int>] [--num_epochs <int>] infile

    Description:
        Train a transformer to rewrite character sequences. 
        The input file should contain one input output pair per line.  Lines can be commented with #.
        For example, the following is a valid input file:

        # Minimal training data to get off the ground
        AB,AAB
        ABCAB,AABCAAB
        AB,AAB
        ABCAB,AABCAAB
        AB,AAB
        ABCAB,AABCAAB

    Example:
      python {name} --block_sz 32 --embed_sz 16 --batch_sz 64 --num_layers 1 --num_heads 2 --num_epochs 1000 input_0.txt

'''
    msg += '\n '
    return msg

#-------------
def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('infile', type=str)
    parser.add_argument( '--block_sz', type=int, default=32)
    parser.add_argument( '--embed_sz', type=int, default=16)
    parser.add_argument( '--batch_sz', type=int, default=64)
    parser.add_argument( '--num_layers', type=int, default=1)
    parser.add_argument( '--num_heads', type=int, default=2)
    parser.add_argument( '--dropout', type=float, default=0.2)
    parser.add_argument( '--device', type=str, default='cpu')
    parser.add_argument( '--learning_rate', type=float, default=3e-4)
    parser.add_argument( '--eval_interval', type=int, default=100)
    parser.add_argument( '--num_epochs', type=int, default=1000)
    args = parser.parse_args()
    args = args.__dict__
    run(**args)


def read_data(fname):
    with open(fname, 'r', encoding='utf-8') as f:
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

def get_batch(tok,samps,batch_sz,block_sz):
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
    while len(batchx) < batch_sz:
        batchelx = []
        while len(batchelx) < block_sz + 1:
            idx = torch.randint(0, len(samps), (1,))
            batchelx += (samps[idx])
        batchely = get_y(batchelx[:block_sz+1])
        batchelx = batchelx[:block_sz]
        batchx.append(batchelx)
        batchy.append(batchely)

    batchx = torch.tensor(batchx)
    batchy = torch.tensor(batchy)
    return batchx, batchy # (B,T)

@torch.no_grad()
def estimate_loss(m,train_data,val_data,batch_sz,block_sz):
    """ Runs a few batches through the model and returns the average train and val loss"""
    n_batches = 100
    losses = [0.0, 0.0]
    m.eval() 
    for split,samps in enumerate([train_data,val_data]):
        losses = torch.zeros(n_batches)
        for k in range(n_batches):
            x,y = get_batch( m.tok, samps,batch_sz,block_sz)
            logits, loss = m(x,y)
            losses[k] = loss.item()
        losses[split] = losses.mean()
    m.train()
    return losses

def generate(model,prompt):
    return model.generate( model.tok.encode(prompt), 
                            stoptoken=model.tok.encode('}'), 
                            max_new_tokens=20)

def run( block_sz,embed_sz,batch_sz,num_layers,num_heads,dropout,
        device,learning_rate,eval_interval,num_epochs,infile):
    samples = read_data(infile)
    tok = Tokenizer(samples)
    toksamps = [tok.encode(s) for s in samples]
    print(toksamps[:3])   
    train_data, val_data = split_train_val(toksamps)

    xb, yb = get_batch(tok, train_data, batch_sz, block_sz)

    model = TransformerModel( tok, embed_sz, num_layers, num_heads, block_sz, dropout)
    m = model.to(device)

    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print( generate(model,'{A,'))

    optimizer = torch.optim.Adam( m.parameters(), lr=learning_rate) 
    
    for iter in range(num_epochs):
        if iter % eval_interval == 0:
            losses = estimate_loss(m,train_data,val_data,batch_sz,block_sz)
            print(f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
        xb,yb = get_batch(tok,train_data,batch_sz,block_sz)
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
    print(generate(model,'{AB,'))
    print(generate(model,'{ABCAB,'))


main()