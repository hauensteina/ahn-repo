
import argparse
import os
import re
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerModel
from tokenizer import Tokenizer


def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Train a transformer to rewrite character sequences

    Synopsis:
      {name} [--block_sz <int>] [--embed_sz <int>] [--batch_sz <int>] [--num_layers <int>] [--num_heads <int>] [--dropout <float>] 
      [--device <cpu|cuda>] [--learning_rate <float>] [--eval_interval <int>] [--num_epochs <int>] [--checkpoint_base <str>] infile

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

        The model will be loaded from a numbered checkpoint file if it exists.  If you give 'cp' as a checkpoint_base,
        files cp_0001.pt, cp_0002.pt, etc. will be used to load and save the model.

        Training data are taken from <infile>_train.txt, validation data from <infile>_val.txt.  

    Example:
      python {name} --block_sz 32 --embed_sz 16 --batch_sz 64 --num_layers 1 --num_heads 2 --num_epochs 1000 --checkpoint_base cp samples_da

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('infile', type=str)
    parser.add_argument('--block_sz', type=int, default=32)
    parser.add_argument('--embed_sz', type=int, default=16)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--checkpoint_base', type=str)
    args = parser.parse_args()
    args = args.__dict__
    if 'SM_CHANNEL_TRAIN' in os.environ:
        args['infile'] = os.path.join(os.environ['SM_CHANNEL_TRAIN'], args['infile'])
    model = run(**args)

def run(block_sz, embed_sz, batch_sz, num_layers, num_heads, dropout,
        device, learning_rate, eval_interval, num_epochs, infile, checkpoint_base):
    
    # Read all data into memory    
    train_data, val_data = read_data(infile)

    # Build or load model
    if not checkpoint_base or not newest_checkpoint(checkpoint_base): # new model
        print(f'>>>> Fresh model')
        tok = Tokenizer(train_data)
        model = TransformerModel(tok, embed_sz, num_layers,
                                num_heads, block_sz, dropout)
        model.add_optimizer(learning_rate)
    else: # load from file
        tok = Tokenizer([])
        checkpoint_file = newest_checkpoint(checkpoint_base)
        print(f'>>>> Loading model from {checkpoint_file}')
        model = TransformerModel.load(tok, checkpoint_file)
    m = model.to(device)

    train_data = [ tok.encode(x) for x in train_data ]
    val_data = [ tok.encode(x) for x in val_data]
    print(train_data[:3])

    # Check whether things are working
    xb, yb = get_batch(tok, train_data, batch_sz, block_sz)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)
    print(generate(model, tok, '{A,'))

    # Train 
    for iter in range(num_epochs):
        xb, yb = get_batch(tok, train_data, batch_sz, block_sz)
        if iter % eval_interval == 0:
            if iter: testcases(model, tok, val_data)
            losses = estimate_loss(m, tok, train_data, val_data, batch_sz, block_sz)
            print( f"step {iter}: train loss {losses[0]:.4f}, val loss {losses[1]:.4f}")
            print(tok.decode(xb[0].tolist()))
            print(tok.decode(yb[0].tolist()))
        logits, loss = m(xb, yb)
        m.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        m.optimizer.step()

    if checkpoint_base:
        fname = next_checkpoint(checkpoint_base)
        print(f'Saving model to {fname}')
        m.save(fname)

    # Test on selected prompts
    print(generate(model, tok, '{AB,'))
    print(generate(model, tok, '{ABAB,'))

def read_data(fname):
    """
    Read training and validation data, from two different files.
    Lines are comma separated input output pairs.  Lines can be commented with #.
    Strip and enclose each line in curly braces.
    """
    trainfname = fname + '_train.txt'
    valfname = fname + '_val.txt'
    sets = [[],[]]
    for idx,fname in enumerate([trainfname, valfname]):  
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [
            '{' + l + '}' for l in lines if len(l) > 0 and not l.startswith('#')]
        print('Number of samples in {fname}: ', len(lines))
        sets[idx] = lines
    return sets

# def split_train_val(samps):
#     train_frac = 0.9
#     n = int(train_frac * len(samps))
#     train_data = samps[:n]
#     val_data = samps[n:]
#     return train_data, val_data

def get_batch(tok, samps, batch_sz, block_sz):
    """
    The output is the input shifted by one character.
    Output chars to the left of comma are replaced with 0.
    The rightmost output token looks one step into the future.
    """
    def get_y(x):
        """ 
        Replace anything between { up to and including , with 0 
        {AKA,AAKAA} -> {0000AAKAA}
        Then drop first char
        {0000AAKAA} -> 0000AAKAA}
        """
        ystr0 = tok.decode(x)
        def makezero(m): return '{' + '0' * (len(m.group(0)) - 1)
        ystr1 = re.sub(r'{[^,]*', makezero, ystr0)
        ystr2 = ystr1.replace(',', '0')[1:]
        return tok.encode(ystr2)

    batchx = []  # A list of lists
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
    return batchx, batchy  # (B,T)

@torch.no_grad()
def estimate_loss(m, tok, train_data, val_data, batch_sz, block_sz):
    """ Runs a few batches through the model and returns the average train and val loss"""
    n_batches = 100
    losses = [0.0, 0.0]
    m.eval()
    for split, samps in enumerate([train_data, val_data]):
        losses = torch.zeros(n_batches)
        for k in range(n_batches):
            x, y = get_batch(tok, samps, batch_sz, block_sz)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        losses[split] = losses.mean()
    m.train()
    return losses

def testcases(m, tok, val_data):
    print('>>>> Test cases:')
    sampsize = 10
    for s in val_data[:sampsize]:
        prompt = tok.decode(s).split(',')[0] + ','        
        res = m.generate(prompt, stoptoken=tok.encode('}'), max_new_tokens=20)
        parts = res.split(',')
        prefix = ''
        if parts[0][1:] != parts[1][:-1]:
            prefix = 'ERROR: '
        print(f'{prefix}Prompt -> Res: {prompt} -> {res}')

def generate(model, tok, prompt):
    out = model.generate(prompt,
            stoptoken=tok.encode('}'),
            max_new_tokens=20)
    return out

def newest_checkpoint(checkpoint_base):
    """
    Given a checkpoint file base name, find the most recent checkpoint file.
    """
    files = os.listdir('.')
    files = [f for f in files if f.startswith(checkpoint_base + '_')]
    files = [f for f in files if f.endswith('.pt')]
    if not files: return []
    files.sort()
    return files[-1]

def next_checkpoint(checkpoint_base):
    """
    Given a checkpoint file base name, find the latest checkpoint and increment by 1.
    """
    latest = newest_checkpoint(checkpoint_base)
    if not latest: return checkpoint_base + '_0000.pt'
    latest = os.path.splitext(latest)[0]
    num = int(latest.split('_')[-1]) + 1
    out = latest.split('_')[:-1] + [f'''{num:04d}''']
    out = '_'.join(out) + '.pt'
    return out

main()
