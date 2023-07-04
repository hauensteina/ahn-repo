
import argparse
import os, re
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from single_layer_perceptron import SingleLayerPerceptron

CHECKPOINT_BASE = 'cp_iris'
DATA_FILE = 'iris_data/iris.data'
LEARNING_RATE = 3E-4
#LEARNING_RATE = 0.01

def usage():
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name}: Train a perceptron on the Iris dataset

    Synopsis:
      {name} --num_epochs <int>] 

    Description:
        Train a perceptron on the Iris dataset.
        Data are taken from iris_data/iris.data.
        A line looks like
            5.1,3.5,1.4,0.2,Iris-setosa

    Example:
      python {name} --num_epochs 1000

    '''
    msg += '\n '
    return msg

# -------------

def main():
    torch.manual_seed(1337)
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('--num_epochs', type=int, default=1000)
    args = parser.parse_args()
    args = args.__dict__
    model = run(**args)

def run(num_epochs):
    # Read all data into memory    
    train_datax, train_datay, val_datax, val_datay, c2i, i2c = read_data(DATA_FILE)
    #train_datax = train_datax[:2]
    #train_datay = train_datay[:2]
    
    model = SingleLayerPerceptron( n_in=len(train_datax[0]), n_hidden=10, n_classes=len(c2i))
    model.add_optimizer(LEARNING_RATE)
    logits, loss = model.forward(train_datax, train_datay)

    # Train 
    for iter in range(num_epochs):
        train_logits, train_loss = model.forward(train_datax, train_datay)
        val_logits, val_loss = model.forward(val_datax, val_datay)
        train_acc = accuracy(model, train_datax, train_datay)
        val_acc = accuracy(model, val_datax, val_datay)
        print( f"epoch {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train acc {train_acc:.4f}, val acc {val_acc:.4f}" )
        logits, loss = model(train_datax, train_datay)
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()

@torch.no_grad()
def accuracy( model, datax, datay):
    logits, loss = model.forward(datax, datay)
    pred = torch.argmax(logits, dim=1)
    acc = torch.mean( (pred == datay).float() )
    return acc

def read_data(fname):
    """
    Read data and split into train and validation sets.
    A line looks like
            5.1,3.5,1.4,0.2,Iris-setosa
    Subtract mean and divide by standard deviation across all samples.   
    """
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [ l for l in lines if len(l) > 0 and not l.startswith('#') ]
    print(f'Number of samples in {fname}: ', len(lines))

    classes = sorted( list( set( [ line.split(',')[-1] for line in lines ] )))
    c2i = { c:i for i,c in enumerate(classes) }
    i2c = { i:c for i,c in enumerate(classes) }

    #  Get data into tensors 
    datax = [ line.split(',')[:-1] for line in lines ]
    datax = tensor([ [ float(x) for x in line ] for line in datax ], dtype=torch.float)
    datay = tensor([ c2i[line.split(',')[-1]] for line in lines ], dtype=torch.long)

    # Normalize
    mmean = datax.mean(dim=0)
    sstd = datax.std(dim=0)
    datax = (datax - mmean) / sstd

    # Shuffle
    perm = torch.randperm(len(datax))
    datax = datax[perm]
    datay = datay[perm]

    # Split into train and validation sets
    train_frac = 0.9
    n = int(train_frac * len(datax))
    train_datax = datax[:n]
    train_datay = datay[:n]
    val_datax = datax[n:]
    val_datay = datay[n:]

    return train_datax, train_datay, val_datax, val_datay, c2i, i2c

main()
