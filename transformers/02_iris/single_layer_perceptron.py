import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import tensor

"""
A single layer perceptron classifier for simple tests.
"""

class SingleLayerPerceptron(nn.Module):
    """ A simple linear layer followed by a nonlinearity """
    def __init__(self, n_in, n_hidden, n_classes):
        super().__init__()
        # With hidden layer and nonlinearity
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(), # needs z-normalized data
            #nn.Tanh(),
            nn.Linear( n_hidden, n_classes, bias=True)
        )
        # Without hidden layer.  This is a linear classifier.
        #self.net = nn.Sequential(
        #    nn.Linear( n_in, n_classes, bias=True)
        #)

    def forward(self, x, y=None):
        """ nn.Module.__call__() calls forward(). """
        B,C = x.shape
        logits = self.net(x) # (B,n_classes)
        if y is None:
            loss = None
            return logits, loss
        else:
            sm = nn.Softmax(dim=1)(logits)
            loss = F.cross_entropy(logits, y)
            return logits, loss
        
    def add_optimizer(self, lr):
        self.learning_rate = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        

