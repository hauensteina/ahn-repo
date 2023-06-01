import random
from engine import Value
from visualize import draw_dot

class Neuron:
    def __init__(self, nin):
        self.w = [ Value(random.uniform(-1, 1)) for _ in range(nin) ]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """ w*x + b"""
        act = sum( (wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [ Neuron(nin) for _ in range(nout) ]

    def __call__(self, x):
        outs = [ n(x) for n in self.neurons ]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [ Layer(sz[i], sz[i+1]) for i in range(len(nouts)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == '__main__':
    # Training examples
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    # Desired outputs
    ys = [1.0,-1.0,-1.0,1.0]

    n = MLP(3,[4,4,1])

    NITER = 1000
    EPS = 0.2
    for i in range(NITER):
        # Each input gets its own expression tree.
        # But all expression trees share pointers to the same parameters in memory.
        # The inputs are constant parts of the function we optimze.
        # The parameters are the variables.     
        ypred = [ n(x) for x in xs ]
        loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))
        print(loss)
        for p in n.parameters():
            p.grad = 0.0
        # Compute the partial derivatives of the loss w.r.t. the parameters while inputs are held constant.
        # Each parameter accumulates the gradient from several inputs via addition.
        # Imagine len(xs) arrows from a single parameter to each input expression tree.     
        loss.backward()
        for p in n.parameters():
            p.data -= EPS * p.grad

