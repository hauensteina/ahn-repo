import math
import numpy as np

class Value:
    """ A single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # Internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"{self.label}: {self.data:.4f} {self.grad:.4f}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # Parent
        out = Value(self.data + other.data, (self, other), '+')

        # Calling _backward on the parent updates the child gradients.
        # Addition  
        # Change with respect to either child is the same as change with respect to parent
        # p = a + b => de / da = de / dp 
        # and          de / db = de / dp 
        def _backward():
            self.grad += out.grad  # += because we might feed into more than one parent
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): # other + self
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # Multiplication  
        # Change with respect to one child is multiplied by the value of the other
        # p = a * b => de / da = de / dp *  b
        # and          de / db = de / dp *  a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)  

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        x = self.data**other
        out = Value(x, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out 

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out 

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out 

    def backward(self):
        """ 
        Backpropagate gradients through the graph.
        Only called on the loss node.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
 
def build_topo(v):
    """
    Topologically sort the expression graph below v. Leaves come first.
    The backward pass will need to traverse this list in reverse.
    """
    topo = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(v)
    return topo    