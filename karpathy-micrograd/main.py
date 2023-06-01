
from pdb import set_trace as BP
import visualize as vis
from engine import Value, build_topo
from matplotlib import pyplot as plt
import numpy as np


# plt.plot( np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)) )
# plt.grid()
# plt.show()

# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias
b = Value(6.881373, label='b')
 
x1w1 = x1 * w1; x1w1.label = 'x1w1'
x2w2 = x2 * w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'

#o = n.tanh(); o.label = 'o'

e = (2*n).exp(); e.label = 'e'
o = (e - 1) / (e + 1); o.label = 'o'

o.backward()
vis.draw_dot(o)
