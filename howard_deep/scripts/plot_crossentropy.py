#!/usr/bin/env python

from pdb import set_trace as BP
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import exp,log

# p is the true probability, q is the estimate
#-----------------------------------------------
def crossentropy(p,q):
 return -1 * (p * log(q) + (1.0-p) * log(1.0-q))

#-----------------------------------------------
def contourplot(title,Z,xmin,xmax,ymin,ymax):
    fig, ax = plt.subplots(1,1)
    # draw
    im = plt.imshow(Z,
                    cmap=plt.cm.Greys,
                    origin='lower',
                    extent=(xmin,xmax,ymin,ymax)
    )
    # adding the Contour lines with labels
    cset = plt.contour(Z,
                       np.arange(0.0, 2.0 ,0.1),
                       linewidths=2,
                       cmap=plt.cm.Set2,
                       extent=(xmin,xmax,ymin,ymax)
    )
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(im) # adding the colobar on the right
    plt.title(title)
    plt.show()

#--------------------------------------------------------
def lineplot(ax, title,func,pmin,pmax,qmin,qmax,dp,dq):
    p = np.arange(pmin,pmax,dp)
    q = np.arange(qmin,qmax,dq)

    for qval in q:
        ax.plot(p, func(p, qval),label = 'q=%.2f' % qval)
    ax.set_title(title)
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', prop = fontP) # bbox_to_anchor=(1.5, 1.05))
    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#------------
def main():
    fig, axes = plt.subplots(nrows=1, ncols=2)
    pmin,pmax,qmin,qmax = 0.0, 1.0, 0.0, 1.0
    delta = 0.01
    p = np.arange(pmin+delta, pmax, delta)
    q = np.arange(qmin+delta, qmax, delta)
    #BP()
    P,Q = np.meshgrid(p, q) # grid of points
    Z = crossentropy(P, Q)  # evaluation of the function on the grid
    # latex fashion title
    title = r'$z=-(p\/\log(q) + (1-p)\/\log(1-q))$'
    #contourplot(title,Z,pmin,pmax,qmin,qmax)
    lineplot(axes[0], title, crossentropy, pmin+delta, pmax, 0.1, 1.0, delta, 0.1)
    plt.show()
if __name__ == '__main__':
    main()
