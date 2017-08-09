#!/usr/bin/env python

from pdb import set_trace as BP
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp,log

#-----------------------
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

#---------------------------------------------------
def lineplot(title,func,pmin,pmax,qmin,qmax,dp,dq):
    p = np.arange(pmin,pmax,dp)
    q = np.arange(qmin,qmax,dq)

    for qval in q:
        plt.plot(p, func(p, qval),label = 'q=%.2f' % qval)
    plt.title(title)
    plt.legend()
    plt.show()

#------------
def main():
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
    lineplot(title, crossentropy, pmin+delta, pmax, 0.1, 1.0, delta, 0.1)

if __name__ == '__main__':
    main()
