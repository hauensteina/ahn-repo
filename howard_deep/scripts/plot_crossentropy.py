#!/usr/bin/env python

from pdb import set_trace as BP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import exp,log

# p is the true probability, q is the estimate
#-----------------------------------------------
def crossentropy(p,q):
 return -1 * (p * log(q) + (1.0-p) * log(1.0-q))

#-----------------------------------------------------
def contourplot(fig,ax,title,Z,xmin,xmax,ymin,ymax):
    ax.set_xlabel('p')
    ax.set_ylabel('q')
    im = ax.imshow(Z,
                   cmap=plt.cm.Greys,
                   origin='lower',
                   extent=(xmin,xmax,ymin,ymax)
    )
    # adding the Contour lines with labels
    cset = ax.contour(Z,
                      np.arange(0.0, 2.0 ,0.1),
                      linewidths=2,
                      cmap=plt.cm.Set2,
                      extent=(xmin,xmax,ymin,ymax)
    )
    ax.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    box = ax.get_position()
    cax = fig.add_axes([box.x0, box.y0+0.10, box.width, box.height/20])

    #im = ax.imshow(data, cmap='gist_earth')
    fig.colorbar(im, cax=cax, orientation='horizontal')
    #plt.colorbar(im) # adding the colobar on the right
    ax.set_title(title)

#--------------------------------------------------------
def lineplot(fig,ax,title,func,pmin,pmax,qmin,qmax,dp,dq):
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
    ax.set_xlabel('p')
    #ax.set_ylabel('z')
    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#-----------------------------------
def wireplot(fig,ax,title,P,Q,Z):
    ax.set_title(title)
    ax.set_xlabel('p')
    ax.set_ylabel('q')
    #ax.set_zlabel('z')
    p = ax.plot_wireframe(P, Q, Z, rstride=10, cstride=10)

#------------
def main():
    NCOLS=3
    fig = plt.figure(figsize=(15,10))
    fig.canvas.set_window_title('Cross-Entropy')
    expression = r'$z=-(p\/\log(q) + (1-p)\/\log(1-q))$'
    fig.suptitle(expression)
    contourax = fig.add_subplot(1,NCOLS,1)
    contourax.set_aspect(1)
    lineax = fig.add_subplot(1,NCOLS,2)
    lineax.set_aspect(1)
    wireax = fig.add_subplot(1,NCOLS,3,projection='3d')
    wireax.set_aspect(1)

    pmin,pmax,qmin,qmax = 0.0, 1.0, 0.0, 1.0
    delta = 0.01
    p = np.arange(pmin+delta, pmax, delta)
    q = np.arange(qmin+delta, qmax, delta)
    #BP()
    P,Q = np.meshgrid(p, q) # grid of points
    Z = crossentropy(P, Q)  # evaluation of the function on the grid
    # latex fashion title
    contourplot(fig, contourax, '', Z, pmin, pmax, qmin, qmax)
    lineplot(fig, lineax, '', crossentropy, pmin+delta, pmax, 0.1, 1.0, delta, 0.1)
    wireplot(fig, wireax, '', P,Q,Z)
    plt.show()

if __name__ == '__main__':
    main()
