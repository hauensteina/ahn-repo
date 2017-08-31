#!/usr/bin/env python

# /********************************************************************
# Filename: generate_images.py
# Author: AHN
# Creation Date: Aug 30, 2017
# **********************************************************************/
#
# Generate training and validation data for project blobcount
#

from __future__ import division,print_function
from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
#from numpy.random import random
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate training and validation data for project blobcount
    Synopsis:
      %s --res <n> --gridsize <n> --ntrain <n> --nval <n>
    Description:
      Generates jpegs in subfolders train and val, plus labels in json files.
      Each image has between 1 and gridsize*gridsize black circles in it.
      The circles are aligned on an gridsize*gridsize grid.
      Res must be a multiple of 2 * gridsize.
    Example:
      %s --res 120 --gridsize 5 --ntrain 1000 --nval 100
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--res",      required=True, type=int)
    parser.add_argument( "--gridsize", required=True, type=int)
    parser.add_argument( "--ntrain",   required=True, type=int)
    parser.add_argument( "--nval",     required=True, type=int)
    args = parser.parse_args()
    if (args.res % (2*args.gridsize)): usage(True)
    #np.random.seed(0) # Make things reproducible
    trainfolder = 'train/all_files'
    valfolder   = 'valid/all_files'
    if not os.path.exists(trainfolder): os.makedirs(trainfolder)
    if not os.path.exists(valfolder):   os.makedirs(valfolder)
    gen_images(args.ntrain, args.res, args.gridsize, trainfolder)
    gen_images(args.nval,   args.res, args.gridsize, valfolder)

# Generate one image of resolution resxres with a random number
# between 1 and gridsize*gridsize circles in it.
# The circles are aligned with the grid.
#------------------------------------------
def gen_image(res,gridsize,nblobs,ofname):
    # Set up matplotlib
    dpi=100.0
    fig = plt.figure(figsize=(res/dpi,res/dpi),dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Where should the circles be
    pos = range(gridsize*gridsize)
    np.random.shuffle(pos)
    linpos = pos[:nblobs]
    pairpos = [(x // gridsize, x % gridsize) for x in linpos]
    r = res // (2*gridsize)
    pairpos = [(p[1]*2*r + r, p[0]*2*r + r) for p in pairpos]

    for i,p in enumerate(pairpos):
        circle = plt.Circle((p[0]/res, p[1]/res), 0.8*r/res, color='k')
        ax.add_artist(circle)
    plt.savefig(ofname)

# Generate nb_imgs images with minblobs to maxblobs circles.
# Also generate a json file for each, giving the number of circles.
#-------------------------------------------------------------------
def gen_images(nb_imgs,resolution,gridsize,folder):
    #BP()
    nblobs = 1 + np.random.randint(gridsize*gridsize, size=nb_imgs)
    for i in range(nb_imgs):
        fname = '%07d' % i
        fjpg  = folder + '/' + fname + '.jpg'
        fjson = folder + '/' + fname + '.json'
        # Make jpeg
        gen_image(resolution, gridsize, nblobs[i], fjpg)
        gen_image(resolution, gridsize, nblobs[i], fjpg)
        plt.close('all')
        # Dump metadata
        meta = { 'class':nblobs[i] }
        meta_json = json.dumps(meta) + '\n'
        with open(fjson,'w') as f: f.write(meta_json)



if __name__ == '__main__':
    main()
