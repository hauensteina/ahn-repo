#!/usr/bin/env python

# /********************************************************************
# Filename: generate_images.py
# Author: AHN
# Creation Date: Aug 26, 2017
# **********************************************************************/
#
# Generate training and validation data for project blobcount
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate training and validation data for project blobcount
    Synopsis:
      %s --res <n> --minblobs <n> --maxblobs <n> --ntrain <n> --nval <n>
    Description:
      Generates jpegs in subfolders train and val, plus labels in json files.
      Each image has between minblobs and maxblobs black circles in it.
    Example:
      %s --res 128 --minblobs 0 --maxblobs 2 --ntrain 10000 --nval 100
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

    parser=argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--res",      required=True, type=int)
    parser.add_argument( "--minblobs", required=True, type=int)
    parser.add_argument( "--maxblobs", required=True, type=int)
    parser.add_argument( "--ntrain",   required=True, type=int)
    parser.add_argument( "--nval",     required=True, type=int)
    args=parser.parse_args()
    gen_image(args.res,1,'train','tt')

# Generate one image of resolution res with n circles in it.
# Image goes to folder/fname.jpg, labels go to folder/fname.json.
#----------------------------------------
def gen_image(res,nblobs,folder,fname):
    dpi=100.0
    #fig = plt.figure(figsize=(res/1000.0, res/1000.0), dpi=1000,frameon=False)
    fig = plt.figure(figsize=(res/dpi,res/dpi),dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    circle = plt.Circle((0, 0), 1.0, color='r')
    ax.add_artist(circle)
    # A line
    # ax.plot([0, 1], [0.5, 0.5], color='k', linestyle='-', linewidth=5)
    plt.savefig('myfig.jpg')

if __name__ == '__main__':
    main()
