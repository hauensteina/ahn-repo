#!/usr/bin/env python

# Script to plot any two columns from a csv as a line, as svg file.

# AHN, Apr 2019

import os,sys,re,glob,shutil,json, math
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dateutil.parser import parse

import argparse

from pdb import set_trace as BP

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = '''
    Description:
      %s:  Plot any two columns from a csv as a line

    Synopsis:
      %s --file <str>.csv --xcol <str> --ycol <str> [--title <str>] [--xlabel <str>] [--ylabel <str>]

    Example:
      %s --file tt.csv --xcol n --ycol x --title Demo --xlabel x --ylabel y

    Output goes to plot_csv.svg .

''' % (name,name,name)

    msg += '\n '
    return msg

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--file', required=True)
    parser.add_argument( '--xcol', required=True)
    parser.add_argument( '--ycol', required=True)
    parser.add_argument( '--title', default='')
    parser.add_argument( '--xlabel', default='')
    parser.add_argument( '--ylabel', default='')
    args = parser.parse_args()

    csvstr = open( args.file).read()
    lines = csv2dict( csvstr)[0]
    xvals = [line[args.xcol] for line in lines]
    yvals = [line[args.ycol] for line in lines]

    mpl.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.plot( xvals, yvals)
    ax.set_xlabel( args.xlabel)
    ax.set_ylabel( args.ylabel)
    ax.set_title( args.title)

    fig.tight_layout()
    plt.savefig('plot_csv.svg')

# Transform csv format to a list of dicts
#-------------------------------------------
def csv2dict( csvstr):
    lines = csvstr.split('\n')
    colnames = []
    res = []
    for idx, line in enumerate( lines):
        line = line.strip()
        if len(line) == 0: continue
        if line[0] == '#': continue
        words = line.split(',')
        words = [w.strip() for w in words]
        if not colnames:
            colnames = words
            continue
        ddict = { col:number(words[idx]) for idx,col in enumerate(colnames) }
        res.append(ddict)
    return res, colnames

# Convert a string to a float, if it is a number
#--------------------------------------------------
def number( tstr):
    try:
        res = float( tstr)
        return res
    except ValueError:
        return tstr

if __name__ == '__main__':
    main()
