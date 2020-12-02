#!/usr/bin/env python

import sys,os
import argparse
import pstats
from pstats import SortKey

#-----------------------------
def usage( printmsg=False):
    name = os.path.basename( __file__)
    msg = '''

    Description:
      %s: Print cProfile output in a meaningful way
    Synopsis:
      %s --infile <fname> --topn <n>
    Example:
      $ python -m cProfile -o cprofile.out algo_x_2d.py --case pentomino
      $ %s --infile cprofile.out --topn 10

--
''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------------
def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--infile', required=True)
    parser.add_argument( '--topn', type=int, required=True)
    args = parser.parse_args()

    p = pstats.Stats( args.infile)
    p.sort_stats( SortKey.CUMULATIVE).print_stats(args.topn)

main()
