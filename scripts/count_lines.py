#!/usr/bin/env python

# /********************************************************************
# Filename: count_lines.py
# Author: AHN
# Creation Date: Apr 6, 2018
# **********************************************************************/
#
# Count lines in the folders specified
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
from numpy.random import random
import argparse

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Count non-empty lines in all source files in the given folders
    Synopsis:
      %s --folders <folder1> <folder2> ...
    Example:
      %s --folders protxxpilot | sort -n
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
    parser.add_argument('--folders', nargs='*', required=True)
    #parser.add_argument( "--substr",      required=True)
    #parser.add_argument( "--trainpct",    required=True, type=int)
    args = parser.parse_args()

    lcount = 0
    for folder in args.folders:
        dirlist = [x for x in os.walk( folder)]
        for d in dirlist:
            files = d[2]
            for f in files:
                if not os.path.splitext( f)[1] in ('.h', '.hpp', '.c', '.cpp', '.m', '.mm', '.py', '.swift'):
                    continue
                with open( '%s/%s' % (d[0],f)) as fin:
                    lines = fin.readlines()
                lcount_in_file = 0
                for line in lines:
                    if re.match( r'^[\s]*$', line):
                        continue
                    lcount_in_file += 1
                    lcount += 1
                print( '%5d %s/%s' % (lcount_in_file, d[0], f))
    print( '%d TOTAL' % lcount)


if __name__ == '__main__':
    main()
