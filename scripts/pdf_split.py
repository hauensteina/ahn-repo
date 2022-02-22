#!/usr/bin/env python

# Script to split pdf into pages

# AHN, FEb 2022

import os,sys,re,glob,shutil,json, math
import pikepdf

import argparse

from pdb import set_trace as BP

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Description:
      {name}: Split pdf into pages

    Synopsis:
      {name} --fname <fname>.pdf

    Example:
      {name} --fname hauenstein.pdf

    Output goes to files <fname>_<n>.pdf .

''' 
    msg += '\n '
    return msg

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--fname', required=True)
    args = parser.parse_args()

    my_pdf = pikepdf.Pdf.open(args.fname)
    for pnum, page in enumerate(my_pdf.pages):
        dst = pikepdf.Pdf.new()
        dst.pages.append(page)
        outfn = '%s_%04d.pdf' % (os.path.splitext(args.fname)[0],pnum+1)
        dst.save(outfn)

if __name__ == '__main__':
    main()
