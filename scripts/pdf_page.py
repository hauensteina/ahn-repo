#!/usr/bin/env python

# Script to save one page from a pdf

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
      {name}:  Extract one page from a pdf

    Synopsis:
      {name} --page <n> --password <password> file1.pdf file2.pdf ...

    Example:
      {name} --page 5 --password 20036 *.PDF

    Output goes to <fname>_<n>.pdf .

''' 
    msg += '\n '
    return msg

#-------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--password', required=True)
    parser.add_argument( '--page', required=True, type=int)
    parser.add_argument('file', type=str, nargs='+')
    args = parser.parse_args()

    for fname in args.file:
        my_pdf = pikepdf.Pdf.open(fname, password=args.password)
        for pnum, page in enumerate(my_pdf.pages):
            if pnum + 1 == args.page:
                dst = pikepdf.Pdf.new()
                dst.pages.append(page)
                outfn = '%s_%04d.pdf' % (os.path.splitext(fname)[0],args.page)
                dst.save( '%s_%04d.pdf' % (outfn, args.page))
                break

if __name__ == '__main__':
    main()
