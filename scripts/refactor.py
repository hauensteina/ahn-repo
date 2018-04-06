#!/usr/bin/env python
# /********************************************************************
# Filename: refactor.py
# Author: AHN
# Creation Date: Feb 22, 2016
# **********************************************************************/
#
# Replace variable and class names in C++ source files
#


from pdb import set_trace as BP
import os,sys,re,json
from collections import defaultdict
import shutil
import argparse
import subprocess
import tempfile


#---------------------------
def usage( printmsg=False):
    msg = '''
    Synopsis:
      refactor.py --oldname <old_var_name> --newname <new_var_name> [--file <sourcefile>] [--filelist <file_with_filenames>]
    Examples:
      refactor.py --oldname Vector --newname Tensor1D --file mysource.cpp
      refactor.py --oldname Vector --newname Tensor1D --filelist sourcefiles.txt
    '''
    if printmsg:
        print msg
        exit(1)
    else:
        return msg

#------------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser=argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--oldname", required=True, type=str)
    parser.add_argument( "--newname", required=True, type=str)
    parser.add_argument( "--filelist", required=False, type=str)
    parser.add_argument( "--file", required=False, type=str)
    args=parser.parse_args()

    if not args.filelist and not args.file:
        print 'Either --file or --filelist is required\n'
        usage()
    if args.filelist:
        files = slurp_lines_stripped( args.filelist)
    else:
        files = [args.file]

    regex = r'([^\w]|^)' # non word char to the left or beginning of line
    regex += '(' + args.oldname + ')'
    regex += r'([^\w]|$)' # non word char to the right or line end
    pat = re.compile(regex)
    replwith = r'\1' + args.newname + r'\3'
    for idx, f in enumerate(files):
        print f
        outsrc = ''
        lines = slurp_lines_raw( f)
        for line in lines:
            # Avoid comments
            parts = line.split( '//')
            parts[0] = re.sub( pat, replwith, parts[0])
            res = '//'.join( parts)
            # sys.stdout.write( res)
            outsrc += res
        with open( f,'w') as of:
            of.write( outsrc)


# Get the lines of a file into an array,
# ignore blank lines and comments
#---------------------------------
def slurp_lines_stripped(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [line for line in lines if line]
        lines = [line for line in lines if not line[0] == '#']
    return lines

# Get the lines of a file into an array,
# no questions asked
#----------------------------
def slurp_lines_raw(fname):
    with open(fname) as f:
        lines = f.readlines()
    return lines



if __name__ == '__main__':
    main()
