#!/usr/bin/env python

# Script to move all files with the given extension to a folder
# AHN Jun 2022

from pdb import set_trace as BP
#import os,sys,re,glob,shutil
import os,shutil
import argparse

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = f'''
    Name:
      {name} -- Move all files with the given extension to a folder
    Synopsis:
      {name} --extension <string> --folder <string>
    Example:
      {name} --extension sgf --folder tt
    ''' 
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#--------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--extension", required=True)
    parser.add_argument( "--folder", required=True)
    aargs = parser.parse_args()

    files = os.listdir( '.')
    files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1] == f'.{aargs.extension}']
    files = sorted(files)
    folder = aargs.folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    for idx,f in enumerate(files):
        print( f'{idx+1}/{len(files)} {f} -> {folder}')
        shutil.move( f, folder)

    print('Done')

if __name__ == '__main__':
    main()
