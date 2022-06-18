#!/usr/bin/env python

# Script to take a large number of files in a folder and sort them into subfolders by name.
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
      {name} --  Move files into subfolders by name component
    Synopsis:
      {name} --start <int> --length <int> --prefix <string> --extension <string>
    Description:
      Find all files in the current folder with the given extension.
      Take the substring starting at position <start> of length <length> and prepend <prefix>
      to get the target folder name, then move the file to that folder. Create the folder
      if it does not exist. 
    Examples:
      {name} --start 9 --length 6 --prefix m --extension sgf
      {name} --start 2 --length 4 --prefix y 
    ''' 
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#--------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--start", required=True, type=int)
    parser.add_argument( "--length", required=True, type=int)
    parser.add_argument( "--prefix", required=True)
    parser.add_argument( "--extension", default='')
    aargs = parser.parse_args()

    files = os.listdir( '.')
    if aargs.extension:
        files = [f for f in files if os.path.splitext(f)[1] == f'.{aargs.extension}']
    files = sorted(files)
    for idx,f in enumerate(files):
        folder = aargs.prefix + f[aargs.start-1 : aargs.start-1 + aargs.length]
        print( f'{idx+1}/{len(files)} {f} -> {folder}')
        if not os.path.exists(folder):
            os.mkdir(folder)
        shutil.move( f, folder)

    print('Done')

if __name__ == '__main__':
    main()
