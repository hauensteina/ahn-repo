#!/usr/bin/env python

# Script to renumber all files in a folder consecutively, after sorting them.
# aaa.sgf aaa.png bbb.sgf bbb.png -> <prefix>0001.sgf <prefix>0001.png <prefix>0002.sgf <prefix>0002.png

from __future__ import division, print_function
import os,sys,re,glob,shutil
import argparse
import subprocess
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPTPATH)

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Rename files in current folder to be numbered consecutively
    Synopsis:
      %s --prefix <prefix>
    Description:
      aaa.sgf aaa.png bbb.sgf bbb.png -> <prefix>_0001.sgf <prefix>_0001.png <prefix>_0002.sgf <prefix>_0002.png
      A backup of the old files will be saved to the backup subfolder
    Example:
      %s --prefix testcase_
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#--------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--prefix", required=True)
    args = parser.parse_args()

    # Make backup
    BACKUP_FOLDER='renum_backup'
    try:
        shutil.rmtree( BACKUP_FOLDER)
    except:
        pass
    os.mkdir( BACKUP_FOLDER)
    files = os.listdir( '.')
    files = [f for f in files if os.path.isfile(f)]
    files = sorted( files)
    for f in files:
        shutil.copy2( f, BACKUP_FOLDER)

    # Make a rename map based on basename
    newnames = {}
    fnum = 0
    for f in files:
        basename = f.split( os.extsep, 1)[0]
        if not basename in newnames:
            fnum += 1
            newnames[basename] = '%s%04d' % (args.prefix,fnum)

    # Move files based on basename, keep extension
    for f in files:
        basename, ext = f.split( os.extsep, 1)
        newbase = newnames[basename]
        newname = newbase + '.' + ext
        print( '%s -> %s' % (f, newname))
        shutil.move( f, newname)

if __name__ == '__main__':
    main()
