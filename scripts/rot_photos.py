#!/usr/bin/env python

# Rotate all *JPG files in the current folder so that the image header
# matches the actual image rotation.
# Then you can rotate them manually for upload to hauenstein.nine.ch/andiconny .

# AHN, Jan 2020

from __future__ import division, print_function
import os,sys,re,glob,shutil
import subprocess
import argparse
from pdb import set_trace as BP


#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Rotate jpeg images consistent with header
    Synopsis:
      %s --run
    Description:
       Rotate all *JPG files in the current folder so that the image header
       matches the actual image rotation. Then you can rotate them manually
       for upload to hauenstein.nine.ch/andiconny .
       HEIC images are converted to jpg on the way.
    Example:
      %s --run
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#--------------
def main():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--run", required=True, action='store_true')
    args = parser.parse_args()

    IMG_FOLDER = '.'

    images =  glob.glob(IMG_FOLDER + '/*.jpg')
    images += glob.glob(IMG_FOLDER + '/*.jpeg')
    images += glob.glob(IMG_FOLDER + '/*.JPG')
    images += glob.glob(IMG_FOLDER + '/*.JPEG')
    images += glob.glob(IMG_FOLDER + '/*.HEIC')

    ORIGFOLDER = 'orig'
    if not os.path.exists( ORIGFOLDER):
        os.mkdir( ORIGFOLDER)
    for img in images:
        shutil.move( img, ORIGFOLDER)

    for img in images:
        print( img)
        inf = os.path.basename( img)
        ext = os.path.splitext( inf)[1]
        jpgfile = '%s.%s' % (os.path.splitext( inf)[0], 'jpg')
        if ext == '.HEIC':
            cmd = 'convert %s/%s %s/%s' % (ORIGFOLDER, inf, ORIGFOLDER, jpgfile)
            subprocess.check_output( cmd, shell=True)
            inf = jpgfile

        cmd = 'ffmpeg -i %s/%s -c:a copy %s' % (ORIGFOLDER, inf, jpgfile)
        subprocess.check_output( cmd, shell=True)


if __name__ == '__main__':
    main()
