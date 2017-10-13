#!/usr/bin/env python

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
      %s --  Resize jpeg images
    Synopsis:
      %s --percent <percent>
    Description:
      Run convert on all *.jpg *.JPG files in the current folder.
      A backup of the originals is stored in a subfolder "orig".
    Example:
      %s --percent 25
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#--------------
def main():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--percent", required=True, type=int)
    args = parser.parse_args()

    IMG_FOLDER = '.'

    images =  glob.glob(IMG_FOLDER + '/*.jpg')
    images += glob.glob(IMG_FOLDER + '/*.jpeg')
    images += glob.glob(IMG_FOLDER + '/*.JPG')
    images += glob.glob(IMG_FOLDER + '/*.JPEG')

    ORIGFOLDER = 'orig'
    if not os.path.exists( ORIGFOLDER):
        os.mkdir( ORIGFOLDER)
    for img in images:
        shutil.move( img, ORIGFOLDER)

    for img in images:
        print( img)
        cmd = 'convert  %s/%s -resize %d%% %s' % (ORIGFOLDER, img, args.percent, img)
        subprocess.check_output( cmd, shell=True)


if __name__ == '__main__':
    main()
