#!/usr/bin/python

# Convert a PNG into all the sizes needed for an iOS app

# Author: Andreas Hauenstein
# Date: Feb 2015

import sys
from subprocess import call
#call(["ls", "-l"])

#----------
def usage():
#----------
  msg = '''

usage: make_ios_icons.py <myimage>.png

Generates all sizes of myimage needed for an iOS app icon.

'''
  print(msg)
  exit(1)


#------------
def main():
#------------
  if len(sys.argv) < 2:
    usage()
  inf = sys.argv[1]

  for size in (20,29,40,58,60,76,80,87,120,152,167,180,1024) :
    cmd = 'convert %s -resize %dx%d -quality 100 Icon-%d.png' % (inf,size,size,size)
    print(cmd)
    call (["convert",inf,"-resize","%dx%d" % (size,size),"-quality","100","Icon-%d.png" % size])

# Icon-80.png

if __name__ == '__main__':
  main()
