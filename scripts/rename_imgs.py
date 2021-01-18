#!/usr/bin/env python

'''
For all images in current folder, get time taken and rename to a sortable filename
AHN, Jan 2021
'''

from pdb import set_trace as BP
import os,re
from PIL import Image

files = os.listdir()
files = [f for f in files if os.path.splitext(f)[1].lower() in ['.mov','.heic','.jpg','.jpeg']]
data = {}

newfiles = []
for idx,fname in enumerate(files):
    try:
        print( 'Working on %s' % fname)
        info = os.popen( 'exiftool %s | grep "Create Date" | head -1' % fname).read()
        info = re.sub( r'^[^:]*:','',info)
        dt = info.split()[0]
        t =  info.split()[1]
        creation_date = dt + '_' + t # '2020-12-25_13:48:32'
        creation_date = re.sub( r':','-',creation_date) # '2020-12-25_13-48-32'
        base,ext = os.path.splitext(fname)
        if ext.lower() == '.heic': # ios format to jpg
            print( 'Converting %s to jpg' % fname)
            os.system( 'mogrify -monitor -format jpg %s' %fname)
            fname = base + '.jpg'
        data[fname] = {'cd':creation_date}
        newfiles.append( fname)
    except Exception as e:
        print( 'Could not get metadata for file %s' % fname)
        print( str(e))
        data[fname] = {'cd':'0000'}

files = sorted( newfiles, key = lambda x: data[x]['cd'])

for idx, fname in enumerate(files):
    new_fname = '%04d_' % idx + data[fname]['cd'] + '_' + fname

    print( 'renaming %s to %s' % (fname, new_fname))
    os.rename( fname, new_fname)
