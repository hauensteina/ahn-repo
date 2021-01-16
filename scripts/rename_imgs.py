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

for idx,fname in enumerate(files):
    try:
        info = os.popen( 'mdls %s | grep kMDItemContentCreationDate | head -1' % fname).read()
        dt = info.split()[2]
        t =  info.split()[3]
        creation_date = dt + '_' + t # '2020-12-25_13:48:32'
        creation_date = re.sub( r':','-',creation_date) # '2020-12-25_13-48-32'
        data[fname] = {'cd':creation_date}
    except:
        print( 'Could not get metadata for file %s' % f)
        data[fname] = {'cd':'0000'}

files = sorted( files, key = lambda x: data[x]['cd'])

for idx, fname in enumerate(files):
    new_fname = '%04d_' % idx + data[fname]['cd'] + '_' + fname
    print( 'renaming %s to %s' % (fname, new_fname))
    os.rename( fname, new_fname)
