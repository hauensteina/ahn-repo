#!/usr/bin/env python

'''
Take images from the specified folders, get time taken and rename to a sortable filename
AHN, Jan 2021
'''

from pdb import set_trace as BP
import os,sys,re
import glob,shutil
import argparse

GOODFOLDER = 'rename_imgs_converted'
BADFOLDER  = 'rename_imgs_failed'

#-----------------------------
def usage( printmsg=False):
    name = os.path.basename( __file__)
    msg = '''

    Name:
      %s: Rename exported photos and movies for web publishing.
    Synopsis:
      %s --folder <infolder>
    Description:
       All image and movie files found somewhere below folder will be
       sorted by image taken date and numbered.
       Successfully converted files go to a new folder 'rename_imgs_converted'.
       If the date was not found in the exif data, the image goes to 'rename_images_failed'.
    Examples:
      %s --folder pics_2020

--
''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg


#-------------
def main():
    if len(sys.argv) == 1: usage( True)
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--folder", required=True)
    args = parser.parse_args()

    if os.path.exists( GOODFOLDER):
        print( 'ERROR: %s exists. Please move it away.' % GOODFOLDER)
        exit(1)

    if os.path.exists( BADFOLDER):
        print( 'ERROR: %s exists. Please move it away.' % BADFOLDER)
        exit(1)

    os.mkdir( GOODFOLDER)
    os.mkdir( BADFOLDER)

    files = glob.glob( args.folder + '/**', recursive=True)
    files = [f for f in files if os.path.splitext(f)[1].lower() in ['.mov','.heic','.jpg','.jpeg','.png']]
    data = {}

    for idx,fname in enumerate(files):
        try:
            print( 'Working on %s' % fname)
            info = os.popen( 'exiftool %s | grep "Create Date" | head -1' % fname).read()
            if not info:
                info = os.popen( 'exiftool %s | grep "Date Created" | head -1' % fname).read()
            if not info:
                data[fname] = {'cd':'date_missing'}
            info = re.sub( r'^[^:]*:','',info)
            dt = info.split()[0]
            t =  info.split()[1]
            creation_date = dt + '_' + t # '2020-12-25_13:48:32'
            creation_date = re.sub( r':','-',creation_date) # '2020-12-25_13-48-32'
            data[fname] = {'cd':creation_date}
        except Exception as e:
            print( 'Exception for file %s' % fname)
            print( str(e))
            data[fname] = {'cd':'date_missing'}

    files = sorted( files, key = lambda x: data[x]['cd'])

    for idx, fname in enumerate(files):
        base = os.path.split( fname)[-1]
        if 'date_missing' in data[fname]['cd']:
            new_fname = BADFOLDER + '/%04d_' % idx + data[fname]['cd'] + '_' + base
        else:
            new_fname = GOODFOLDER + '/%04d_' % idx + data[fname]['cd'] + '_' + base
        print( 'Copying %s to %s' % (fname, new_fname))
        shutil.copy( fname, new_fname)
        base,ext = os.path.splitext( fname)
        if ext.lower() == '.heic': # ios format to jpg
            print( 'Converting %s to jpg' % fname)
            os.system( 'mogrify -monitor -format jpg %s' % new_fname)
            os.remove( new_fname)

if __name__ == '__main__':
    main()
