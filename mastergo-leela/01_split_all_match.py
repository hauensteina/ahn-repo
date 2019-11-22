#!/usr/bin/env python

# Split an sgf files with many games into separate sgf files for mastergo import.
# AHN, Nov 2019

import os,sys,shutil
import argparse
from pdb import set_trace as BP

OUTFOLDER = 'split_sgfs'
MINDATE = '2018-08-01'
FILES_PER_FOLDER = 10 * 1000

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = '''

    Description:
      %s: Split a multi-sgf file into separate sgf files for mastergo import
    Example:
      %s --file all_match.sgf
    Output goes to folders %s/n, where n increases every %d games.

--
''' % (name, name, OUTFOLDER, FILES_PER_FOLDER)

    return msg

#------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--file", required=True)
    args = parser.parse_args()

    if os.path.exists( OUTFOLDER):
        print( '%s exists. Please move it away.' % OUTFOLDER)
        exit(1)

    os.mkdir( OUTFOLDER)
    run( args.file)

#-------------------------
def run( infname):
    sgf = ''
    sgfcount = 0
    foldercount = 0

    print( 'patience please, chugging through old games...')
    with open( infname) as inf:
        for line in inf:
            line = line.strip()
            if line.startswith( '(;GM'):
                if sgf:
                    ofname, dt = getfname( sgf)
                    if ofname and (dt >= MINDATE):
                        if not sgfcount % FILES_PER_FOLDER:
                            foldercount += 1
                            outfolder = '%s/%0.3d' % (OUTFOLDER, foldercount)
                            os.mkdir( outfolder)
                        with open( outfolder + '/' + ofname, 'w') as of:
                            print( ofname)
                            of.write( sgf)
                            sgfcount += 1
                    sgf = ''

            sgf += line + '\n'

# Return a god outfile name for an sgf game record.
# Also returns the date played.
#----------------------------------------------------
def getfname( sgf):
    getfname.gamenum += 1
    tstr = sgf.split( 'DT[')[1].strip()
    dt = tstr.split( ']')[0].strip()
    fname = '%0.10d-%s.sgf' % (getfname.gamenum, dt)
    return fname, dt
getfname.gamenum = 0

if __name__ == '__main__':
    main()
