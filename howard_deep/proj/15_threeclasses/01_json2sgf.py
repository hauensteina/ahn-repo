#!/usr/bin/env python

# /********************************************************************
# Filename: json2sgf.py
# Author: AHN
# Creation Date: Feb 28, 2018
# **********************************************************************/
#
# Copy intersection coords from wallstedt format json to sgf GC tag
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json,copy
import numpy as np
from numpy.random import random
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Copy intersection coords from wallstedt format json to sgf GC tag
    Synopsis:
      %s --folder <folder>
    Description:
      Finds all pairs (*_intersections.json, *.sgf) and cpoies the
      intersections coords into the sgf GC tag.
      The results go into a subfolder <folder>/json2sgf.
    Example:
      %s --folder ~/kc-trainingdata/andreas/mike_fixed_20180228
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Collect matching jpeg and json in a dictionary
#----------------------------------------------------
def collect_files( infolder):
    # Find images
    jsons =  ut.find( infolder, '[!.]*_intersections.json')
    # Basenames
    basenames = [os.path.basename(f) for f in jsons]
    basenames = [re.sub( '_intersections.json','',x) for x in basenames]
    sgfs = [ut.find( infolder, '%s.sgf' % f)[0] for f in basenames]
    res = zip( jsons, sgfs)
    return res

# Rescale image and intersections to width 350.
# Make the board square.
# Returns the new image and the transformed list of intersections like
# [{u'y': 17, u'x': 17, u'val': u'EMPTY'}, ...
#------------------------------------------------
def zoom_in( imgfile, jsonfile):
    # Read the image
    img = cv2.imread( imgfile, 1)
    # Parse json
    columns = json.load( open( jsonfile))
    # Linearize
    board_sz = len(columns)
    intersections = [0] * (board_sz * board_sz)
    for c,col in enumerate( columns):
        for r, row in enumerate( col):
            idx = board_sz * r + c
            intersections[idx] = row
    #         x = row['x']
    #         y = row['y']
    #         cv2.circle( img,
    #                     (int(x), int(y)),
    #                     10,
    #                     ( 0, 0, 255 ),
    #                     -1 )
    # cv2.imwrite( '/home/ubuntu/ahn-repo/howard_deep/proj/15_threeclasses/tt.jpg', img);

    # Perspective transform
    #-------------------------
    # Corners
    tl = intersections[0]
    tr = intersections[board_sz-1]
    br = intersections[board_sz * board_sz - 1]
    bl = intersections[board_sz * board_sz - board_sz]
    corners = np.array([
        [tl['x'], tl['y']],
        [tr['x'], tr['y']],
        [br['x'], br['y']],
        [bl['x'], bl['y']]], dtype = "float32")

    WIDTH = 350
    marg = WIDTH / 20.0;
    orig_height = img.shape[0]
    orig_width = img.shape[1]
    orig_marg = orig_width / 20.0
    # Target square for transform
    square = np.array([
        [marg, marg],
        [WIDTH - marg, marg],
        [WIDTH - marg, WIDTH - marg],
        [marg, WIDTH - marg]], dtype = "float32")
    M = cv2.getPerspectiveTransform( corners, square)
    warped_img = cv2.warpPerspective( img, M, (WIDTH, WIDTH))

    coords = []
    for isec in intersections:
        coords.append( [isec['x'], isec['y']])
    coords = np.array( coords)
    # Transform the intersections
    # This needs a stupid empty dimension added
    sz = len(coords)
    coords_zoomed = cv2.perspectiveTransform( coords.reshape( 1, sz, 2).astype('float32'), M)
    # And now get rid of the extra dim and back to int
    coords_zoomed = coords_zoomed.reshape(sz,2).astype('int')
    # Back to the old format
    intersections_zoomed = []
    for idx,isec in enumerate( intersections):
        intersections_zoomed.append( isec.copy())
        nnew = intersections_zoomed[-1]
        nnew['x'] = coords_zoomed[idx][0]
        nnew['y'] = coords_zoomed[idx][1]
        # Mark if off board
        if (isec['x'] < orig_marg
            or isec['y'] < orig_marg
            or isec['x'] > orig_width - marg
            or isec['y'] > orig_height - marg):
            nnew['off_board'] = 1
    res = (warped_img, intersections_zoomed)
    return res

# Save intersection crops of size rxr
#-------------------------------------------------------------------
def save_intersections( img, intersections, r, basename, folder):
    dx = int(r / 2)
    dy = int(r / 2)
    for i,isec in enumerate( intersections):
        color = isec['val'][0]
        x = isec['x']
        y = isec['y']
        hood = img[y-dy:y+dy+1, x-dx:x+dx+1]
        fname = "%s/%s_rgb_%s_hood_%03d.jpg" % (folder, color, basename, i)
        if color in ['B','W','E'] and not 'off_board' in isec:
            cv2.imwrite( fname, hood)

# e.g for board size, call get_sgf_tag( sgf, "SZ")
#---------------------------------------------------
def get_sgf_tag( tag, sgf):
    m = re.search( tag + '\[[^\]]*', sgf)
    if not m: return ''
    mstr = m.group(0)
    res = mstr.split( '[')[1]
    res = res.split( ']')[0]
    return res

# Read and sgf file and linearize it into a list
# ['b','w','e',...]
#-------------------------------------------------
def linearize_sgf( sgf):
    boardsz = int( get_sgf_tag( 'SZ', sgf))
    if 'GOWrite' in sgf:
        # The AW[ab][ce]... case
        match = re.search ( 'AW(\[[a-s][a-s]\])*', sgf)
        whites = match.group(0)
        whites = re.sub( 'AW', '', whites)
        whites = re.sub( '\[', 'AW[', whites)
        whites = re.findall( 'AW' + '\[[^\]]*', whites)
        match = re.search ( 'AB(\[[a-s][a-s]\])*', sgf)
        blacks = match.group(0)
        blacks = re.sub( 'AB', '', blacks)
        blacks = re.sub( '\[', 'AB[', blacks)
        blacks = re.findall( 'AB' + '\[[^\]]*', blacks)
    else:
        # The AW[ab]AW[ce]... case
        whites = re.findall( 'AW' + '\[[^\]]*', sgf)
        blacks = re.findall( 'AB' + '\[[^\]]*', sgf)

    res = ['EMPTY'] * boardsz * boardsz
    for w in whites:
        pos = w.split( '[')[1]
        col = ord( pos[0]) - ord( 'a')
        row = ord( pos[1]) - ord( 'a')
        idx = col + row * boardsz
        res[idx] = 'WHITE'

    for b in blacks:
        pos = b.split( '[')[1]
        col = ord( pos[0]) - ord( 'a')
        row = ord( pos[1]) - ord( 'a')
        idx = col + row * boardsz
        res[idx] = 'BLACK'

    return res

# Make a Wallstedt type json file from an sgf with the
# intersection coordinates in the GC tag
#--------------------------------------------
def make_json_file( sgffile, ofname):
    with open( sgffile) as f: sgf = f.read()
    sgf = sgf.replace( '\\','')
    if not 'intersections:' in sgf and not 'intersections\:' in sgf:
        print('no intersections in ' + sgffile)
        return
    boardsz = int( get_sgf_tag( 'SZ', sgf))
    diagram = linearize_sgf( sgf)
    intersections = get_sgf_tag( 'GC', sgf)
    intersections = re.sub( '\(','[',intersections)
    intersections = re.sub( '\)',']',intersections)
    intersections = re.sub( 'intersections','"intersections"',intersections)
    intersections = '{' + intersections + '}'
    intersections = json.loads( intersections)
    intersections = intersections[ 'intersections']
    elt = {'x':0, 'y':0, 'val':'EMPTY'}
    coltempl = [ copy.deepcopy(elt) for _ in range(boardsz) ]
    res = [ copy.deepcopy(coltempl) for _ in range(boardsz) ]
    for col in range(boardsz):
        for row in range(boardsz):
            idx = row * boardsz + col
            res[col][row]['val'] = diagram[idx]
            res[col][row]['x'] = intersections[idx][0]
            res[col][row]['y'] = intersections[idx][1]
    jstr = json.dumps( res)
    with open( ofname, 'w') as f: f.write( jstr)

# Read both files and return sgf with intersections in GC tag
#-------------------------------------------------------------
def fix( jsonfile, sgffile):
    jobj = json.load( open( jsonfile))
    sgf  = open( sgffile).read()
    boardsz = len(jobj)
    coords = [0] * boardsz * boardsz
    if boardsz != 19:
        print ('Size %d is not a 19x19 board. Skipping' % boardsz)
        return ''
    for c,col in enumerate( jobj):
        for r,isec in enumerate( col):
            idx = r * boardsz + c
            coords[idx] = (isec['x'],isec['y'])
    tstr = json.dumps( coords)
    tstr = re.sub( '\[','(',tstr)
    tstr = re.sub( '\]',')',tstr)
    tstr = 'GC[intersections:' + tstr + ']'
    res = sgf
    res = re.sub( '(SZ\[[^\]]*\])', r'\1' + tstr, res)
    return res

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--folder",      required=True)
    args = parser.parse_args()

    outfolder = args.folder + '/json2sgf/'
    os.makedirs( outfolder)
    file_pairs = collect_files( args.folder)
    for p in file_pairs:
        newsgf = fix( p[0], p[1])
        fname = outfolder + os.path.basename( p[1])
        with open( fname, 'w') as f:
            f.write( newsgf)

if __name__ == '__main__':
    main()
