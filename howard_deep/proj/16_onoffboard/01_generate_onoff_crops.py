#!/usr/bin/env python

# /********************************************************************
# Filename: generate_training_crops.py
# Author: AHN
# Creation Date: Mar 5, 2018
# **********************************************************************/
#
# Get crops to train a two class classifier onboard/offboard.
# Crops are taken *before* perspective transform.
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json,copy
import numpy as np
from numpy.random import random, randint
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

import cv2

CROPSZ = 23

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --   Get crops to train a two class classifier onboard/offboard
    Synopsis:
      %s --infolder <ifolder> --outfolder <ofolder>
    Description:
      Gets an equal number of crops on and off the board.
      The on-board crops are intersections, the off-board ones are random.
    Example:
      %s --infolder ~/kc-trainingdata/andreas/20180227 --outfolder kc-onoff-crops
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Collect matching jpeg and sgf in a dictionary
#----------------------------------------------------
def collect_files( infolder):
    # Find images
    imgs =  ut.find( infolder, '[!.]*.jpeg')
    imgs += ut.find( infolder, '[!.]*.jpg')
    imgs += ut.find( infolder, '[!.]*.png')
    # Basenames
    basenames = [os.path.basename(f) for f in imgs]
    basenames = [os.path.splitext(f)[0] for f in basenames]
    # json files
    # jsons = []
    # if ut.find( infolder, '*_intersections.json'):
    #     jsons = [ut.find( infolder, '%s_intersections.json' % f)[0] for f in basenames]
    sgfs = []
    if ut.find( infolder, '*.sgf'):
        sgfs = [ut.find( infolder, '%s.sgf' % f)[0] for f in basenames]

    # Collect in dictionary
    files = {}
    for i,bn in enumerate( basenames):
        d = {}
        files[bn] = d
        d['img'] = imgs[i]
        #if jsons: d['json'] = jsons[i]
        if sgfs:  d['sgf']  = sgfs[i]
    # Sanity check
    for bn in files.keys():
        d = files[bn]
        if not bn in d['img']:
            print( 'ERROR: Wrong img name for key %s' % (d['img'], bn))
            exit(1)
        # elif jsons and not bn in d['json']:
        #     print( 'ERROR: Wrong json name for key %s' % (d['json'], bn))
        #     exit(1)
        elif sgfs and not bn in d['sgf']:
            print( 'ERROR: Wrong sgf name for key %s' % (d['sgf'], bn))
            exit(1)
    return files

# Rescale image and intersections to width 350
#-----------------------------------------------
def scale_350( imgfile, jsonfile):
    TARGET_WIDTH = 350
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

    orig_height = img.shape[0]
    orig_width = img.shape[1]

    # orig_width = img.shape[1]
    # if orig_with == WIDTH:
    #     return( img, intersections)

    # Perspective transform
    #-------------------------
    # Corners
    tl = (0,0)
    tr = (orig_width-1,0)
    br = (orig_width-1, orig_height-1)
    bl = (0, orig_height-1)
    corners = np.array( [tl,tr,br,bl], dtype = "float32")

    # Target coords for transform
    scale = TARGET_WIDTH / orig_width
    target = np.array([
        (0,0),
        (TARGET_WIDTH - 1, 0),
        (TARGET_WIDTH - 1, orig_height * scale - 1),
        (0,  orig_height * scale - 1)], dtype = "float32")
    M = cv2.getPerspectiveTransform( corners, target)
    scaled_img = cv2.warpPerspective( img, M, (TARGET_WIDTH, int(orig_height * scale)))

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
        # Mark if off screen
        marg = int( CROPSZ / 2.0 + 1)
        if (isec['x'] < marg
            or isec['y'] < marg
            or isec['x'] > orig_width - marg
            or isec['y'] > orig_height - marg):
            nnew['off_screen'] = 1
    res = (scaled_img, intersections_zoomed)
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
        if color in ['I','O'] and not 'off_screen' in isec:
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
    if not 'KifuCam' in sgf:
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

# Randomly find boardsize*boardsz offboard crops and save them to folder.
# This won't work if the board fills the image. Print a warning and return.
#----------------------------------------------------------------------------
def save_offboard_crops( img, intersections, r, basename, folder):
    boardsz = int( np.sqrt( len( intersections)) + 0.5)
    tl = intersections[0]
    tr = intersections[boardsz-1]
    br = intersections[boardsz*boardsz-1]
    bl = intersections[boardsz*boardsz - boardsz]
    cnt = np.array (
        ((tl['x'], tl['y']),
        (tr['x'], tr['y']),
        (br['x'], br['y']),
        (bl['x'], bl['y']))
        , dtype='int32' )

    dgrid = np.round( cv2.norm( (bl['x'],bl['y']), (br['x'], br['y']) ) / (boardsz - 1))
    height = img.shape[0]
    width  = img.shape[1]

    if bl['y'] - tl['y'] > height * 0.9:
        print( 'ERROR: board fills image')
        return

    outside_points = []
    marg = r // 2 + 1
    for i in range( boardsz * boardsz):
        d = 0
        while d > -dgrid:
            x = randint( marg, width-marg)
            y = randint( marg, height-marg)
            d = cv2.pointPolygonTest (cnt, (x,y), measureDist=True)
        isec = {'x':x, 'y':y, 'val':'O'} # 'O' like OUT
        outside_points += [isec]
    save_intersections( img, outside_points, r, basename, folder)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--infolder",      required=True)
    parser.add_argument( "--outfolder",     required=True)
    args = parser.parse_args()

    os.makedirs( args.outfolder)
    files = collect_files( args.infolder)

    for i,k in enumerate( files.keys()):
        print( '%s ...' % k)
        f = files[k]
        f['json'] = os.path.dirname( f['img']) + '/%s_intersections.json' % k
        make_json_file( f['sgf'], f['json'])
        # BP()
        img, intersections = scale_350( f['img'], f['json'])
        if len(intersections) != 19*19:
            print( 'not a 19x19 board, skipping')
            continue
        for isec in intersections: isec['val'] = 'I' # I like 'IN'
        save_intersections(  img, intersections, CROPSZ, k, args.outfolder)
        save_offboard_crops( img, intersections, CROPSZ, k, args.outfolder)


if __name__ == '__main__':
    main()