#!/usr/bin/env python

# Helper funcs for 3D tiling puzzles
# AHN, Dec 2020

import json
import numpy as np
import hashlib

#-----------------
def hhash( sstr):
    ' Produce a reproducible hash from a string'
    sstr = sstr.encode('utf8')
    m = hashlib.sha256()
    m.update( sstr)
    res = m.hexdigest()
    return res

#------------------------------
def hash_piece_image_3d( image):
    '''
    Compute a hopefully unique hash identifying the original piece
    of which this image is an instance.
    '''
    image = trim_array( image)
    h = ''
    for rot in rotations3D( image):
        h = max( h, hhash( repr(rot)))
    return h

#-----------------------------
def parse_puzzle( fname):
    ' Read a puzzle definition (any number os dimensions) from json '
    with open( fname) as f:
        puzzle = json.load(f)
    dims = puzzle['dims']
    piece_counts = puzzle['piece_counts']
    piece_names = list(puzzle['pieces'].keys())
    pieces = []
    for p in puzzle['pieces']:
        pieces.append( np.array( puzzle['pieces'][p]))
    return pieces, piece_names, piece_counts, dims, puzzle.get( 'one_sided', False)

#---------------------
def trim_array( a):
    '''
    Trim the zeros off a numpy array, along all dimensions,
    to get the smallest brick containing all nonzero elements.
    '''
    s = a.shape
    goners = []
    for axis in range( a.ndim):
        goners.append( [])
        # Trim left
        for subspace_idx in range( s[axis]): # subspace has n-1 dimensions
            subspace_values = a.take( subspace_idx, axis)
            if np.sum( subspace_values) == 0:
                goners[-1].append( subspace_idx)
            else:
                break
        # Trim right
        for subspace_idx in range( s[axis]):
            idx = -1 - subspace_idx
            subspace_values = a.take( idx, axis)
            if np.sum( subspace_values) == 0:
                goners[-1].append( idx)
            else:
                break
    # Delete goners
    for axis,goner_idxs in enumerate( goners):
        a = np.delete( a, goner_idxs, axis=axis)

    return a

#---------------------------
def rotations3D( brick):
    ' Return a list with all 24 rotations of a brick, or less if symmetries exist '
    h = brick.shape[0]
    w = brick.shape[1]
    d = brick.shape[2]
    res = []
    strs = set()

    for i in range(6):
        # Top remaining, four sides can face forward
        for j in range(4):
            #print('----------')
            #print(brick)
            if not repr(brick) in strs:
                res.append( brick)
            strs.add( repr(brick))
            brick = rot_top( brick)
        # Change top
        if i < 3:
            #print('>>>>>>>>>>>>>> < 3')
            brick = rot_front( brick)
        elif i == 3: # bring back to top
            #print('############## == 3')
            brick = rot_front( brick) # initial orientation
            brick = rot_left( brick)
        elif i == 4: # bring front to top
            #print('$$$$$$$$$$$$$ == 4')
            brick = rot_left( brick)
            brick = rot_left( brick)

    return res

#-------------------------------------
def add_window( arr, window, r, c, l):
    ' Add window to arr at position r,c,l for left upper corner of win in arr '
    arr[ r:r+window.shape[0], c:c+window.shape[1], l:l+window.shape[2] ] += window

#---------------------------
def rot_top( brick):
    ' Rotate a brick clockwise viewed from top '
    return np.rot90( brick,1,(2,0))

#---------------------------
def rot_front( brick):
    ' Rotate a brick clockwise viewed from front '
    return np.rot90( brick,1,(2,1))

#---------------------------
def rot_left( brick):
    ' Rotate a brick clockwise viewed from left '
    return np.rot90( brick,1,(0,1))

#---------------------------
def isnumeric(s):
    try:
        res = float( s)
        return True
    except ValueError:
        return False
