#!/usr/bin/env python

# Tile a s^3 cube with the given pieces, using Knuth's Algorithm X
# AHN, Nov 2020

from pdb import set_trace as BP
import sys,os,pickle
import argparse
import numpy as np
import tiling_helpers as helpers
from algox_assaf import AlgoX

OUTFILE = 'algo_x_3d_solutions.pickle'

g_pieces = {
    '3x3x3':
    [
        # The Waiter
        np.array([
            [   # Rear Plane
                [0,0],
                [1,0],
                [0,0]
            ],
            [   # Front plane
                [1,0],
                [1,0],
                [1,1]
            ]
        ]),
        # The Pillar
        np.array([
            [
                [1,0,0],
                [1,1,1]
            ],
            [
                [0,0,0],
                [1,0,0]
            ]
        ]),
        # The Cobra
        np.array([
            [
                [0,0],
                [0,0],
                [1,1]
            ],
            [
                [0,1],
                [0,1],
                [0,1]
            ]
        ]),
        # The Kiss
        np.array([
            [
                [1,1]
            ]
        ]),
        # The L-wiggle
        np.array([
            [
                [1,0,0],
                [0,0,0]
            ],
            [
                [1,1,1],
                [0,0,1]
            ]
        ]),
        # The Mirror-wiggle
        np.array([
            [
                [0,0],
                [1,1]
            ],
            [
                [0,0],
                [1,0]
            ],
            [
                [1,0],
                [1,0]
            ]
        ])
    ],
    'abraxis':
    [
        # The Pillar
        np.array([
            [   # Rear Plane
                [0,0,1],
                [1,1,1]
            ],
            [   # Front plane
                [0,0,0],
                [0,0,1]
            ]
        ]),
        # Letter S plus Head
        np.array([
            [
                [0,0,0],
                [0,1,1]
            ],
            [
                [0,1,0],
                [1,1,0]
            ]
        ]),
        # Yellow Wiggle
        np.array([
            [
                [0,1,1],
                [1,1,0]
            ],
            [
                [0,0,0],
                [1,0,0]
            ]
        ]),
        # The Cannon
        np.array([
            [
                [0,0,0],
                [0,1,0]
            ],
            [
                [0,1,0],
                [1,1,1]
            ]
        ]),
        # Letter L plus Head
        np.array([
            [
                [0,0,0],
                [0,0,1]
            ],
            [
                [0,1,0],
                [1,1,1]
            ]
        ]),
        # The Barkeeper
        np.array([
            [
                [0,1,0],
                [0,1,0]
            ],
            [
                [0,0,0],
                [1,1,1]
            ]
        ]),
        # Two plus Head
        np.array([
            [
                [0,1,0],
                [1,1,0]
            ],
            [
                [0,0,0],
                [0,1,1]
            ]
        ]),
        # Letter L Snake
        np.array([
            [
                [0,0,0],
                [0,0,1]
            ],
            [
                [1,0,0],
                [1,1,1]
            ]
        ]),
        # The Shamrock
        np.array([
            [
                [0,1],
                [1,1]
            ],
            [
                [0,0],
                [0,1]
            ]
        ]),
        # Left Stack
        np.array([
            [
                [1,1,0],
                [0,1,0]
            ],
            [
                [0,0,0],
                [0,1,1]
            ]
        ]),
        # Right Stack
        np.array([
            [
                [0,1,1],
                [0,1,0]
            ],
            [
                [0,0,0],
                [1,1,0]
            ]
        ]),
        # Roman Couch
        np.array([
            [
                [0,0,0],
                [1,1,1]
            ],
            [
                [0,0,1],
                [0,0,1]
            ]
        ]),
        # Left L plus Head
        np.array([
            [
                [0,1,0],
                [1,1,1]
            ],
            [
                [0,0,0],
                [0,0,1]
            ]
        ])
    ]
} # g_pieces

# Turn the ones into piece number
for k in g_pieces.keys():
    g_pieces[k] = [p * (idx + 1) for idx,p in enumerate( g_pieces[k])]

#-----------------------------
def usage( printmsg=False):
    name = os.path.basename( __file__)
    msg = '''

    Description:
      %s: Solve 3D nxn tiling puzzles.
    Synopsis:
      %s --case <case_id> [--max_solutions <n>]
      %s --json <file>
    Examples:
      %s --case 3x3x3
      %s --case abraxis
      %s --json gelo-1289.json

--
''' % (name,name,name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------------
def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--case")
    parser.add_argument( "--json")
    parser.add_argument( "--max_solutions", type=int)
    args = parser.parse_args()

    if not args.case and not args.json:
        usage( True)

    if args.case:
        solver = AlgoX3D( g_pieces[args.case], max_solutions=args.max_solutions)
    else:
        pieces, piece_names, piece_counts, dims, one_sided = helpers.parse_puzzle( args.json)
        solver = AlgoX3D( pieces, piece_names, piece_counts, args.max_solutions)

    solver.solve()
    print( '\nFound %d solutions' % len( solver.solutions))

#=================================================================================
class AlgoX3D:
    '''
    Knuth's Algorithm X for 3D tiling puzzles.
    We don't do the dancing links (DLX), just use dicts of sets.
    A shifted rotated instance of a piece is called an image.
    '''
    def __init__( self, pieces, piece_names=None, piece_counts=None, max_solutions=0):
        '''
        Build a matrix with a column per gridpoint plus a column per piece.
        The rows are the images (rot + trans) of the pieces that fit in the grid.
        '''

        piece_ids = [chr( ord('A') + x) for x in range( len(pieces))]

        if piece_counts is None:
            self.piece_counts = [1] * len(pieces)
        else:
            self.piece_counts = piece_counts.values()

        if piece_names is None:
            piece_names = [chr( ord('A') + 1 + x) for x in range( len( pieces))]

        if len(pieces) != len(piece_counts):
            print( 'ERROR: Piece counts wrong length: %d' % len(piece_counts))
            exit(1)

        # Make multiple copies of a piece explicit
        newpieces = []
        piece_ids = []
        for idx,c in enumerate( self.piece_counts):
            newpieces.extend( [pieces[idx]] * c)
            piece_ids.extend( [piece_names[idx] + '#' + str(x) for x in range(c)])
        pieces = newpieces
        self.nholes = sum( [np.sum( np.sign(p)) for p in pieces])
        self.size = int( np.cbrt( self.nholes))
        self.dims = (self.size, self.size, self.size)
        if self.nholes != np.prod( self.dims):
            print( 'ERROR: Pieces do not cover grid: %d of %d covered' % (self.nholes, np.prod( self.dims)))
            exit(1)

        rownames = [] # Multiple copies of a piece are different
        rowclasses = [] # Multiple copies of a piece are the same
        colnames = [str(x) for x in range( self.nholes)] + piece_ids
        entries = set() # Pairs (rowidx, colidx)

        def add_image( piece_id, img_id, img, row, col, layer):
            '''
            Add an image to the set of images to try.
            We call the third dimension *layer*
            '''
            rowname = piece_id + '_' + str(img_id) # A#0_1
            rowclass = piece_id.split('#')[0] + '#' + str(img_id) # A#1
            rownames.append( rowname)
            rowclasses.append( rowclass)
            rowidx = len( rownames) - 1
            entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
            cube = np.zeros( (self.size, self.size, self.size))
            helpers.add_window_3D( cube, img, row, col, layer)
            filled_holes = set( np.flatnonzero( cube))
            for h in filled_holes: # Image fills these holes
                colidx = colnames.index( str(h))
                entries.add( (rowidx, colidx) )

        # Add all images
        worst_piece_idx = self.get_worst_piece_idx( pieces) # most symmetries, positions
        for pidx,p in enumerate( pieces):
            rots = helpers.rotations3D( p)
            #print( len(rots))
            piece_id = piece_ids[pidx]
            img_id = -1
            for rotidx,img in enumerate( rots):
                if pidx == worst_piece_idx and rotidx > 0: # Restrict worst piece to eliminate syms
                    break
                for row in range( self.size - img.shape[0] + 1):
                    for col in range( self.size - img.shape[1] + 1):
                        for layer in range( self.size - img.shape[2] + 1):
                            img_id += 1
                            add_image( piece_id, img_id, img, row, col, layer)
        self.solver = AlgoX( rownames, rowclasses, colnames, entries, max_solutions)

    def get_worst_piece_idx( self, pieces):
        '''
        Find the piece with most symmetries and least positions.
        '''
        maxrots = residx = -1
        minpositions = int(1E9)
        for pidx,p in enumerate( pieces):
            rots = helpers.rotations3D( p)
            if len(rots) <= maxrots: continue
            if len(rots) > maxrots: minpositions = int(1E9)
            maxrots = len(rots)
            npositions = (self.size - p.shape[0] + 1) * (self.size - p.shape[1] + 1)* (self.size - p.shape[2] + 1)
            if npositions >= minpositions: continue
            minpositions = npositions
            residx = pidx
        return residx

    def solve( self):
        self.solutions = list(self.solver.solve())
        self.save_solutions()

    def save_solutions( self):
        solutions = []
        for idx,s in enumerate( self.solutions):
            # s is a list of row names like 'H_23'
            solution = []
            for rowname in s:
                piece = rowname.split('_')[0]
                filled_holes = [x for x in self.solver.get_col_idxs( rowname)
                                if x < self.size * self.size * self.size]
                piece_in_cube = np.full( self.size * self.size * self.size, 0)
                for h in filled_holes:
                    if '#' in piece:
                        piece = piece.split('#')[0]
                    piece_in_cube[int(h)] = ord(piece) - ord('A') + 1
                piece_in_cube = piece_in_cube.reshape( self.size, self.size, self.size)
                solution.append( piece_in_cube)
            solutions.append( solution)
        with open( OUTFILE, 'wb') as f:
            pickle.dump( solutions, f)

main()
