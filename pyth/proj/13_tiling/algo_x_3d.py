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
    'test_222':
    [
        np.array([
            [   # Rear Plane
                [1,1]
            ],
            [   # Front plane
                [1,1]
            ]
        ]),
        np.array([
            [
                [1,1]
            ],
            [
                [1,1]
            ]
        ])
    ],
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

    Name:
      %s: Solve 3D nxn tiling puzzles.
    Synopsis:
      %s --case <case_id> [--max_solutions <n>] [--mode basic|nopieces]
      %s --json <file> [--max_solutions <n>] [--mode basic|nopieces]
    Description:
      --mode nopieces
      Run without adding columns for the pieces.
      This only works if one piece repeats many times.
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
    parser.add_argument( "--mode", choices=['basic','nopieces'], default='basic')

    args = parser.parse_args()

    if not args.case and not args.json:
        usage( True)

    if args.case:
        solver = AlgoX3D( g_pieces[args.case], max_solutions=args.max_solutions)
    else:
        pieces, piece_names, piece_counts, dims, one_sided = helpers.parse_puzzle( args.json)
        solver = AlgoX3D( pieces, piece_names, piece_counts, dims, mode=args.mode, max_solutions=args.max_solutions)

    solver.solve()

    # print( '\nFound %d solutions' % len( solver.solutions))

    # dups = solver.find_duplicate_solutions()
    # for d in dups:
    #     n = len(dups[d])
    #     if n == 1: continue
    #     sols = [x['idx'] for x in dups[d]]
    #     print('Sols %s are the same!' % str(sols))


#=================================================================================
class AlgoX3D:
    '''
    Knuth's Algorithm X for 3D tiling puzzles.
    We don't do the dancing links (DLX), just use dicts of sets.
    A shifted rotated instance of a piece is called an image.
    '''
    def __init__( self, pieces, piece_names=None, piece_counts=None, dims=None, mode='basic', max_solutions=0):
        '''
        Build a matrix with a column per gridpoint plus a column per piece.
        The rows are the images (rot + trans) of the pieces that fit in the grid.
        '''
        self.dims = dims
        self.mode = mode
        if dims is None:
            self.nholes = sum( [np.sum( np.sign(p)) for p in pieces])
            size = int(np.cbrt( self.nholes))
            self.dims = (size, size, size)

        if piece_counts is None:
            piece_counts = [1] * len(pieces)
        else:
            piece_counts = list(piece_counts.values())

        if piece_names is None:
            piece_names = [chr( ord('A') + 1 + x) for x in range( len( pieces))]

        if len(pieces) != len( piece_counts):
            print( 'ERROR: Piece counts wrong length: %d' % len(piece_counts))
            exit(1)

        if self.mode == 'nopieces':
            if len(pieces) != 1:
                print( 'ERROR: nopieces mode only works with one repeating piece')
                exit(1)
            piece_counts = [1]

        self.nholes = np.prod( self.dims)
        if self.mode != 'nopieces' and sum(
                [piece_counts[idx] * np.sum( np.sign(p)) for idx,p in enumerate(pieces)]) != self.nholes:
            print( 'ERROR: Pieces do not cover block: %d of %d covered' % (self.nholes, self.nholes))
            exit(1)

        rownames = []
        colnames = [str(x) for x in range( self.nholes)]
        colcounts = [1] * len(colnames)
        if self.mode != 'nopieces':
            colnames += piece_names
            colcounts += piece_counts
        entries = set() # Pairs (rowidx, colidx)

        def add_image( piece_id, img_id, img, row, col, layer):
            '''
            Add an image to the set of images to try.
            We call the third dimension *layer*
            '''
            rowname = piece_id + '_' + str(img_id) # A#0_1
            rownames.append( rowname)
            rowidx = len( rownames) - 1
            if self.mode != 'nopieces':
                entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
            entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
            block = np.zeros( (self.dims[0], self.dims[1], self.dims[2]))
            helpers.add_window_3D( block, img, row, col, layer)
            filled_holes = set( np.flatnonzero( block))
            for h in filled_holes: # Image fills these holes
                colidx = colnames.index( str(h))
                entries.add( (rowidx, colidx) )

        # Add all images
        worst_piece_idx = self.get_worst_piece_idx( pieces) # freeze this piece into one symmetry
        if self.mode == 'nopieces': worst_piece_idx = -1

        for pidx,p in enumerate( pieces):
            if pidx == worst_piece_idx:
                rots = helpers.one_rot_per_shape_3D( p, self.dims)
            else:
                rots = helpers.rotations3D( p)

            piece_id = piece_names[pidx]
            img_id = -1
            for rotidx,img in enumerate( rots):
                for row in range( self.dims[0] - img.shape[0] + 1):
                    for col in range( self.dims[1] - img.shape[1] + 1):
                        for layer in range( self.dims[2] - img.shape[2] + 1):
                            img_id += 1
                            add_image( piece_id, img_id, img, row, col, layer)
        self.solver = AlgoX( rownames, colnames, colcounts, entries, max_solutions)

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
            npositions = (self.dims[0] - p.shape[0] + 1) * (self.dims[1] - p.shape[1] + 1)* (self.dims[2] - p.shape[2] + 1)
            if npositions >= minpositions: continue
            minpositions = npositions
            residx = pidx
        return residx

    def solve( self):
        self.solutions = list(self.solver.solve())

        # Remove dups
        dups = {}
        for s in self.solutions:
            hhash = self.hash_solution( s)
            if not hhash in dups:
                dups[hhash] = [s]
            else:
                dups[hhash].append( s)
        ndups = sum( [ len(x) - 1 for x in dups.values() if len(x) > 1 ] )
        unique = [ x[0] for x in dups.values() ]

        print( 'Found %d solutions with %d dups, %d unique' % ( len(self.solutions), ndups, len(unique)))
        print( 'Saving %d unique solutions to algo_x_3d_solutions.pickle' % min( 100, len(unique)))

        self.save_solutions( unique)

    def save_solutions( self, sols):
        solutions = []
        for idx,s in enumerate( sols):
            # s is a list of row names like 'H_23'
            solution = []
            for rowname in s:
                piece = rowname.split('_')[0]
                filled_holes = [x for x in self.solver.get_col_idxs( rowname)
                                if x < np.prod( self.dims)]
                piece_in_cube = np.full( np.prod( self.dims), 0)
                for h in filled_holes:
                    if '#' in piece:
                        piece = piece.split('#')[0]
                    piece_in_cube[int(h)] = ord(piece) - ord('A') + 1
                piece_in_cube = piece_in_cube.reshape( self.dims)
                solution.append( piece_in_cube)
            solutions.append( solution)
        with open( OUTFILE, 'wb') as f:
            pickle.dump( solutions, f)

    # def find_duplicate_solutions( self):
    #     ' Count frequency of each solution '
    #     dups = {}
    #     for idx,s in enumerate( self.solutions):
    #         hhash = self.hash_solution( s)
    #         elt = {'solution':s, 'idx':idx+1}
    #         if not hhash in dups:
    #             dups[hhash] = [elt]
    #         else:
    #             dups[hhash].append( elt)
    #     return dups

    def hash_solution( self, s):
        grid = self.gridify_solution( s)
        rots = helpers.rotations3D( grid)
        res = ''
        for r in rots:
            r = helpers.number_grid( r)
            hhash = helpers.hhash( repr(r))
            if hhash > res:
                res = hhash
        return res

    def gridify_solution( self, s):
        ' Turn a solution from AlgoX into a 3D array '
        grid = np.full( self.dims[0] * self.dims[1] * self.dims[2], 'xxx')
        # s is a list of rownames like 'L#0_41'
        for rowname in s:
            filled_holes = [x for x in self.solver.get_col_idxs( rowname)
                            if x < self.dims[0] * self.dims[1] * self.dims[2]]
            for h in filled_holes:
                grid[int(h)] = rowname
        grid = grid.reshape( self.dims[0], self.dims[1], self.dims[2])
        return grid


main()
