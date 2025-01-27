#!/usr/bin/env python

# Tile a size by size grid with the given pieces, using Knuth's Algorithm X
# AHN, Nov 2020

from pdb import set_trace as BP
import sys,os,json
import argparse
import numpy as np
from algox_assaf import AlgoX
from tiling_helpers import parse_puzzle
import tiling_helpers as helpers

g_pieces = {
    '3x3':
    [
        np.array([
            [1,1]
        ]),
        np.array([
            [1,1,1]
        ]),
        np.array([
            [0,0,1],
            [1,1,1]
        ])
    ],
    '4x4':
    [
        np.array([
            [1,1,1]
        ]),
        np.array([
            [1,0],
            [1,0],
            [1,1]
        ]),
        np.array([
            [1,0],
            [1,0],
            [1,0],
            [1,1]
        ]),
        np.array([
            [1,1,1,1]
        ])
    ],
    '5x5':
    [
        np.array([
            [1,1,1]
        ]),
        np.array([
            [1,1],
            [1,1]
        ]),
        np.array([
            [1,1,1],
            [0,1,1]
        ]),
        np.array([
            [1,1],
            [1,1],
            [0,1]
        ]),
        np.array([
            [1,0],
            [1,0],
            [1,1]
        ]),
        np.array([
            [1,1,0],
            [0,1,1]
        ])
    ],
    '6x6':
    [
        np.array([
            [1,1,1]
        ]),
        np.array([
            [1,1,1],
            [0,1,0]
        ]),
        np.array([
            [1,1,1,1]
        ]),
        np.array([
            [1,1,1],
            [0,1,0],
            [0,1,0]
        ]),
        np.array([
            [0,0,1],
            [1,1,1]
        ]),
        np.array([
            [1,0,0],
            [1,0,0],
            [1,1,1]
        ]),
        np.array([
            [1,1],
            [1,1]
        ]),
        np.array([
            [0,1,1],
            [0,1,1],
            [1,1,1]
        ])
    ],
    '7x7':
    [
        np.array([
            [1,1,1,1]
        ]),
        np.array([
            [1,1,1],
            [1,0,0]
        ]),
        np.array([
            [0,1,1],
            [1,1,1]
        ]),
        np.array([
            [0,1],
            [0,1],
            [1,1]
        ]),
        np.array([
            [0,1],
            [1,1],
            [0,1]
        ]),
        np.array([
            [1,0,0],
            [1,0,0],
            [1,1,1]
        ]),
        np.array([
            [1,1],
            [1,1]
        ]),
        np.array([
            [1,0,0,0],
            [1,1,1,1]
        ]),
        np.array([
            [0,1],
            [0,1],
            [1,1],
            [0,1]
        ]),
        np.array([
            [1,1,1,1,1]
        ]),
        np.array([
            [1],
            [1],
            [1],
            [1]
        ])
    ],
    '7x7a':
    [
        np.array([
            [1]
        ]),
        np.array([
            [1,1],
            [1,0]
        ]),
        np.array([
            [1,1,1]
            ,[1,0,0]
            ,[1,0,0]
        ]),
        np.array([
            [1,1,1,1]
            ,[1,0,0,0]
            ,[1,0,0,0]
            ,[1,0,0,0]
        ]),
        np.array([
            [1,1,1,1,1]
            ,[1,0,0,0,0]
            ,[1,0,0,0,0]
            ,[1,0,0,0,0]
            ,[1,0,0,0,0]
        ]),
        np.array([
            [1,1,1,1,1,1]
            ,[1,0,0,0,0,0]
            ,[1,0,0,0,0,0]
            ,[1,0,0,0,0,0]
            ,[1,0,0,0,0,0]
            ,[1,0,0,0,0,0]
        ]),
        np.array([
            [1,1,1,1,1,1,1]
            ,[1,0,0,0,0,0,0]
            ,[1,0,0,0,0,0,0]
            ,[1,0,0,0,0,0,0]
            ,[1,0,0,0,0,0,0]
            ,[1,0,0,0,0,0,0]
            ,[1,0,0,0,0,0,0]
        ])
    ],
    '7x7b':
    [
        np.array([
            [1,1,1]
            ,[1,1,0]
        ]),
        np.array([
            [1,1,1]
            ,[0,1,0]
        ]),
        np.array([
            [1,1],
            [1,1]
        ]),
        np.array([
            [1,1,1]
            ,[0,0,1]
            ,[0,0,1]
        ]),
        np.array([
            [1,1,1]
            ,[1,1,1]
            ,[1,0,0]
        ]),
        np.array([
            [1,1,1]
            ,[0,0,1]
        ]),
        np.array([
            [1,1,1]
            ,[1,1,1]
            ,[0,1,0]
        ]),
        np.array([
            [1]
            ,[1]
        ]),
        np.array([
            [1,1,1]
            ,[1,1,1]
        ]),
        np.array([
            [1,0,1]
            ,[1,1,1]
        ])
    ],
    '7x7c':
    [
        np.array([
            [1,1,1,1]
            ,[1,0,0,0]
        ]),
        np.array([
            [1,1,1]
            ,[1,0,0]
        ]),
        np.array([
            [1,1,1]
        ]),
        np.array([
            [0,1,1,1]
            ,[1,1,1,0]
        ]),
        np.array([
            [0,1]
            ,[1,1]
            ,[1,0]
        ]),
        np.array([
            [1,1,1]
            ,[0,1,0]
        ]),
        np.array([
            [1,0,1]
            ,[1,1,1]
            ,[0,0,1]
        ]),
        np.array([
            [0,1,1,1]
            ,[1,1,0,0]
            ,[1,0,0,0]
            ,[1,0,0,0]
        ]),
        np.array([
            [1,1]
        ]),
        np.array([
            [0,1,1]
            ,[1,1,0]
            ,[1,0,0]
        ]),
        np.array([
            [0,1]
            ,[1,1]
        ]),
    ],
    '8x8':
    [
        np.array([
            [1,1,1,1,1]
        ]),
        np.array([
            [1,1,1],
            [0,1,1]
        ]),
        np.array([
            [1,1,0,0],
            [0,1,1,1]
        ]),
        np.array([
            [1,1,1,1],
            [0,0,1,0]
        ]),
        np.array([
            [1,1,1],
            [0,1,0],
            [0,1,0]
        ]),
        np.array([
            [1,0,0,0],
            [1,1,1,1]
        ]),
        np.array([
            [0,0,1,1],
            [1,1,1,0]
        ]),
        np.array([
            [1,1],
            [1,0],
            [1,1]
        ]),
        np.array([
            [0,1,1],
            [0,0,1],
            [1,1,1]
        ]),
        np.array([
            [0,0,1],
            [1,1,1],
            [1,0,0],
            [1,0,0]
        ]),
        np.array([
            [1],
            [1],
            [1]
        ]),
        np.array([
            [0,0,1,0],
            [1,1,1,1]
        ]),
        np.array([
            [1,1,1,1]
        ])
    ],
    'pentomino':
    [
        # The Center Square
        np.array([
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0]
        ]),
        np.array([
            [1,1,1,1,1]
        ]),
        np.array([
            [1,1,1,1],
            [0,0,0,1]
        ]),
        np.array([
            [0,1,1,1],
            [1,1,0,0]
        ]),
        np.array([
            [1,0,1],
            [1,1,1]
        ]),
        np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ]),
        np.array([
            [1,1,0],
            [0,1,1],
            [0,0,1]
        ]),
        np.array([
            [1,1,1],
            [0,1,1]
        ]),
        np.array([
            [0,0,1],
            [1,1,1],
            [0,1,0]
        ]),
        np.array([
            [0,1,1],
            [0,1,0],
            [1,1,0]
        ]),
        np.array([
            [1,0,0],
            [1,1,1],
            [1,0,0]
        ]),
        np.array([
            [0,0,1,0],
            [1,1,1,1]
        ]),
        np.array([
            [0,0,1],
            [0,0,1],
            [1,1,1]
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
      %s: Solve 2D nxn tiling puzzles.
    Synopsis:
      %s --case <case_id> [--print] [--mode basic|nopieces]
      %s --json <file> [--print]
      %s --test
    Description:
      --mode nopieces
      Run without adding columns for the pieces.
      This only works if one piece repeats many times.
      --print
      Print each solution immediately when found.
      At the end, solutions are printed even without --print.
    Examples:
      %s --case 6x6 --print
      %s --json simple.json --print

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
    parser.add_argument( "--test", action='store_true')
    parser.add_argument( "--print", action='store_true')
    parser.add_argument( "--mode", choices=['basic','nopieces'], default='basic')
    args = parser.parse_args()

    if args.test:
        unittest()

    if not args.case and not args.json:
        usage( True)
    if args.case:
        solver = AlgoX2D( g_pieces[args.case])
    else:
        pieces, piece_names, piece_counts, dims, one_sided = parse_puzzle( args.json)
        solver = AlgoX2D( pieces, piece_names, piece_counts, dims, one_sided, mode=args.mode)
    solver.solve( args.print)

#----------------
def unittest():
    solver = AlgoX2D( g_pieces['5x5'])
    solver.solve( print_flag=False)
    if len(solver.solutions) == 74:
        print( 'Unit test passed')
    else:
        print( 'Unit test failed: Found %d solutions, should be 74' % len(solver.solutions))
    exit(1)

#=================================================================================
class AlgoX2D:
    '''
    Knuth's Algorithm X for 2D tiling puzzles.
    We don't do the dancing links (DLX), just use dicts of sets.
    A shifted rotated instance of a piece is called an image.
    '''
    def __init__( self, pieces, piece_names=None, piece_counts=None, dims=None, one_sided=False, mode='basic'):
        '''
        Build a matrix with a column per gridpoint plus a column per piece.
        The rows are the images (rot + trans) of the pieces that fit in the grid.
        '''
        self.dims = dims
        self.mode = mode
        if dims is None:
            self.nholes = sum( [np.sum( np.sign(p)) for p in pieces])
            size = int(np.sqrt( self.nholes))
            self.dims = (size, size)

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
            print( 'ERROR: Pieces do not cover grid: %d of %d covered' % (self.nholes, self.nholes))
            exit(1)

        rownames = []

        colnames = [str(x) for x in range( self.nholes)]
        colcounts = [1] * len(colnames)
        if self.mode != 'nopieces':
            colnames += piece_names
            colcounts += piece_counts
        entries = set() # Pairs (rowidx, colidx)

        def add_image( piece_id, img_id, img, row, col):
            ' Add an image to the set of images to try '
            rowname = piece_id + '_' + str(img_id) # A#0_1
            rownames.append( rowname)
            rowidx = len( rownames) - 1
            if self.mode != 'nopieces':
                entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
            grid = np.zeros( (self.dims[0], self.dims[1]))
            helpers.add_window_2D( grid, img, row, col)
            filled_holes = set( np.flatnonzero( grid))
            for h in filled_holes: # Image fills these holes
                colidx = colnames.index( str(h))
                entries.add( (rowidx, colidx) )

        # Add all images
        worst_piece_idx = self.get_worst_piece_idx( pieces, piece_counts) # freeze this piece into one symmetry
        if self.mode == 'nopieces': worst_piece_idx = -1

        for pidx,p in enumerate( pieces):
            if pidx == worst_piece_idx:
                rots = helpers.one_rot_per_shape_2D( p, self.dims)
            else:
                rots = helpers.rotations2D( p, one_sided)

            piece_id = piece_names[pidx]
            img_id = -1
            for rotidx,img in enumerate( rots):
                for row in range( self.dims[0] - img.shape[0] + 1):
                    for col in range( self.dims[1] - img.shape[1] + 1):
                        img_id += 1
                        add_image( piece_id, img_id, img, row, col)
        self.solver = AlgoX( rownames, colnames, colcounts, entries)

    def get_worst_piece_idx( self, pieces, piece_counts):
        '''
        Find the piece with most symmetries and least positions.
        '''
        maxrots = residx = -1
        minpositions = int(1E9)
        for pidx,p in enumerate( pieces):
            # Do not freeze a piece if it has several instances
            if piece_counts[pidx] > 1: continue
            rots = helpers.rotations2D( p)
            if len(rots) <= maxrots: continue
            if len(rots) > maxrots: minpositions = int(1E9)
            maxrots = len(rots)
            npositions = (self.dims[0] - p.shape[0] + 1) * (self.dims[1] - p.shape[1] + 1)
            if npositions >= minpositions: continue
            minpositions = npositions
            residx = pidx
        return residx

    def solve( self, print_flag):
        self.solutions = []
        for idx, s in enumerate( self.solver.solve()):
            if print_flag:
                self.print_solution( idx, s)
            self.solutions.append(s)

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
        print( 'Printing %d unique solutions' % min( 100, len(unique)))
        self.print_solutions( unique)

    def print_solutions( self, sols=None):
        if not sols: sols = self.solutions
        for idx,s in enumerate( sols):
            self.print_solution( idx, s)

    def print_solution( self, idx, s):
        pic = np.full( self.dims[0] * self.dims[1], ' ' * 10)
        print()
        print( 'Solution %d:' % (idx+1))
        print( '=============')
        # s is a list of row names like 'L#0_41'
        for rowname in s:

            filled_holes = [x for x in self.solver.get_col_idxs( rowname)
                            if x < self.dims[0] * self.dims[1]]
            for h in filled_holes:
                pic[int(h)] = rowname
        pic = pic.reshape( self.dims[0], self.dims[1])
        for r in range( self.dims[0]):
            for c in range( self.dims[1]):
                helpers.print_colored_letter( pic[r,c])
            print()
        print()

    def gridify_solution( self, s):
        ' Turn a solution from AlgoX into a 2D array '
        grid = np.full( self.dims[0] * self.dims[1], ' ' * 10)
        # s is a list of row names like 'L#0_41'
        for rowname in s:
            filled_holes = [x for x in self.solver.get_col_idxs( rowname)
                            if x < self.dims[0] * self.dims[1]]
            for h in filled_holes:
                grid[int(h)] = rowname
        grid = grid.reshape( self.dims[0], self.dims[1])
        return grid

    def hash_solution( self, s):
        grid = self.gridify_solution( s)
        rots = helpers.rotations2D( grid)
        res = ''
        for r in rots:
            r = helpers.number_grid( r)
            hhash = helpers.hhash( repr(r))
            if hhash > res:
                res = hhash
        return res

main()
