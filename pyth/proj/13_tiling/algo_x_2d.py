#!/usr/bin/env python

# Tile a size by size grid with the given pieces, using Knuth's Algorithm X
# AHN, Nov 2020

from pdb import set_trace as BP
import sys,os
import argparse
import numpy as np
from algox import AlgoX

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
      %s: Solve 2D nxn tiling puzzles.
    Synopsis:
      %s --n <size>
      %s --test
    Example:
      %s --n 5

--
''' % (name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg


def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--n", type=int, choices=[3,4,5,6,7])
    parser.add_argument( "--test", action='store_true')
    args = parser.parse_args()

    if args.test:
        unittest()

    if not args.n:
        usage( True)

    solver = AlgoX2D( g_pieces['' + str(args.n) + 'x' + str(args.n)])
    solver.solve()
    solver.print_solutions()

def unittest():
    solver = AlgoX2D( g_pieces['5x5'])
    solver.solve()
    if len(solver.solver.solutions) == 74:
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
    def __init__( self, pieces):
        '''
        Build a matrix with a column per gridpoint plus a column per piece.
        The rows are the images (rot + trans) of the pieces that fit in the grid.
        '''
        self.nholes = sum( [np.sum( np.sign(p)) for p in pieces])
        self.size = int(np.sqrt( self.nholes))

        rownames = []
        piece_ids = [chr( ord('A') + x) for x in range( len(pieces))]
        colnames = [str(x) for x in range( self.nholes)] + piece_ids
        entries = set() # Pairs (rowidx, colidx)

        worst_piece_idx = self.get_worst_piece_idx( pieces) # most symmetries, positions

        for pidx,p in enumerate( pieces):
            rots = AlgoX2D.rotations2D( p)
            piece_id = piece_ids[pidx]
            img_id = -1
            for rotidx,img in enumerate( rots):
                if pidx == worst_piece_idx and rotidx > 0: # Restrict worst piece to eliminate syms
                    break
                for row in range( self.size - img.shape[0] + 1):
                    for col in range( self.size - img.shape[1] + 1):
                        img_id += 1
                        rowname = piece_id + '_' + str(img_id)
                        rownames.append( rowname)
                        rowidx = len( rownames) - 1
                        entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
                        grid = np.zeros( (self.size, self.size))
                        AlgoX2D.add_window( grid, img, row, col)
                        filled_holes = set( np.flatnonzero( grid))
                        for h in filled_holes: # Image fills these holes
                            colidx = colnames.index( str(h))
                            entries.add( (rowidx, colidx) )

        self.solver = AlgoX( rownames, colnames, entries)

    def get_worst_piece_idx( self, pieces):
        ' Find the piece with most symmetries,positions '
        maxpositions = maxrots = maxidx = -1
        for pidx,p in enumerate( pieces):
            rots = AlgoX2D.rotations2D( p)
            if len(rots) <= maxrots: continue
            if len(rots) > maxrots: maxpositions = 0
            maxrots = len(rots)
            npositions = (self.size - p.shape[0] + 1) * (self.size - p.shape[1] + 1)
            if npositions <= maxpositions: continue
            maxpositions = npositions
            maxidx = pidx
        return maxidx

    def solve( self):
        self.solver.solve()

    def print_solutions( self):
        for idx,s in enumerate( self.solver.solutions):
            pic = np.full( self.size * self.size, 'A')
            print()
            print( 'Solution %d:' % (idx+1))
            print( '=============')
            # s is a list of row headers
            for row in s:
                es = row.entries
                filled_holes = [ x.colheader.name for x in row.entries if AlgoX2D.isnumeric (x.colheader.name) ]
                piece = [ x.colheader.name for x in row.entries if not AlgoX2D.isnumeric (x.colheader.name) ][0]
                for h in filled_holes:
                    pic[int(h)] = piece
            pic = pic.reshape( self.size, self.size)
            for r in range( self.size):
                for c in range( self.size):
                    print( pic[r,c] + ' ', end='')
                print()

    @staticmethod
    def rotations2D(grid):
        ' Return a list with all 8 rotations/mirrors of a grid '
        h = grid.shape[0]
        w = grid.shape[1]
        res = []
        strs = set()

        for i in range(4):
            if not repr(grid) in strs: res.append( grid)
            strs.add(repr(grid))
            grid = AlgoX2D.rot( grid)
        grid = AlgoX2D.mirror( grid)
        for i in range(4):
            if not repr(grid) in strs: res.append( grid)
            strs.add(repr(grid))
            grid = AlgoX2D.rot( grid)
        return res

    @staticmethod
    def add_window( arr, window, r, c):
        ' Add window to arr at position r,c for left upper corner of win in arr '
        arr[ r:r+window.shape[0], c:c+window.shape[1] ] += window

    @staticmethod
    def rot( grid):
        ' Rotate a 2d grid clockwise '
        return np.rot90( grid,1,(1,0))

    @staticmethod
    def mirror( grid):
        ' Mirrors a 2d grid left to right'
        return np.flip( grid,1)

    @staticmethod
    def isnumeric(s):
        try:
            res = float( s)
            return True
        except ValueError:
            return False

main()
