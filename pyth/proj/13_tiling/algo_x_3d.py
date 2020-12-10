#!/usr/bin/env python

# Tile a s^3 cube with the given pieces, using Knuth's Algorithm X
# AHN, Nov 2020

from pdb import set_trace as BP
import sys,os,pickle
import argparse
import numpy as np
from algox import AlgoX

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
    Example:
      %s --case 3x3x3

--
''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------------
def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--case", required=True)
    parser.add_argument( "--max_solutions", type=int)
    args = parser.parse_args()

    solver = AlgoX3D( g_pieces[args.case], args.max_solutions)
    solver.solve()
    print( '\nFound %d solutions' % len( solver.solver.solutions))

#=================================================================================
class AlgoX3D:
    '''
    Knuth's Algorithm X for 3D tiling puzzles.
    We don't do the dancing links (DLX), just use dicts of sets.
    A shifted rotated instance of a piece is called an image.
    '''
    def __init__( self, pieces, max_solutions=0):
        '''
        Build a matrix with a column per gridpoint plus a column per piece.
        The rows are the images (rot + trans) of the pieces that fit in the grid.
        '''
        self.nholes = sum( [np.sum( np.sign(p)) for p in pieces])
        self.size = int(np.cbrt( self.nholes))

        rownames = []
        piece_ids = [chr( ord('A') + x) for x in range( len(pieces))]

        colnames = [str(x) for x in range( self.nholes)] + piece_ids
        entries = set() # Pairs (rowidx, colidx)

        def add_image( piece_id, img_id, img, row, col, layer):
            '''
            Add an image to the set of images to try.
            We call the third dimension *layer*
            '''
            rowname = piece_id + '_' + str(img_id)
            rownames.append( rowname)
            rowidx = len( rownames) - 1
            entries.add( (rowidx, colnames.index(piece_id))) # Image is instance of this piece
            cube = np.zeros( (self.size, self.size, self.size))
            AlgoX3D.add_window( cube, img, row, col, layer)
            filled_holes = set( np.flatnonzero( cube))
            for h in filled_holes: # Image fills these holes
                colidx = colnames.index( str(h))
                entries.add( (rowidx, colidx) )

        # Add all images
        worst_piece_idx = self.get_worst_piece_idx( pieces) # most symmetries, positions
        for pidx,p in enumerate( pieces):
            rots = AlgoX3D.rotations3D( p)
            print( len(rots))
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
        self.solver = AlgoX( rownames, colnames, entries, max_solutions)

    def get_worst_piece_idx( self, pieces):
        '''
        Find the piece with most symmetries and least positions.
        '''
        maxrots = residx = -1
        minpositions = int(1E9)
        for pidx,p in enumerate( pieces):
            rots = AlgoX3D.rotations3D( p)
            if len(rots) <= maxrots: continue
            if len(rots) > maxrots: minpositions = int(1E9)
            maxrots = len(rots)
            npositions = (self.size - p.shape[0] + 1) * (self.size - p.shape[1] + 1)* (self.size - p.shape[2] + 1)
            if npositions >= minpositions: continue
            minpositions = npositions
            residx = pidx
        return residx

    def solve( self):
        self.solver.solve()
        self.save_solutions()

    def save_solutions( self):
        solutions = []
        for idx,s in enumerate( self.solver.solutions):
            # s is a list of row headers
            solution = []
            for row in s:
                es = row.entries
                filled_holes = [ x.colheader.name for x in row.entries if AlgoX3D.isnumeric (x.colheader.name) ]
                piece = [ x.colheader.name for x in row.entries if not AlgoX3D.isnumeric (x.colheader.name) ][0]
                piece_in_cube = np.full( self.size * self.size * self.size, 0)
                for h in filled_holes:
                    piece_in_cube[int(h)] = ord(piece) - ord('A') + 1
                piece_in_cube = piece_in_cube.reshape( self.size, self.size, self.size)
                solution.append( piece_in_cube)
            solutions.append( solution)
        with open( OUTFILE, 'wb') as f:
            pickle.dump( solutions, f)

        with open( OUTFILE, 'rb') as f:
            xx = pickle.load( f)


    # def print_solutions( self):
    #     for idx,s in enumerate( self.solver.solutions):
    #         pic = np.full( self.size * self.size, 'A')
    #         print()
    #         print( 'Solution %d:' % (idx+1))
    #         print( '=============')
    #         # s is a list of row headers
    #         for row in s:
    #             es = row.entries
    #             filled_holes = [ x.colheader.name for x in row.entries if AlgoX2D.isnumeric (x.colheader.name) ]
    #             piece = [ x.colheader.name for x in row.entries if not AlgoX2D.isnumeric (x.colheader.name) ][0]
    #             for h in filled_holes:
    #                 pic[int(h)] = piece
    #         pic = pic.reshape( self.size, self.size)
    #         for r in range( self.size):
    #             for c in range( self.size):
    #                 AlgoX2D.print_colored_letter( pic[r,c])
    #             print()

    # @staticmethod
    # def print_colored_letter( letter):
    #     ' Print a letter. Color depends on what letter it is. '
    #     color = ord(letter) - ord('A')
    #     color %= 16
    #     print( '\x1b[48;5;%dm%s \x1b[0m' % (color, letter), end='')

    @staticmethod
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
                brick = AlgoX3D.rot_top( brick)
            # Change top
            if i < 3:
                #print('>>>>>>>>>>>>>> < 3')
                brick = AlgoX3D.rot_front( brick)
            elif i == 3: # bring back to top
                #print('############## == 3')
                brick = AlgoX3D.rot_front( brick) # initial orientation
                brick = AlgoX3D.rot_left( brick)
            elif i == 4: # bring front to top
                #print('$$$$$$$$$$$$$ == 4')
                brick = AlgoX3D.rot_left( brick)
                brick = AlgoX3D.rot_left( brick)

        return res

    @staticmethod
    def add_window( arr, window, r, c, l):
        ' Add window to arr at position r,c,l for left upper corner of win in arr '
        arr[ r:r+window.shape[0], c:c+window.shape[1], l:l+window.shape[2] ] += window

    @staticmethod
    def rot_top( brick):
        ' Rotate a brick clockwise viewed from top '
        return np.rot90( brick,1,(2,0))

    @staticmethod
    def rot_front( brick):
        ' Rotate a brick clockwise viewed from front '
        return np.rot90( brick,1,(2,1))

    @staticmethod
    def rot_left( brick):
        ' Rotate a brick clockwise viewed from left '
        return np.rot90( brick,1,(0,1))

    @staticmethod
    def isnumeric(s):
        try:
            res = float( s)
            return True
        except ValueError:
            return False

main()
