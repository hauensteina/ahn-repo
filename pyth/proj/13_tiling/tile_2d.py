# Tile a size by size grid with the given pieces
# AHN, Nov 2020

from pdb import set_trace as BP
import numpy as np
#import cupy as np

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

#===================
class Solution:
    ' A solution is a hashable grid '
    def __init__( self, grid):
        self.grid = grid

    def __hash__(self):
        ' Hash and take out symmetries '
        maxhash = None
        for rs in rotations( self.grid):
            h = hash( repr(rs))
            if maxhash is None or maxhash < h:
                maxhash = h
        return maxhash

    def __eq__(self, other):
        ' Compare and take out symmetries '
        rots = rotations( self.grid)
        for rs in rots:
            if repr(rs) == repr(other.grid):
                return True
        return False

    def size( self):
        ' Grid size without n-1 padding '
        return int( (self.grid.shape[0] + 2) / 3)

    def pr( self):
        print()
        for row in range( self.size()):
            for col in range( self.size()):
                r,c = padrc( self.size(), row, col)
                print( '%d' % self.grid[r,c], end='')
            print()

#-------------
def main():
    pieces = g_pieces['5x5']
    size2 = sum( [np.sum( np.sign(p)) for p in pieces])
    size = int(np.sqrt( size2))
    pad = size-1
    padsize = size + 2 * pad # pad both sides
    grid = np.zeros( (padsize, padsize))
    solve( size, grid, pieces)
    print( 'Inner loops: %d' % solve.loopcount)
    print( 'Found a total of %d solutions.' % len(solve.solutions))
    for s in solve.solutions:
        s.pr()

#---------------------------------------------------------
def solve( size, grid, pieces, depth=0, row_=0, col_=0):
    ' Tile the grid with the pieces '
    if depth == 0: solve.progress = np.zeros( len(pieces))
    #if (depth == size) and (row_ + col_ == 0):
        #print( 'depth: %d %s' % (depth, str(solve.progress)))
    try:
        if len(pieces) == 0:
            #BP()
            #print( solve.progress)
            solve.solutions.add( Solution(grid))
            #print( solve.progress)
            print( '>>>>>>>>> New solution found after %d loops' % solve.loopcount)
            return 'success'
        p = pieces[0]
        rots = rotations( p)
        for ridx,rp in enumerate( rots):
            solve.progress[depth] = ridx
            for row in range( size - rp.shape[0] + 1):
                for col in range( size - rp.shape[1] + 1):
                    #if (row + col == 0) and (depth+1 == len(pieces)):
                    #    print( 'xx ' + str(solve.progress))
                    indent = '  ' * depth
                    if depth < 2: #int(np.log(size*size)):
                        print( '%spiece:%d rot:%d/%d row:%d col:%d' % (indent, depth+1, ridx+1, len(rots), row, col))
                    solve.loopcount += 1
                    g = grid.copy()
                    r,c = padrc( size, row, col)
                    add_window( g, rp, r, c)
                    if islegal( size, g, depth+1):
                        #print( 'placed piece %d at %d,%d' % (pidx+depth+1, row, col))
                        #print( 'placing piece %d at %d %d with rotation %d' % (depth, row, col, ridx))
                        remaining_pieces = pieces[1:]
                        solve( size, g, remaining_pieces, depth+1, row, col)
                    else:
                        pass
                        #print( 'piece %d illegal at %d,%d' % (depth+1, row, col))
        #print( 'dead end')
        return 'failure'
    except Exception as e:
        BP()
        tt=42

solve.loopcount = 0
solve.solutions = set()
solve.progress = None

#-------------------------
def padrc( size, r, c):
    ' Shift row, col into padded grid '
    return r + size-1, c + size-1

#------------------------------------
def add_window( arr, window, r, c):
    ' Add window to arr at position r,c for left upper corner of win in arr '
    arr[ r:r+window.shape[0], c:c+window.shape[1] ] += window

#-------------------------
def rotations( grid):
    ' Return a list with all 8 rotations/mirrors of a grid '
    h = grid.shape[0]
    w = grid.shape[1]
    res = []
    strs = set()

    for i in range(4):
        if not repr(grid) in strs: res.append( grid)
        strs.add(repr(grid))
        grid = rot( grid)
    grid = mirror( grid)
    for i in range(4):
        if not repr(grid) in strs: res.append( grid)
        strs.add(repr(grid))
        grid = rot( grid)
    return res

#---------------------------------------
def islegal( size, g, cur_piece_num):
    '''
    A grid is legal if the inner sum equals the total sum (no protrusions)
    and the maximum is less than the current piece number (no overlap)
    '''
    pad = size - 1
    ssum = np.sum(g)
    innersum = np.sum( g[ pad : pad + size, pad : pad + size ])
    if ssum != innersum: return False
    if np.max(g) > cur_piece_num: return False
    return True

#-------------------------
def rot( grid):
    ' Rotate a 2d grid clockwise '
    return np.rot90( grid,1,(1,0))

#-------------------------
def mirror( grid):
    ' Mirrors a 2d grid left to right'
    return np.flip( grid,1)

main()
