# Tile a size by size grid with the given pieces
# AHN, Nov 2020

from pdb import set_trace as BP
import numpy as np

g_size = 3 # size x size grid
g_marg = g_size-1 # Add some margin
g_totsize = g_size + 2 * g_marg # pad both sides

g_pieces = [
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
]

g_pieces = [p * (idx + 1) for idx,p in enumerate( g_pieces)]

def main():
    grid = np.zeros( (g_totsize, g_totsize))
    res = solve( grid, g_pieces)
    #print( res)
    print( 'Inner loops: %d' % solve.loopcount)
    print( 'Found a total of %d solutions.' % solve.nsolutions)
    for s in solve.solutions:
        display_solution(s)

def solve( grid, pieces, depth=0):
    ' Tile the grid with the pieces '
    if len(pieces) == 0:
        sol_hash = hash_solution( grid)
        if sol_hash not in solve.solution_hashes:
            solve.solution_hashes.add( sol_hash)
            solve.solutions.append( grid)
            solve.nsolutions += 1
            #print( '>>>>>>>>> New solution found after %d loops' % solve.loopcount)
        return 'success'
    for pidx, p in enumerate( pieces):
        for rp in rotations( p):
            for row in range( g_size - rp.shape[0] + 1):
                for col in range( g_size - rp.shape[1] + 1):
                    solve.loopcount += 1
                    g = grid.copy()
                    r,c = totrc( row, col)
                    add_window( g, rp, r, c)
                    if islegal(g, pidx+depth+1):
                        #print( 'placed piece %d at %d,%d' % (pidx+depth+1, row, col))
                        remaining_pieces = pieces[1:]
                        solve( g, remaining_pieces, depth+1)
                    else:
                        #print( 'piece %d illegal at %d,%d' % (pidx+depth+1, row, col))
                        pass

        break # if one piece cannot be placed, no use trying other
    #print( 'dead end')
    return 'failure'
solve.loopcount = 0
solve.nsolutions = 0
solve.solutions = []
solve.solution_hashes = set()

def display_solution(s):
    ' Show solution on screen '
    print()
    for row in range(g_size):
        for col in range(g_size):
            r,c = totrc( row, col)
            print( '%d' % s[r,c], end='')
        print()

def hash_solution(s):
    ' Hash a solution, eliminating symmetries '
    maxhash = None
    for rs in rotations( s):
        h = hash( repr(rs))
        if maxhash is None or maxhash < h:
            maxhash = h
    return maxhash

def totrc( r, c):
    ' Shift row, col into totsize grid with margin '
    return r + g_marg, c + g_marg

def add_window( arr, window, r, c):
    ' Add window to arr at position r,c for left upper corner of win in arr '
    arr[ r:r+window.shape[0], c:c+window.shape[1] ] += window

def rotations( piece):
    ' Return a list with all 8 rotations/mirrors of a piece '
    h = piece.shape[0]
    w = piece.shape[1]
    res = []
    strs = set()

    for i in range(4):
        if not repr(piece) in strs: res.append( piece)
        strs.add(repr(piece))
        piece = rot( piece)
    piece = mirror( piece)
    for i in range(4):
        if not repr(piece) in strs: res.append( piece)
        strs.add(repr(piece))
        piece = rot( piece)
    return res

def islegal( g, cur_piece_num):
    '''
    A grid is legal if the inner sum equals the total sum (no protrusions)
    and the maximum is less than the current piece number (no overlap)
    '''
    ssum = np.sum(g)
    innersum = np.sum( g[ g_marg : g_marg + g_size, g_marg : g_marg + g_size ])
    if ssum != innersum: return False
    if np.max(g) > cur_piece_num: return False
    return True

def rot( piece):
    ' Rotate a piece clockwise '
    return np.rot90( piece,1,(1,0))

def mirror( piece):
    ' Mirrors a piece left to right'
    return np.flip( piece,1)

def test_add_win( grid):
    win = np.array( [
        [1,1],
        [1,1]
    ])
    add_win( grid, win, 1, 2)
    print( grid)

main()
