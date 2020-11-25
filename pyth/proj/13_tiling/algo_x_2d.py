# Tile a size by size grid with the given pieces, using Knuth's Algorithm X
# AHN, Nov 2020

from pdb import set_trace as BP
import numpy as np

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

def main():
    solver = AlgoX2D( g_pieces['3x3'])
    solver.solve()

class CompleteCoverage:
    ''' Knuth's Algorithm X '''

    class Entry:
        ''' A hashable X matrix element '''
        def __init__( self, rname, cname, rowheader, colheader):
            self.hhash = hash(rname + '#' + cname)
            self.rowheader = rowheader
            self.colheader = colheader
        def __hash__( self):
            return self.hhash
        def __eq__( self, other):
            self.hhash == other.hhash
        def __str__( self):
            print( self.hhash)

    class Header:
        ''' A hashable X matrix col or row header '''
        def __init__( self, name):
            self.name = name
            self.entries = set()
            self.hhash = hash(name)
        def __hash__( self):
            return self.hhash
        def __eq__( self, other):
            self.name == other.name
        def __str__( self):
            print( self.name)
            print( self.entries)

    def __init__( self, rownames, colnames, entries):
        ''' Entries are pairs of (rowidx, colidx) '''

        self.colnames = colnames
        self.solution = []
        # Row headers
        self.rows = {}
        for rname in rownames:
            self.rows[rname] = CompleteCoverage.Header( rname)
        # Column headers
        self.cols = {}
        for cname in colnames:
            self.cols[cname] = CompleteCoverage.Header( cname)

        # Matrix entries
        for rowidx,colidx in entries:
            rname = rownames[rowidx]
            cname = colnames[colidx]
            entry = CompleteCoverage.Entry( rname, cname, self.rows, self.cols)
            self.rows[rname].entries.add( entry)
            self.cols[cname].entries.add( entry)

    def remove_col( self, colheader):
        '''
        Remove a column.
        '''
        # Remove column from all its rows; @@@ is this needed?
        for e in colheader.entries:
            e.rowheader.entries.remove(e)

        # Remove the col itself
        self.cols.remove( colheader)
        return colheader

    def restore_col( self, colheader):
        ''' Put the column back '''
        self.cols.add( colheader)
        # Add each col entry to its row @@@ Is this needed?
        for e in colheader.entries:
            e.rowheader.entries.add(e)

    def remove_row( self, rowheader):
        '''
        Remove a row.
        '''
        # Remove row from all its cols
        for e in rowheader.entries:
            e.colheader.entries.remove(e)
        # Remove the row itself
        self.rows.remove( rowheader)
        return rowheader

    def restore_row( self, rowheader):
        ''' Put the row back '''
        self.rows.add( rowheader)
        # Add each row entry to its col
        for e in rowheader.entries:
            e.colheader.entries.add(e)

    def restore( self, rowheaders, colheaders):
        ''' Backtrack '''
        # Restore rows
        for r in rowheaders:
            self.restore_row( r)
        # Restore columns
        for c in colheaders:
            sef.restore_col( c)

    def get_overlappers( self, rowheader):
        ''' Find all row headers which overlap this row '''
        for e in rowheader.entries:
            for colentry in e.colheader.entries:
                res.add( colentry.rowheader)
        return res

    def pick_col( self):
        '''
        Pick a column to fill next.
        We choose the column with fewest elements.
        Returns the column header.
        This convoluted implementation is the fastest.
        '''
        counts = [ len(x.entries) for x in self.cols.values() ]
        mmin = min(counts)
        keys = list(self.cols.keys())
        k = keys[ counts.index(mmin)]
        return self.cols[k]

    def row2list( self, rowheader):
        '''
        Return a list of 0 or 1, ordered by colnames
        '''
        cnames = set( [x.colheader.name  for x in rowheader.entries])
        res = []
        for cname in self.colnames:
            if cname in cnames: res.append(1)
            else: res.append(0)
        return res

    def print_solution( self):
        for rowheader in self.solution:
            print( self.row2list( rowheader))

    def solve( self):
        ''' Run Algorithm X '''
        colheader = self.pick_col() # A hole to cover or a piece to place
        for rowentry in colheader.entries: # for images covering this hole
            self.solution.append( rowentry.rowheader)
            rem_rows = set()
            rem_cols = set()
            rowheader = rowentry.rowheader
            # Remove all columns we cover
            for e in rowheader.entries:
                rem_cols.add( self.remove_col( e.colheader))
            # Remove all overlapping rows
            orows = self.get_overlappers( rowheader)
            for r in orows:
                rem_rows.add( self.remove_row( r))
            # If no cols are left, we are done
            if len(cols) == 0:
                self.print_solution()
                self.restore( rem_rows, rem_cols)
                self.solution.pop()
                continue # Look for more solutions
            # If any column has no more entries, we are stuck
            for chead,cname in enumerate(self.cols):
                if len(chead.entries) == 0:
                    print( 'Dead End')
                    self.restore( rem_rows, rem_cols)
                    self.solution.pop()
                    continue
            # Alright, so we filled a hole. Now fill another one.
            self.solve()

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

        for pidx,p in enumerate( pieces):
            rots = AlgoX2D.rotations2D( p)
            piece_id = piece_ids[pidx]
            img_id = -1
            for img in rots:
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

        self.solver = CompleteCoverage( rownames, colnames, entries)

    def solve( self):
        self.solver.solve()

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

# #===================
# class Solution:
#     ' A solution is a hashable grid '
#     def __init__( self, grid):
#         self.grid = grid

#     def __hash__(self):
#         ' Hash and take out symmetries '
#         maxhash = None
#         for rs in rotations( self.grid):
#             h = hash( repr(rs))
#             if maxhash is None or maxhash < h:
#                 maxhash = h
#         return maxhash

#     def __eq__(self, other):
#         ' Compare and take out symmetries '
#         rots = rotations( self.grid)
#         for rs in rots:
#             if repr(rs) == repr(other.grid):
#                 return True
#         return False

#     def size( self):
#         ' Grid size without n-1 padding '
#         return int( (self.grid.shape[0] + 2) / 3)

#     def pr( self):
#         print()
#         for row in range( self.size()):
#             for col in range( self.size()):
#                 r,c = padrc( self.size(), row, col)
#                 print( '%d' % self.grid[r,c], end='')
#             print()

main()
