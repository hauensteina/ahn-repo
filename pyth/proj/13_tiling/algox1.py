#!/usr/bin/env python

from pdb import set_trace as BP

class AlgoX:

    def solve( self):
        self.solve_()
        return self.solutions

    def solve_( self, depth=0):
        #print( 'Depth: %d' % depth)
        #BP()
        # If no column is left, we have a solution
        if len( self.cols) == 0:
            #print( 'Solution!')
            self.solutions.append( self.solution.copy())
            #print( self.solution)
            return
        # Pick the column with the least rows
        min_col_idx = min( self.cols, key = lambda x: len( self.cols[x]))
        min_col = self.cols[min_col_idx]
        # If min column empty, dead end; return
        #if len( min_col) == 0:
            #print( 'Dead End!')
        #    return
        #BP()
        for row in list(min_col): # Tentatively try this row
            #print( 'Trying col,row %d %d' % (min_col_idx, row))
            self.solution.append( self.rownames[row])
            cols = self.remove( row)
            self.solve_( depth+1)
            #print( 'Depth: %d' % depth)
            self.restore( row, cols)
            #BP()
            self.solution.pop()
        #self.cols[min_col_idx] = min_col

    def remove( self, row):
        #print('removing')
        #BP()
        cols = []
        #BP()
        for c in self.rows[row][1:]: # for each hole the row fills
            for r in self.cols[c]: # r also fills hole c
                #print( 'piece %d also fills hole %d' % (r,c))
                for c1 in self.rows[r][1:]: # remove r from all its columns c1
                    if c1 != c:
                        self.cols[c1].discard( r)
                        #print( 'removing piece %d from hole %d' % (r,c1))
            cols.append( self.cols.pop(c))
        return cols

    def restore( self, row, cols):
        #print('restoring')
        #BP()
        for c in reversed( self.rows[row][1:]):
            self.cols[c] = cols.pop()
            for r in self.cols[c]:
                for c1 in self.rows[r][1:]:
                    if c1 != c:
                        self.cols[c1].add( r)

    def get_col_idxs( self, rowname):
        rowidx = self.rownames.index(rowname)
        return self.rows[rowidx][1:]

    def __init__( self, rownames, colnames, entries, max_solutions=0):
        ''' Entries are pairs of (rowidx, colidx) '''
        self.max_solutions = max_solutions
        self.colnames = colnames
        self.rownames = rownames
        self.solution = [] # A solution is a list of row indexes
        self.solutions = [] # A list of solutions
        # Each row is an id, followed by tuple of column indexes. A column is a set of rows.
        self.rows = [[idx] for idx in range( len(self.rownames))]
        # Each column is a set of row indexes.
        self.cols = { idx:set()  for idx in range( len(self.colnames)) }
        # Matrix entries.
        for rowidx,colidx in entries:
            self.rows[rowidx].append( colidx)
            self.cols[colidx].add( rowidx)

def main():
    entries = [
        (0,0),(0,3),(0,6),
        (1,0),(1,3),
        (2,3),(2,4),(2,6),
        (3,2),(3,4),(3,5),
        (4,1),(4,2),(4,5),(4,6),
        (5,1),(5,6),
        (6,1),(6,2),(6,4),(6,5)
    ]
    rownames = ['A','B','C','D','E','F','G']
    colnames = [0,1,2,3,4,5,6]

    solver = AlgoX( rownames, colnames, entries)
    solutions = solver.solve()
    for s in solutions:
        print( s) # Should be ['A', 'G'] and ['B', 'D', 'F']

if __name__ == '__main__':
    main()
