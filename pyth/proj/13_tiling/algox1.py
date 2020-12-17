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
            return
        # Pick the column with the least rows
        min_col_idx = min( self.cols, key = lambda x: len( self.cols[x]))
        # If min column empty, dead end; return
        if len( self.cols[min_col_idx]) == 0:
            #print( 'Dead End!')
            return
        # Remove mmin_col from the list of columns.
        min_col = self.cols.pop( min_col_idx)

        for row in min_col: # Tentatively try this row
            #print( 'Trying col,row %d %d' % (min_col_idx, row[0]))
            self.solution.append( self.rownames[row[0]])
            # Remove overlapping rows
            removed_rows = set()
            removed_cols = { idx:self.cols[idx] for idx in row[1:] if idx in self.cols }
            # For each column in row
            #BP()
            for col_idx in row[1:] :
                #if rcol == c: continue
                # Now rcol is a set of overlapping rows
                # Remember the overlappers
                if col_idx in self.cols:
                    removed_rows.update( self.remove( col_idx))
                # Each overlapping row has to be removed from all its columns
                #for crow in rcol:
                    # Now crow is a list of columns that need to lose the rows in the set rcol
                #    for col in crow:
                #        col.difference_update( rcol)
                        #self.cols[col].discard( crow) # Kill the overlapper
            #del( self.cols[min_col_idx])
            # Solve the rest without this row/col
            self.solve_( depth+1)
            #print( 'Depth: %d' % depth)
            self.restore( removed_rows, removed_cols)
            #BP()
            self.solution.pop()
        self.cols[min_col_idx] = min_col

    def remove( self, col_idx):
        rows = self.cols[col_idx]
        del( self.cols[col_idx])
        for row in rows:
            for colidx in row[1:] :
                if colidx != col_idx:
                    if colidx in self.cols:
                        self.cols[colidx].discard( row)
        return rows

    def restore( self, overlapping_rows, removed_col_idxs):
        for col_idx in removed_col_idxs:
            self.cols[col_idx] = removed_col_idxs[col_idx]
        for row in overlapping_rows:
            for col_idx in row[1:]:
                if col_idx in self.cols:
                    self.cols[col_idx].add(row)

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
        # Each column is a set of rows. A row is a tuple of column indexes.
        self.cols = { idx:set()  for idx in range( len(self.colnames)) }
        # Matrix entries. Weirdly self-referential.
        for rowidx,colidx in entries:
            self.rows[rowidx].append( colidx)
        # Convert rows from sets to tuples so they hash
        for idx, r in enumerate( self.rows):
            self.rows[idx] = tuple( self.rows[idx])
        # Put the rows into the column sets
        for rowidx,colidx in entries:
            self.cols[colidx].add( self.rows[rowidx])

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
