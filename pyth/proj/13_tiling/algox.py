#!/usr/bin/env python

'''
A Python implementation of Knuth's Algorithm X
AHN, Nov 2020
'''

from pdb import set_trace as BP

class AlgoX:
    ''' Knuth's Algorithm X, general purpose '''

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
        def __repr__( self):
            return self.colheader.name

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
        def __repr__( self):
            return '\n' + self.name + '\n' + str( self.entries)

    def __init__( self, rownames, colnames, entries, max_solutions=0):
        ''' Entries are pairs of (rowidx, colidx) '''
        self.max_solutions = max_solutions
        self.nsolutions = 0
        self.colnames = colnames
        self.solution = [] # A solution is a list of row headers
        self.solutions = [] # A list of solutions
        # Row headers
        self.rows = {}
        self.complete_rows = {} # Retain to print solution
        for rname in rownames:
            self.rows[rname] = AlgoX.Header( rname)
            self.complete_rows[rname] = AlgoX.Header( rname)
        # Column headers
        self.cols = {}
        for cname in colnames:
            self.cols[cname] = AlgoX.Header( cname)

        # Matrix entries
        for rowidx,colidx in entries:
            rname = rownames[rowidx]
            cname = colnames[colidx]
            entry = AlgoX.Entry( rname, cname, self.rows[rname], self.cols[cname])
            self.rows[rname].entries.add( entry)
            self.complete_rows[rname].entries.add( entry)
            self.cols[cname].entries.add( entry)

    def print_state( self):
        print( 'State:')
        print( '---------------------')
        print( '%d rows' % len(self.rows))
        print( '%d columns:' % len(self.cols))
        for col in self.cols.values():
            print( '%s:%s ' % (col.name, len( col.entries)), end='')
        rowelts = sum( [len(x.entries) for x in self.rows.values()])
        colelts = sum( [len(x.entries) for x in self.cols.values()])
        print( 'rowelts:%d colelts:%d' % (rowelts, colelts))
        if colelts != rowelts:
            print( 'ERROR: rowelts not equal to colelts!')

    def remove_col( self, colheader):
        '''
        Remove a column.
        '''
        # Remove column from all its rows. This is necessary to avoid dups.
        for e in colheader.entries:
            e.rowheader.entries.remove(e)

        # Remove the col itself
        del self.cols[colheader.name]
        return colheader

    def restore_col( self, colheader):
        ''' Put the column back '''
        self.cols[colheader.name] = colheader
        # Add each col entry to its row
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
        del self.rows[rowheader.name]
        return rowheader

    def restore_row( self, rowheader):
        ''' Put the row back '''
        self.rows[rowheader.name] = rowheader
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
            self.restore_col( c)

    def get_overlappers( self, rowheader):
        ''' Find all row headers which overlap this row '''
        res = set()
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
        print( '\nSolution: %s' % str( [x.name for x in self.solution]))
        print( '---------------')
        for rowheader in self.solution:
            print( self.row2list( rowheader))

    def get_col_idxs( self, rowname): #@@@
        rowheader = self.rows[rowname]
        res = [self.colnames.index( e.colheader.name) for e in rowheader.entries]
        return res

    def solve( self):
        self.solve_()
        res = [ [r.name for r in s] for s in self.solutions]
        return res

    def solve_( self, depth=0):
        ''' Run Algorithm X '''

        if self.max_solutions and (self.nsolutions >= self.max_solutions): return

        def check_dead_end( cheads):
            'If any column has no more entries, we are stuck'
            for chead in cheads:
                if len(chead.entries) == 0:
                    return True
            return False

        colheader = self.pick_col() # A hole to cover or a piece to place
        #print( '\nworking on col %s' % colheader.name)
        for ridx, rowentry in enumerate(colheader.entries): # for images covering this hole
            if depth == 0:
                print( 'Working on row %d/%d' % (ridx+1, len(colheader.entries)))
            elif depth == 1:
                print( '    Working on row %d/%d' % (ridx+1, len(colheader.entries)))
            rem_rows = set()
            rem_cols = set()
            rowheader = rowentry.rowheader
            orows = self.get_overlappers( rowheader)
            # Remove all columns we cover
            cols_to_remove = rowheader.entries.copy()
            for e in cols_to_remove:
                rem_cols.add( self.remove_col( e.colheader))
            # Remove all overlapping rows
            for r in orows:
                rem_rows.add( self.remove_row( r))
            # If no cols are left, we are done
            if len( self.cols) == 0:
                self.nsolutions += 1
                print( 'Found solution %d' % self.nsolutions)
                self.solution.append( self.complete_rows[rowentry.rowheader.name])
                self.solutions.append( self.solution.copy())
                self.solution.pop()
                self.restore( rem_rows, rem_cols)
                if self.max_solutions and (self.nsolutions >= self.max_solutions): return
                continue # Look for more solutions

            if check_dead_end( self.cols.values()):
                #print( 'Dead End')
                self.restore( rem_rows, rem_cols)
                continue

            partial_solution = self.solution.copy()
            self.solution.append( self.complete_rows[rowentry.rowheader.name])
            # Alright, so we filled a hole. Now fill another one.
            self.solve_( depth+1)
            self.solution = partial_solution
            self.restore( rem_rows, rem_cols)

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
    colnames = ['0','1','2','3','4','5','6']

    solver = AlgoX( rownames, colnames, entries)
    solutions = solver.solve()
    for s in solutions:
        print( s) # Should be ['A', 'G'] and ['B', 'D', 'F']

if __name__ == '__main__':
    main()
