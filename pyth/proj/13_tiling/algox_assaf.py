#!/usr/bin/env python

from pdb import set_trace as BP

class AlgoX:
    '''
    Knuth's Algorithm X, as implemented at
    https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
    '''
    def __init__( self, rownames, rowclasses, colnames, entries, max_solutions=0):
        ''' Entries are pairs of (rowidx, colidx) '''
        self.max_solutions = max_solutions
        self.rownames = rownames
        self.rowclasses = rowclasses
        self.colnames = colnames
        self.n_solutions = 0
        self.seen_classes = set()

        # Build the row dictionary Y:dict(int:list(int))
        self.Y = { y:[] for y in range( len(rownames)) }
        for e in entries:
            self.Y[e[0]].append( e[1])
        # Build the column dictionary X:dict(int:set(int))
        self.X = { x:set() for x in range( len(colnames)) }
        for e in entries:
            self.X[e[1]].add( e[0])

    def solve( self, solution=[], depth=0):
        ''' Beautiful recursive generator '''
        #print( 'depth %d' % depth)
        seen_classes = set()
        if self.max_solutions and (self.n_solutions == self.max_solutions): return
        if not self.X:
            self.n_solutions += 1
            yield list(solution)
        else:
            c = min( self.X, key=lambda c: len( self.X[c]))
            #print('column %s' % self.colnames[c])
            seen_here = set()
            rows =  list( self.X[c])
            for ridx, r in enumerate( rows):
                if depth == 0:
                    print( 'Working on row %d/%d' % (ridx+1, len(rows)))
                elif depth == 1:
                    print( '    Working on row %d/%d' % (ridx+1, len(rows)))

                # If we have a piece more than once, don't try the dups again
                if self.rowclasses[r] in self.seen_classes:
                    continue
                self.seen_classes.add( self.rowclasses[r])
                seen_here.add( self.rowclasses[r])
                #print('rowclass:%s' % (self.rowclasses[r]))
                solution.append( self.rownames[r])
                cols = self.select_( r)
                for s in self.solve( solution, depth+1):
                    yield s
                self.deselect_( r, cols)
                solution.pop()
            self.seen_classes.difference_update( seen_here)

    def get_col_idxs( self, rowname):
        rowidx = self.rownames.index(rowname)
        return self.Y[rowidx]

    def select_( self, r):
        cols = []
        for j in self.Y[r]:
            for i in self.X[j]:
                for k in self.Y[i]:
                    if k != j:
                        self.X[k].remove(i)
            cols.append( self.X.pop(j))
        return cols

    def deselect_( self, r, cols):
        for j in reversed( self.Y[r]):
            self.X[j] = cols.pop()
            for i in self.X[j]:
                for k in self.Y[i]:
                    if k != j:
                        self.X[k].add(i)


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

    solvgen = AlgoX( rownames, colnames, entries).solve()
    for s in solvgen:
        print( s) # Should be ['A', 'G'] and ['B', 'D', 'F']

if __name__ == '__main__':
    main()
