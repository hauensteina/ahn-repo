#!/usr/bin/env python

from pdb import set_trace as BP

class AlgoX:
    '''
    Knuth's Algorithm X, as implemented at
    https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
    '''
    def __init__( self, rownames, colnames, entries, max_solutions=0):
        ''' Entries are pairs of (rowidx, colidx) '''
        self.max_solutions = max_solutions
        self.rownames = rownames
        self.colnames = colnames
        self.n_solutions = 0

        # Build the row dictionary Y
        self.Y = { y:[] for y in range( len(rownames)) }
        for e in entries:
            self.Y[e[0]].append( e[1])
        # Build the column dictionary X
        self.X = { x:set() for x in range( len(colnames)) }
        for e in entries:
            self.X[e[1]].add( e[0])

    def solve( self, solution=[]):
        ''' Beautiful recursive generator '''
        if self.max_solutions and (self.n_solutions == self.max_solutions): return
        if not self.X:
            self.n_solutions += 1
            yield list(solution)
        else:
            c = min( self.X, key=lambda c: len( self.X[c]))
            for r in list( self.X[c]):
                solution.append( self.rownames[r])
                cols = self.select( r)
                for s in self.solve( solution):
                    yield s
                self.deselect( r, cols)
                solution.pop()

    def select( self, r):
        cols = []
        for j in self.Y[r]:
            for i in self.X[j]:
                for k in self.Y[i]:
                    if k != j:
                        self.X[k].remove(i)
            cols.append( self.X.pop(j))
        return cols

    def deselect( self, r, cols):
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
        (6,1),(6,6)
    ]
    rownames = ['A','B','C','D','E','F','G']
    colnames = [0,1,2,3,4,5,6]

    solvgen = AlgoX( rownames, colnames, entries, 1).solve()
    for s in solvgen:
        print( s) # Should be ['B', 'D', 'F'] and ['B', 'D', 'G']

if __name__ == '__main__':
    main()
