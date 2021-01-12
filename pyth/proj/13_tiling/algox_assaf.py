#!/usr/bin/env python

from pdb import set_trace as BP

LOGGING_ON = False

'''
If solving pentomino puzzles, we need to deal with more than one instance per piece.
It there are several F pieces, those are called the representatives (aka reprs) of F .
Each repr(F) can occur in many orientations and positions.
These are called *images*. So there is a two level hierarchy, where a piece
has several representatives, and each representative has several images.
The current representative of F is reprs[F][0].
rowname: F#0_3 is the third image of repr 0 of F. We do not parse this.
colname: Either a numeric string (eg '1'), or repr names like F#1. We do not parse this.
rowclass: (F,0) is the class of all rows like F#0_N
reprs[piece]: A list of remaining reprs of piece. reprs['F'] = [('F',0), ('F',1), ... ]
X contains the columns as a dict where the keys are ints
Y contains the rows as a dict where the keys are ints
'''

class AlgoX:
    '''
    Knuth's Algorithm X, as implemented at
    https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
    '''
    def __init__( self, rownames, colnames, colcounts, entries, max_solutions=0):
        '''
        Entries is a set of (rowidx, colidx)
        '''
        self.max_solutions = max_solutions
        self.rownames = rownames
        #self.rowclasses = rowclasses
        self.colnames = colnames
        self.colcounts = colcounts
        self.n_solutions = 0
        self.ncalls = 0
        #self.seen_classes = set()
        # Build the row dictionary Y:dict(int:list(int))
        #self.Y = { y:[] for y in range( len(rowclasses)) }
        self.Y = { y:[] for y in range( len(rownames)) }
        #classes = { x[0] for x in rowclasses }
        #self.reprs = { x:[] for x in classes }

        #self.row_active = [ False for _ in range( len(rownames)) ]
        for e in entries:
            row,col = e[0],e[1]

            # Shenanigans to efficiently deal with multiple instances of a piece
            #piece = rowclasses[row][0] # 'F'
            #if not rowclasses[row] in self.reprs[piece]:
            #    self.reprs[piece].append( rowclasses[row]) # reprs = [('F',0), ('F',1), ... ]
            #if rowclasses[row] == self.reprs[piece][0]:
	        #    self.row_active[row] = True

            self.Y[row].append( col)

        # Build the column dictionary X:dict(int:set(int))
        self.X = { x:set() for x in range( len(colnames)) }
        for e in entries:
            self.X[e[1]].add( e[0])

    def solve( self, mode='basic'):
        if mode == 'basic':
            return self.solve_basic()
        elif mode == 'nopieces':
            return self.solve_basic()
        elif mode == 'classes':
            return self.solve_classes()
        elif mode == 'queue':
            return self.solve_queue()

    def print_state( self, depth, colidx):
        if not LOGGING_ON: return
        X = self.X
        print( (' ' * depth) + '>>>>>>>>>> depth %d chose column %s' % (depth, self.colnames[colidx]))
        print( 'columns: %s' %
               (list(zip( [self.colnames[c] for c in X], [ [ r for r in self.X[c] ] for c in X] ))))
        print( 'rows: %s' %
               (list(zip( [r for r in self.Y], [self.rownames[r] for r in self.Y ]))))
        print( 'counts: %s' %
               (list(zip( [self.colnames[c] for c in X], [len(self.X[c]) for c in X]))))

    def solve_basic( self, solution=[], depth=0):
        ''' No frills clean implementation '''
        #print( 'Enter depth %d' % depth)
        self.ncalls += 1
        if self.max_solutions and (self.n_solutions == self.max_solutions): return
        if not self.X:
            self.n_solutions += 1
            print( 'Found solution %d' % self.n_solutions)
            yield list(solution)
        else:
            colidx = min( self.X, key=lambda c: len( self.X[c]))
            self.print_state( depth, colidx)
            rows = list(self.X[colidx])
            for ridx,r in enumerate( rows):
                if depth in range(10):
                    print( ('  ' * depth) + 'Working on row %d/%d' % (ridx+1, len(rows)))

                solution.append( self.rownames[r])
                cols = self.select_( r)
                for s in self.solve_basic( solution, depth+1):
                    yield s
                self.deselect_( r, cols)
                solution.pop()
        if depth == 0:
            print( '$$$$$$$$$$$$$ Total calls to solve_basic(): %d' % self.ncalls)

    def solve_queue( self, solution=[], depth=0):
        ''' Instance queues for repeating pieces '''
        #print( 'Enter depth %d' % depth)
        self.ncalls += 1
        #seen_classes = set()

        def print_active_rows( depth):
            print( ' ' * depth, end='')
            for r,rn in enumerate(self.rownames):
                if self.row_active[r]:
                    print( '%s ' % rn, end='')
            print()

        def active_rows(colidx):
            return [ rowidx for rowidx in self.X[colidx] if self.row_active[rowidx] ]

        def active_cols():
            return [ colidx for colidx in self.X if len(active_rows( colidx)) > 0 ]

        if self.max_solutions and (self.n_solutions == self.max_solutions): return
        X = active_cols()
        if not self.X:
            self.n_solutions += 1
            yield list(solution)
        elif not X:
            #print( '########## Dead End')
            return
        else:
            #colidx = min( active_cols(), key=lambda c: len( self.X[c]))
            colidx = min( X, key=lambda c: len( active_rows(c)))

            #print( '>>>>>>>>>> chose column %s' % self.colnames[colidx])
            #print( 'columns: %s' % (list(zip( [self.colnames[c] for c in X], [ [ r for r in self.X[c] if self.row_active[r]] for c in X] ))))
            #print( 'rows: %s' % (list(zip( [r for r in self.Y if self.row_active[r]], [self.rownames[r] for r in self.Y if self.row_active[r]]))))
            #print( 'counts: %s' % (list(zip( [self.colnames[c] for c in X], [len(self.X[c]) for c in X]))))
            #seen_here = set()
            #rows =  list( self.X[c])
            #rows = [ rowidx for rowidx in self.X[c] if self.row_active[rowidx] ]
            rows = active_rows( colidx)
            # if not rows: print( 'Dead End')
            for ridx, r in enumerate( rows):
                if depth == 0:
                    print( 'Working on row %d/%d' % (ridx+1, len(rows)))
                elif depth == 1:
                    print( '    Working on row %d/%d' % (ridx+1, len(rows)))

                #print( ' ' * depth + 'Depth %d' % depth, end='')
                #print( ' Trying row %s' % self.rownames[r])

                # If we have a piece more than once, don't try the dups again
                #if self.rowclasses[r] in self.seen_classes:
                #    continue
                #self.seen_classes.add( self.rowclasses[r])
                #seen_here.add( self.rowclasses[r])
                #print('rowclass:%s' % (self.rowclasses[r]))
                solution.append( self.rownames[r])
                rclass = self.rowclasses[r] # ('Y',0)
                activated_rows = self.activate_next_repr( rclass)
                cols = self.select_( r)
                #print('before:')
                #print_active_rows(depth)
                #BP()
                for s in self.solve_queue( solution, depth+1):
                    yield s
                self.deselect_( r, cols)
                self.deactivate( rclass, activated_rows, depth)
                #print('after:')
                #print_active_rows(depth)
                #BP()
                solution.pop()
            #self.seen_classes.difference_update( seen_here)
        if depth == 0:
            print( '$$$$$$$$$$$$$ Total calls to solve_queue(): %d' % self.ncalls)

    def get_col_idxs( self, rowname):
        rowidx = self.rownames.index(rowname)
        return self.Y[rowidx]

    def activate_next_repr( self, rclass): # rclass ~ ('Y', 0)
        '''
        Activate the next representative of this piece.
        Return activated rows.
        '''
        activated = []
        piece = rclass[0]
        popped = self.reprs[piece].pop(0)
        #print('popped repr %s' % (str(popped)))
        if popped != rclass:
            BP()
            should_be_the_same = 1

        if len( self.reprs[piece]) == 0: return []
        new_repr = self.reprs[piece][0]
        for r,cclass in enumerate( self.rowclasses):
            if cclass == new_repr:
                self.row_active[r] = True
                activated.append(r)
        #tt = [self.rowclasses[x] for x in activated]
        #BP()
        return activated

    def deactivate( self, rclass, activated_rows, depth):
        #rclass = self.rowclasses[ activated_rows[0]]
        piece = rclass[0]
        self.reprs[piece].insert( 0, rclass)
        #print('Depth %d pushed repr %s' % (depth, str(rclass)))
        for r in activated_rows:
            self.row_active[r] = False

    def select_( self, r):
        cols = []
        for col1 in self.Y[r]:
            self.colcounts[col1] -= 1
            #print( 'decr col %s to %d' % (self.colnames[col1], self.colcounts[col1]))
            if self.colcounts[col1] > 0:
                #print( 'Not removing col %s' % self.colnames[col1])
                continue
            if self.colcounts[col1] < 0:
                print( 'ERROR: colcount below 0')
                exit(1)
            for row in self.X[col1]:
                for col2 in self.Y[row]:
                    if col2 != col1:
                        self.X[col2].remove( row)
            cols.append( self.X.pop( col1))
        return cols

    def deselect_( self, r, cols):
        for col1 in reversed( self.Y[r]):
            self.colcounts[col1] += 1
            #print( 'incr col %s to %d' % (self.colnames[col1],self.colcounts[col1]))
            if self.colcounts[col1] == 1:
                #print( 'popping col %s' % self.colnames[col1])
                self.X[col1] = cols.pop()
            for row in self.X[col1]:
                for col2 in self.Y[row]:
                    if col2 != col1:
                        self.X[col2].add( row)


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
