
'''
State of a shifting puzzle.
Python 3
AHN, Mar 2020
'''

from pdb import set_trace as BP
import numpy as np
import random

#=================
class State:

    LEFT=0
    RIGHT=1
    UP=2
    DOWN=3

    def __init__( self, size):
        self.s = size
        self.empty_idx = 0
        self.arr = None
        self.history = set()
        self.hashval = None

    def __repr__( self):
        res = ''
        old_row = -1
        for idx, x in enumerate( self.arr):
            row, col = self.__coords( idx)
            if row != old_row:
                old_row = row
                res += '\n'
            if x: res += '%3d' % x
            else: res += '   '

        return res

    @classmethod
    def random( cls, size, nmoves=1000):
        '''
        Returns a randomized shifting puzzle ready to be solved.
        '''
        res = cls( size)
        res.arr = np.zeros( size * size, int)
        for i,_ in enumerate( res.arr): res.arr[i] = i

        for _ in range(nmoves):
            acts = res.action_list()
            res = res.act( random.choice(acts), store_history=False)
        return res

    def solved( self):
        for i in range( self.s * self.s):
            if i != self.arr[i]: return False
        return True

    def action_list( self):
        ' Return a list of possible actions. '
        res = np.nonzero( self.__action_flags())[0]
        return res

    def act( self, action_idx, store_history=True):
        '''
        Act method required by Player
        Apply an action and return new state
        '''
        tile_idx = self.__new_idx( action_idx)
        new_state = self.__clone()
        new_state.arr[self.empty_idx] = new_state.arr[tile_idx]
        new_state.arr[tile_idx] = 0
        new_state.empty_idx = tile_idx
        if (store_history):
            new_state.history.add( self.hash())
        return new_state

    def encode( self):
        ' Encode state to feed into DNN '
        # Width, height, tile-channels
        res = np.zeros( (self.s, self.s, self.s * self.s), float)
        for idx in range( self.s * self.s): # for each position
            tile = self.arr[idx] # which tile is here
            r,c = self.__coords( idx)
            res[r,c,tile] = 1 # store position in channel for this tile
        return res

    def hash( self):
        ''' Hash state to avoid cycles '''
        if not self.hashval: self.hashval = hash(self.arr.tobytes())
        return self.hashval


    def __clone( self):
        res = State( self.s)
        res.empty_idx = self.empty_idx
        res.arr = self.arr.copy()
        res.history = self.history.copy()
        return res

    def __action_flags( self):
        'Return an array of length 4 where impossible actions are marked 0'
        mmax = self.s - 1
        row,col = self.__coords( self.empty_idx)
        res = np.zeros( 4, int)
        if col == 0 and row == 0:
            res[State.DOWN] = res[State.RIGHT] = 1
        elif col == 0 and row == mmax:
            res[State.UP] = res[State.RIGHT] = 1
        elif col == mmax and row == 0:
            res[State.DOWN] = res[State.LEFT] = 1
        elif col == mmax and row == mmax:
            res[State.UP] = res[State.LEFT] = 1
        elif col == 0:
            res[State.UP] = res[State.DOWN] = res[State.RIGHT] = 1
        elif col == mmax:
            res[State.UP] = res[State.DOWN] = res[State.LEFT] = 1
        elif row == 0:
            res[State.LEFT] = res[State.RIGHT] = res[State.DOWN] = 1
        elif row == mmax:
            res[State.LEFT] = res[State.RIGHT] = res[State.UP] = 1
        else:
            res[State.LEFT] = res[State.RIGHT] = res[State.UP] = res[State.DOWN] = 1
        return res

    def __coords( self, index):
        row = index // self.s
        col = index % self.s
        return row,col

    def __index( self, row, col):
        return self.s * row + col

    def __new_idx( self, action):
        row, col = self.__coords( self.empty_idx)
        if action == State.LEFT:
            return self.__index( row, col-1)
        elif action == State.RIGHT:
            return self.__index( row, col+1)
        elif action == State.UP:
            return self.__index( row-1, col)
        elif action == State.DOWN:
            return self.__index( row+1, col)
        else:
            print( 'Error: new_idx(): Invalid action %d' % action)
