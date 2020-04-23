
'''
State of a shifting puzzle.
Python 3
AHN, Apr 2020
'''

from pdb import set_trace as BP
import math, random, json
from collections import defaultdict
import numpy as np

#===========================================
class StateJsonEncoder( json.JSONEncoder):
     def default(self, obj):
         if isinstance(obj, State):
             return {'s':obj.s, 'arr':list(obj.arr)}
         elif isinstance(obj, np.int64):
             return int(obj)
         elif isinstance(obj, np.ndarray):
             return list(obj)
         else:
             return json.JSONEncoder.default(self, obj)

#=================
class State:

    LEFT=0
    RIGHT=1
    UP=2
    DOWN=3

    LAMBDA=0.035

    def __init__( self, size):
        self.s = size
        self.empty_idx = 0
        self.arr = None
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
            res = res.act( random.choice(acts))
        # Make sure it is shuffled at all
        while res.solved():
            acts = res.action_list()
            res = res.act( random.choice(acts))

        return res

    @classmethod
    def from_list( cls, size, tile_list):
        '''
        Set position as specified by tile_list
        '''
        res = cls( size)
        res.arr = np.array( tile_list, int)
        res.empty_idx = np.argmin( res.arr)
        return res

    @classmethod
    def n_actions( cls):
        ' Returns maximum number of different actions. '
        return 4

    @classmethod
    def v_from_dist( cls, dist, lmbda=LAMBDA):
        ' Convert steps to go to a number in (-1,1) for tanh output '
        return 2 * math.exp(-lmbda * dist) - 1.0

    @classmethod
    def dist_from_v( cls, v, lmbda=LAMBDA):
        ' Convert tanh to steps to go '
        if v <= -1: return int(1E6)
        return -1 * math.log( (v+1)/2) / lmbda

    @classmethod
    def v_plus_one( cls, v, lmbda=LAMBDA):
        ' Get v(d(v)+1) '
        return cls.v_from_dist( cls.dist_from_v(v) + 1.0 )

    @classmethod
    def v_minus_one( cls, v, lmbda=LAMBDA):
        ' Get v(d(v)-1) '
        return cls.v_from_dist( cls.dist_from_v(v) - 1.0 )

    def solved( self):
        for i in range( self.s * self.s):
            if i != self.arr[i]: return False
        return True

    def action_flags( self):
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

    def action_list( self):
        ' Return a list of possible actions. '
        res = np.nonzero( self.action_flags())[0]
        return res

    def act( self, action):
        '''
        Act method required by Player
        Apply an action and return new state
        '''
        tile_idx = self.__new_idx( action)
        new_state = self.__clone()
        new_state.arr[self.empty_idx] = new_state.arr[tile_idx]
        new_state.arr[tile_idx] = 0
        new_state.empty_idx = tile_idx
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

    def manhattan_dist( self):
        ' Manhattan distance from solution '
        res = 0
        for idx,x in enumerate( self.arr):
            row, col = self.__coords( x)
            sol_row, sol_col = self.__coords( idx)
            d = abs( sol_row - row) + abs( sol_col - col)
            res += d
        return res

    def __clone( self):
        res = State( self.s)
        res.empty_idx = self.empty_idx
        res.arr = self.arr.copy()
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
