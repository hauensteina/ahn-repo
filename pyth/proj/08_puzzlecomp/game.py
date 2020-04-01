
'''
A game is one attempt to solve a puzzle.
Python 3
AHN, Mar 2020

Example usage:

p = Player( state, model, playouts)
g = Game( p)
seq = g.play( movelimit) # Abort if number of moves exeeds movelimit.

'''

from pdb import set_trace as BP
import numpy as np
import math
from player import Player
from shiftmodel import ShiftModel
from state import State

#================
class Game:

    def __init__( self, player):
        self.player = player

    def play( self, movelimit):
        '''
        Kick off the game. Abort if no solution found after movelimit moves.
        Return the winning state sequence, including v and p for training.
        The triples (state,v,p) are used to train the network.
        '''
        pl = self.player
        nmoves = 0
        res = []
        while (not pl.root.state.solved()) and (nmoves < movelimit):
            nmoves += 1
            old_root = pl.root
            action, state = pl.move() # changes pl.root
            res.append( {'state':old_root.state,
                         'visits':pl.N( old_root),
                         'uct':pl.child_scores( old_root),
                         'child_nn_v':pl.child_nn_v( old_root),
                         'v':0.0,
                         'child_visits':pl.child_visits( old_root),
                         'p':pl.normalized_child_visits( old_root) })

        if not pl.root.state.solved():
            found = False
        else:
            found = True
            # Fill in the v values after we know the solution length
            for idx in range( nmoves):
                res[idx]['v'] = State.v_from_dist( nmoves - idx)
            # Add solution at the end
            res.append( {'state':pl.root.state,
                         'visits':pl.N( pl.root),
                         'uct':np.zeros( State.n_actions(),float),
                         'child_nn_v':np.zeros( State.n_actions(),float),
                         'child_visits':np.zeros( State.n_actions(),float),
                         'p':np.zeros( State.n_actions(),float),
                         'v':1.0} )

        return res,found

def main():
    ' Test the Game class '
    SIZE=3
    model = ShiftModel( SIZE)
    model.load_weights( 'model_3x3')
    state = State.random( SIZE, nmoves=16)
    #state = State.from_list( SIZE, [1,4,2,3,0,5,6,7,8])
    player = Player( state, model, playouts=256, c_puct=0.1)
    g = Game(player)
    seq, found = g.play( movelimit=100)
    if not found:
        print( '>>>>>>>>>> No solution found!')
    print( '\nPath:')
    for s in seq:
        print( s['state'])
        print('visits: %d' % s['visits'])
        print('v:%f' % s['v'])
        print('child_nn_v:%s' % s['child_nn_v'])
        print('child_visits:%s' % s['child_visits'])
        print('uct:%s' % s['uct'])
        print('p:%s' % s['p'])


if __name__ == '__main__':
    main()
