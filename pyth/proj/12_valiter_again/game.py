
'''
A game is one attempt to solve a puzzle.
Python 3
AHN, Mar 2020

Example usage:

s = Search( state, model)
g = Game( s)
seq = g.play( movelimit) # Abort if number of moves exeeds movelimit.

'''

from pdb import set_trace as BP
import numpy as np
import math
from search import Search
from shiftmodel import ShiftModel
from state import State

#================
class Game:

    def __init__( self, search):
        self.search = search

    def play( self):
        '''
        Kick off the game. Abort if no solution found after movelimit moves.
        Return the winning state sequence, including v and p for training.
        The triples (state,v,p) are used to train the network.
        '''
        s = self.search
        seq, solved = s.run()
        return seq, solved

def main():
    ' Test the Game class '
    SIZE=3
    model = ShiftModel( SIZE)
    model.load_weights( 'generator.h5')
    state = State.random( SIZE, nmoves=3)
    #state = State.from_list( SIZE, [1,4,2,3,0,5,6,7,8])
    search = Search( state, model)
    g = Game(search)
    seq, found = g.play( movelimit=100)
    if not found:
        print( '>>>>>>>>>> No solution found!')
    print( '\nPath:')
    for s in seq:
        print(s)

if __name__ == '__main__':
    main()
