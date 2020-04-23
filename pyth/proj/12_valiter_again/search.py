
'''
Search tries to solve a shifting puzzle using value iteration and a DNN.
Python 3
AHN, Apr 2020

Example usage (also look in game.py):

state = initial_state
p = Player( state, model, playouts)
seq = p.search( maxvisits=1000)

'''

from pdb import set_trace as BP
import math
import numpy as np
from collections import defaultdict
from sortedcontainers import SortedList
from state import State

#================
class Search:
    '''
    Tries to find a solution using a model and A* search.
    '''

    def __init__( self, state, model, maxnodes, maxdepth):
        '''
        state: Puzzle configuration to solve. Needs a method encode().
        model: The model. Must have a method predict_one( state.endode())
        maxnodes: How many nodes to expand before giving up.
        maxdepth: Only expand nodes with depth < maxdepth.
        '''
        self.model = model
        self.maxnodes = maxnodes
        self.maxdepth = maxdepth
        self.leaves = SortedList()

        self.root = SearchNode( state, depth=0)
        self.root.nn_v = model.predict_one( self.root.state.encode()) # >>>>>>>> Run the network <<<<<<<<<
        if self.root.state.solved():
            self.root.nn_v = 1.0
        self.leaves.add( self.root)

    def run( self):
        '''
        Search by expanding at most maxnodes leaves.
        Returns (best_sequence:<List<SearchNode>>, could_solve:<Bool>)
        '''
        for N in range( self.maxnodes):
            best_leaf = self.leaves[0]
            self.leaves.remove( best_leaf)
            if best_leaf.state.solved():
                best_leaf.v = 1.0
                print( '>>> Solved after %d node expansions' % N)
                break
            if best_leaf.depth < self.maxdepth and N+1 < self.maxnodes:
                newleaves = best_leaf.expand( self.model)
                self.leaves.update( newleaves)
                '''
                if N % 1 == 0:
                    print( 'Expanding node %d' % N)
                    print( '>>>> Leaves:')
                    print( self.leaves)
                    print( '--------------------')
                '''
        seq = best_leaf.get_nodeseq()
        return seq, best_leaf.state.solved()

#==================
class SearchNode:
    '''
    A node in the search tree
    '''
    LARGE = 1E9

    def __init__( self, state, depth, action=None, parent=None):
        '''
        state: State at this node.
        action: What action got us here.
        parent: Parent node.
        '''
        self.state = state
        self.action = action
        self.parent = parent
        self.children = None
        self.nn_v = -1.0 # Our network score when we got expanded
        self.v = -1.0 # Largest child nn_v + 1
        self.depth = depth
        self.dead_end = False

    def __repr__( self):
        res = self.state.__repr__()
        res += '\nchildren: %d\n' % (len(self.children) if self.children else 0)
        res += 'depth: %d\n' % self.depth
        res += 'dead_end: %s\n' % self.dead_end
        res += 'nn_ctg: %f\n' % State.dist_from_v( self.nn_v)
        res += 'better_ctg: %f\n' % State.dist_from_v( self.v)
        return res

    def __lt__( self, other):
        return self.depth + State.dist_from_v( self.nn_v) < other.depth + State.dist_from_v( other.nn_v)

    def __eq__( self, other):
        return self.state.hash() == other.state.hash()

    def in_my_history( self, state):
        ' Return True if state is in my history '
        statehash = state.hash()
        histnode = self
        while histnode:
            if statehash == histnode.state.hash():
                return True
            histnode = histnode.parent
        return False

    def get_nodeseq( self):
        ' Get node sequence leading to this leaf '
        if self.children:
            print( 'ERROR: get_nodeseq(): trying to expand non-leaf')
            exit(1)
        res = []
        node = self
        while node:
            res.append( node)
            node = node.parent
        res.reverse()
        #print( res)
        #BP()
        return res

    def expand( self, model):
        ' Expand leaf by adding non-cycle children '
        if self.children:
            print( 'ERROR: expand(): trying to expand non-leaf')
            exit(1)
        if self.dead_end:
            print( 'expand(): dead end')
            return []
        actions = self.state.action_list()
        self.children = []
        for action in actions:
            next_state = self.state.act( action)
            if self.in_my_history( next_state):
                continue
            new_child = SearchNode( next_state, depth=self.depth+1, action=action, parent=self)
            new_child.nn_v = model.predict_one( next_state.encode()) # >>>>>>>> Run the network <<<<<<<<<
            if new_child.state.solved():
                new_child.nn_v = 1.0
            self.children.append( new_child)
        if not self.children:
            self.dead_end = True
            print('dead end')
            return []
        # Our cost to go estimate is 1 + cost_to_go(best_child)
        # This is what's called value iteration.
        self.v = max( [child.nn_v for child in self.children])
        self.v = State.v_plus_one( self.v)

        return self.children
