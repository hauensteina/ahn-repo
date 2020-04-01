
'''
A Player tries to solve a shifting puzzle using UCTSearch and a DNN.
Python 3
AHN, Mar 2020

Example usage (also look in game.py):

state = initial_state
p = Player( state, model, playouts, c_puct=0.6)
while not state.solved():
  action, state = p.move()

'''

from pdb import set_trace as BP
import math
import numpy as np
from collections import defaultdict
from state import State

#================
class Player:
    '''
    Uses a model and UCT search to find the next move.
    '''

    def __init__( self, state, model, playouts=256, c_puct=0.1):
        '''
        model: The model. Must have a method get_v_p( state).
        playouts: How many nodes to expand before deciding on a move.
        c_puct: How much to rely on hope (exploration) in the puct calculation.
        '''
        self.model = model
        self.c_puct = c_puct
        self.playouts = playouts
        self.root = UCTNode( state)

    def N( self, node=None):
        if not node: node = self.root
        return node.N

    def v( self, node=None):
        if not node: node = self.root
        return node.v / node.N if node.N else 0

    def normalized_child_visits( self, node = None):
        'How many visits did each child have, divided by total child visits'
        res = self.child_visits( node)
        ssum = np.sum( res)
        if ssum:
            res /= ssum
        return res

    def child_visits( self, node = None):
        'How many visits did each child have'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            res[child.action] = child.N
        return res

    def child_scores( self, node = None):
        'UCT score for each child'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            score = child.get_uct_score( self.c_puct)
            res[child.action] = score
        return res

    def child_nn_v( self, node = None):
        'Network v for each child'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            res[child.action] = child.nn_v
        return res

    def move( self):
        '''
        Search by expanding self.playouts leaves.
        Returns an action index, indicating the next move.
        '''
        # Expand the tree
        N = 0
        while N < self.playouts:
            N += 1
            leaf = None
            LIMIT = 1000
            loopcount = 0
            while not leaf and loopcount < LIMIT:
                loopcount += 1
                leaf = self.__pick_leaf_to_expand()
            if loopcount == LIMIT:
                print( 'Error: Not getting anywhere. Strange ...')
                exit(1)
            self.__expand_leaf( leaf)
        # Find best action
        if len(self.root.children) == 0:
            print( "Error: UCTree.search(): I don't think there is a solution.")
            return None
        winner,uct_score = self.root.get_best_child( self.c_puct)

        # Inherit tree below winner
        self.root = winner

        return winner.action, winner.state

    def __pick_leaf_to_expand( self):
        '''
        Decide which leaf to expand.
        Walk down the tree until the end, using best UCT score at each level.
        '''
        node = self.root
        while 1:
            if not node.children: # leaf
                return node
            newnode,uct_score = node.get_best_child( self.c_puct)
            node = newnode

    def __expand_leaf( self, leaf):
        '''
        Add children to a leaf, one per possible action.
        '''
        if leaf.state.solved(): # Solution, do not expand.
            leaf.N += 1
            leaf.v = float(leaf.N)
            leaf.nn_v = 1.0
            self.__update_tree( leaf, v=1.0, N=1)
            #print( 'solution N:%d v:%f' % (self.N(leaf), self.v(leaf)))
            return

        value, policy = self.model.get_v_p( leaf.state) # >>>>>>>> Run the network <<<<<<<<<
        leaf.nn_v = value
        leaf.v = value
        leaf.N = 1
        # Create a child for each policy entry, largest policy first
        leaf.children = []
        for idx,p in enumerate(policy):
            if p == 0.0: continue
            next_state = leaf.state.act( action_idx=idx)
            new_child = UCTNode( next_state, action=idx, parent=leaf, p=p)
            leaf.children.append( new_child)
        leaf.children = sorted( leaf.children)
        self.__update_tree( leaf, leaf.v, leaf.N)

    def __update_tree( self, leaf, v, N):
        '''
        Update visit counts and values of all ancestors.
        '''
        node = leaf
        while node.parent:
            node.parent.v += State.v_plus_one(v)
            node.parent.N += N
            node = node.parent

#================
class UCTNode:
    '''
    A node in the search tree
    '''
    LARGE = 1E9

    def __init__( self, state, action=None, parent=None, p=None):
        '''
        state: State at this node.
        action: What action got us here.
        parent: Parent node.
        p: Policy value we were assigned when the net ran on the parent.
        '''
        self.state = state
        self.action = action
        self.parent = parent
        self.p = p # Our value from the parent policy array
        self.children = None
        self.nn_v = 0.0 # Our network score when we got expanded
        self.v = 0.0 # Accumulated experience
        self.N = 0 # Number of visits

    def __repr__( self):
        res = self.state.__repr__()
        res += '\npolicy: %f\n' % (self.p or 0.0)
        res += 'children: %d\n' % (len(self.children) if self.children else 0)
        return res

    def __lt__( self, other):
        return self.p > other.p

    def get_best_child( self, c_puct):
        mmax = -1 * UCTNode.LARGE
        winner = None
        for child in self.children: # Left to right by descending policy
            score = child.get_uct_score( c_puct)
            if score > mmax:
                mmax = score
                winner = child
        return winner,mmax

    def worst_left_node_experience( self):
        '''
        Quality of the worst expanded node to the left of self at this point in the search.
        Used to estimate v for unexpanded nodes. This is a new idea.
        '''
        res = self.parent.nn_v
        for child in self.parent.children:
            if child is self: break
            if child.N:
                res = min( res, child.v / child.N)
        return res

    def get_uct_score( self, c_puct):
        '''
        UCT score decides which node gets expanded next.
        c_puct: How much to rely on hope. Larger means more exploration.
        '''
        if self.p == 0.0: return 0.0
        try:
            if not self.N: # unexpanded leaf
                experience = self.worst_left_node_experience()
            else:
                experience = self.v / self.N # Our own cost to go estimate. Aka winrate in games.
        except:
            BP()
            tt=42
        hope = self.p * ( math.sqrt(self.parent.N) / (1.0 + self.N) ) # Hope helps us try new things
        res = experience + c_puct * hope
        return res
