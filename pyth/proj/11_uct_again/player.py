
'''
A Player tries to solve a shifting puzzle using UCTSearch and a DNN.
Python 3
AHN, Apr 2020

Example usage (also look in game.py):

state = initial_state
p = Player( state, model, playouts)
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

    def __init__( self, state, model, playouts=256, c_puct=0.016):
        '''
        model: The model. Must have a method get_v_p( state).
        playouts: How many nodes to expand before deciding on a move.
        c_puct: How much to rely on hope (exploration) in the puct calculation.
        '''
        self.model = model
        self.c_puct = c_puct
        self.playouts = playouts
        self.root = UCTNode( state, depth=0)
        self.history = set()

    def N( self, node=None):
        if not node: node = self.root
        return node.N

    def children_scores( self, node = None):
        'UCT score for each child'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            score = child.get_uct_score( self.c_puct)
            res[child.action] = score
        return res

    def children_nn_v( self, node = None):
        'Network v for each child'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            res[child.action] = child.nn_v
        return res

    def children_visits( self, node = None):
        'How many visits did each child have'
        if not node: node = self.root
        res = np.zeros( State.n_actions(), float)
        for child in node.children:
            res[child.action] = child.N
        return res

    '''
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
    '''

    def move( self):
        '''
        Search by expanding self.playouts leaves.
        Returns an action index, indicating the next move.
        '''
        # Expand the tree
        #N = 0
        for N in range(self.playouts):
            #N += 1
            leaf = None
            #LIMIT = 1000
            #loopcount = 0
            #while not leaf and loopcount < LIMIT:
            #    loopcount += 1
            leaf = self.__pick_leaf_to_expand()
            #if loopcount == LIMIT:
            #    print( 'Error: Not getting anywhere. Strange ...')
            #    exit(1)
            self.__expand_leaf( leaf)
        # Find best action
        if len(self.root.children) == 0:
            print( "Error: UCTree.search(): I don't think there is a solution.")
            return None
        winner,uct_score = self.root.get_best_child( self.c_puct, self.history)
        self.history.add( winner.state.hash())

        # Inherit tree below winner
        self.root = winner

        return winner.action, winner.state

    def __pick_leaf_to_expand( self): #@@@
        '''
        Decide which leaf to expand.
        Walk down the tree until the end, using best UCT score at each level.
        '''
        node = self.root
        while 1:
            if not node.children: # leaf
                return node
            newnode,uct_score = node.get_best_child( self.c_puct, self.history)
            #BP()
            node = newnode

    def __expand_leaf( self, leaf):
        '''
        Add children to a leaf, one per possible action.
        '''
        if leaf.state.solved(): # Solution, do not expand.
            print( '#', end='')
            leaf.N += 1
            #leaf.v = float(leaf.N)
            leaf.v = 1.0
            leaf.nn_v = 1.0
            self.__update_tree( leaf)
            #print( 'solution N:%d v:%f' % (self.N(leaf), self.v(leaf)))
            return

        #value = self.model.predict_one( leaf.state.encode()) # >>>>>>>> Run the network <<<<<<<<<
        #print(value)
        #print(policy)
        #leaf.nn_v = value
        #leaf.v = value
        #leaf.N = 1
        # Create children
        print( '.', end='', flush=True)
        actions = leaf.state.action_list()
        leaf.children = []
        for action in actions:
            next_state = leaf.state.act( action)
            value = self.model.predict_one( next_state.encode()) # >>>>>>>> Run the network <<<<<<<<<
            new_child = UCTNode( next_state, depth=leaf.depth+1, action=action, parent=leaf)
            new_child.v = value
            new_child.nn_v = value
            new_child.N = 0
            leaf.children.append( new_child)
        self.__update_tree( leaf)
        #leaf.children = sorted( leaf.children)

    def __update_tree( self, leaf):
        '''
        Update visit counts and values of all ancestors.
        Leaf just got expanded and has new children. If not, leaf is a solution.
        '''
        LARGE = 1E9
        node = leaf
        while node:
            if node.children is None:
                pass
            else:
                mmax = -1 * LARGE
                for child in node.children:
                    if child.v > mmax:
                        mmax = child.v
                node.v = State.v_plus_one( mmax)
            node.N += 1
            node = node.parent

#================
class UCTNode:
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
        self.nn_v = 0.0 # Our network score when we got expanded
        self.v = 0.0 # Accumulated experience
        self.N = 0 # Number of visits
        self.depth = depth

    def __repr__( self):
        res = self.state.__repr__()
        res += '\nchildren: %d\n' % (len(self.children) if self.children else 0)
        return res

    def get_best_child( self, c_puct, history):
        mmax = -1 * UCTNode.LARGE
        winner = None
        for child in self.children:
            if child.state.hash() in history: continue
            score = child.get_uct_score( c_puct)
            if score > mmax:
                mmax = score
                winner = child
        for child in self.children:
            if child.v >= 1.0 and not child is winner:
                BP()
                tt=42
        return winner,mmax

    # def worst_left_node_experience( self):
    #     '''
    #     Quality of the worst expanded node to the left of self at this point in the search.
    #     Used to estimate v for unexpanded nodes. This is a new idea.
    #     '''
    #     res = self.parent.nn_v
    #     for child in self.parent.children:
    #         if child is self: break
    #         if child.N:
    #             res = min( res, child.v / child.N)
    #     return res

    def get_uct_score( self, c_puct):
        '''
        UCT score decides which node gets expanded next.
        c_puct: How much to rely on hope. Larger means more exploration.
        '''
        if self.state.solved():
            return UCTNode.LARGE
        try:
            expected_v = State.v_from_dist( self.depth + State.dist_from_v( self.v))
            '''
            if not self.N: # unexpanded leaf
                #experience = self.worst_left_node_experience()
                #experience = State.v_minus_one( self.parent.v)
                #experience = State.v_plus_one( self.parent.v)
                #experience = self.parent.v
                expected_v = State.v_from_dist( self.depth + State.dist_from_v( parent.v))
            else:
                #experience = self.v
                expected_v = State.v_from_dist( self.depth + State.dist_from_v( self.v))
            '''
        except:
            BP()
            tt=42
        hope = math.sqrt(self.parent.N) / (1.0 + self.N)  # Hope helps us try new things
        res = expected_v + c_puct * hope
        return res
