
'''
A Player tries to solve a shifting puzzle using UCTSearch and a DNN.
Python 3
AHN, Mar 2020

Example usage:

state = initial_state
p = Player( model, playouts, c_puct=0.6)
while not state.solved():
  action, state = p.move( state)

'''

from pdb import set_trace as BP
import math

#================
class Player:
    '''
    Uses a model and UCT search to solve find the next move.
    '''
    LARGE = 1E9

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

    def move( self):
        '''
        Search by expanding at most n_playout leaves.
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
        winner = self.root.get_best_child( self.c_puct)

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
            newnode = node.get_best_child( self.c_puct)
            if not newnode: # all children are dead ends
                node.dead_end = True # Don't try again
                return None
            else:
                node = newnode

    def __expand_leaf( self, leaf):
        '''
        Add children to a leaf, one per possible action.
        '''
        if leaf.state.solved(): # Solution, do not expand.
            leaf.N += 1 # leaf.N = 1
            leaf.v = float(leaf.N)
            #print( 'solution N:%d v:%f' % (leaf.N, leaf.v))
            return

        value, policy = self.model.get_v_p( leaf.state) # >>>>>>>> Run the network <<<<<<<<<
        leaf.v = value
        leaf.N = 1
        # Create a child for each policy entry, largest policy first
        leaf.children = []
        for idx,p in enumerate(policy):
            if p == 0.0: continue
            next_state = leaf.state.act( action_idx=idx)
            new_child = UCTNode( next_state, action=idx, parent=leaf, p=p)
            leaf.children.append( new_child)
        if not leaf.children:
            leaf.dead_end = True
            return
        leaf.children = sorted( leaf.children)
        self.__update_tree( leaf)

    def __update_tree( self, leaf):
        '''
        Update visit counts and values of all ancestors.
        '''
        node = leaf
        while node.parent:
            node.parent.v += leaf.v
            node.parent.N += leaf.N # Usually N=1 for a leaf, except if solution.
            node = node.parent

#================
class UCTNode:
    '''
    A node in the search tree
    '''

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
        self.v = None # Populates when we expand and run the net
        self.N = 0
        self.dead_end = False

    def __repr__( self):
        res = self.state.__repr__()
        res += '\npolicy: %f\n' % (self.p or 0.0)
        res += 'value: %f\n' % (self.v / self.N if self.N  else 0.0)
        res += 'N: %d\n' % (self.N or 0)
        res += 'children: %d\n' % (len(self.children) if self.children else 0)
        return res

    def __lt__( self, other):
        return self.p > other.p

    def get_best_child( self, c_puct):
        mmax = -1 * Player.LARGE
        winner = None
        for child in self.children:
            if child.dead_end:
                continue
            score = child.__get_uct_score( c_puct)
            if score > mmax:
                mmax = score
                winner = child
        return winner

    def __get_uct_score( self, c_puct):
        '''
        UCT score decides which node gets expanded next.
        c_puct: How much to rely on hope. Larger means more exploration.
        '''
        if self.p == 0.0: return 0.0
        if not self.N: # Leaf
            experience = self.parent.v / self.parent.N
        else:
            experience = self.v / self.N # Our own winrate experience
        hope = self.p * ( math.sqrt(self.parent.N) / (1.0 + self.N) ) # Hope helps us try new things
        res = experience + c_puct * hope
        return res
