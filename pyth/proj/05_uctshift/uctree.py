#!/bin/env python

# A generic tree class for UCT search.
# It might play a game, or solve a puzzle, or just sort an array.
# Leela Zero and AlphaGo Zero work like this, with minor tweaks.
# Look in uctsort.py to see how it sorts some numbers.
# Python 3
# AHN, Mar 2020

from pdb import set_trace as BP
import math

# Example usage:
# state = initial_state
# tree = UCTree( state, c_puct=0.6)
# while not tree.done( state):
#   action,state = tree.search( n_playouts=256)

#==============
class UCTree:
    LARGE = 1E9

    # state: The current position. Must have methods get_v_p() and act(action_idx).
    # c_puct: How much to rely on hope (exploration) in the puct calculation.
    #-------------------------------------------------------------------------------
    def __init__( self, state, c_puct):
        self.root = UCTNode( state)
        self.c_puct = c_puct
        self.expand_leaf( self.root) # possible first moves

    # Search by expanding at most n_playout leaves.
    # Returns an action index.
    #-------------------------------------------------
    def search( self, n_playouts):
        # Expand the tree
        N = 0
        while N < n_playouts:
            N += 1
            #print( '>>> playout %d' % N)
            leaf = None
            LIMIT = 1000
            loopcount = 0
            while not leaf and loopcount < LIMIT:
                loopcount += 1
                leaf = self.pick_leaf_to_expand()
            if loopcount == LIMIT:
                print( 'Error: Not getting anywhere. Strange ...')
                exit(1)
            self.expand_leaf( leaf)
        # Find best action
        if len(self.root.children) == 0:
            print( 'error: UCTree.search(): empty search result')
            return None
        winner = self.root.get_best_child( self.c_puct) # We could use largest N here, too.
        #print( '>>> move: %s' % winner.state.arr)
        # The winner is the new root
        self.root = winner
        return winner.action, winner.state

    # Termination
    #-------------------------
    def done( self, state):
        return state.get_v_p() == (1.0, None)

    # Which leaf to expand.
    # Walk down the tree until the end, using best UCT score.
    #----------------------------------------------------------
    def pick_leaf_to_expand( self):
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

    # Add children to a leaf, one per possible action.
    #---------------------------------------------------
    def expand_leaf( self, leaf):
        value, policy = leaf.state.get_v_p() # >>>>>>>> Run the net <<<<<<<<<
        if value == 1.0: # Solution, do not expand.
            leaf.N = 1
            leaf.v = 1.0
            return

        #print( policy)
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
        self.update_tree( leaf)

    # Update visit counts and values of all ancestors.
    #---------------------------------------------------
    def update_tree( self, leaf):
        node = leaf
        while node.parent:
            node.parent.v += leaf.v
            node.parent.N += 1
            node = node.parent

#================
class UCTNode:
    # state: State at this node.
    # action: What action got us here.
    # parent: Parent node.
    # p: Policy value we were assigned when the net ran on the parent.
    #-----------------------------------------------------------------------
    def __init__( self, state, action=None, parent=None, p=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.p = p # Our value from the parent policy array
        self.children = None
        self.v = None # Populates when we expand and run the net
        self.N = 0
        self.dead_end = False

    #---------------------
    def __repr__( self):
        res = self.state.__repr__()
        res += 'policy: %f\n' % (self.p or 0.0)
        res += 'value: %f\n' % (self.v or 0.0)
        res += 'N: %d\n' % (self.N or 0)
        res += 'children: %d\n' % (len(self.children) if self.children else 0)
        return res

    #-----------------------------------
    def get_best_child( self, c_puct):
        mmax = -1 * UCTree.LARGE
        winner = None
        for child in self.children:
            if child.dead_end:
                continue
            score = child.get_uct_score( c_puct)
            if score > mmax:
                mmax = score
                winner = child
        return winner

    # UCT score decides which node gets expanded next.
    # c_puct: How much to rely on hope. Larger means more exploration.
    #-------------------------------------------------------------------
    def get_uct_score( self, c_puct):
        if self.p == 0.0: return 0.0
        if not self.N: # Leaf
            experience = self.parent.v / self.parent.N
        else:
            experience = self.v / self.N # Our own winrate experience
        hope = self.p * ( math.sqrt(self.parent.N) / (1.0 + self.N) ) # Hope helps us try new things
        res = experience + c_puct * hope
        return res

    #--------------------------
    def __lt__( self, other):
        return self.p > other.p
