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
# tree = UCTree( state, get_v_p, get_next_state, c_puct=0.6)
# while not tree.done( state):
#   action,state = tree.search( n_playouts=256)

#==============
class UCTree:
    LARGE = 1E9

    # state: The current position.
    # get_v_p: A function that takes a state and computes a quality estimate v
    #          and a sorted array p of next action probabilities. Typically this
    #          would feed state through a neural net, but not necessarily.
    # get_next_state: A function that takes a (state,action) pair and returns
    #                 the next state.
    # c_puct: A constant used in the puct calculation.
    #-------------------------------------------------------------
    def __init__( self, state, get_v_p, get_next_state, c_puct):
        self.root = UCTNode( state)
        self.get_v_p = get_v_p
        self.get_next_state = get_next_state
        self.c_puct = c_puct
        value, policy = self.get_v_p( self.root.state) # The first move
        self.root.v = value
        self.root.N = 1

    # Search by expanding at most n_playout leaves.
    # Returns an action index.
    #-------------------------------------------------
    def search( self, n_playouts):
        # Expand the tree
        N = 0
        while N < n_playouts:
            N += 1
            #print( '>>> playout %d' % N)
            leaf = self.pick_leaf_to_expand()
            self.expand_leaf( leaf)
        # Find best action
        if len(self.root.children) == 0:
            print( 'error: UCTree.search(): empty search result')
            return None
        winner = self.root.get_best_child( self.c_puct)
        #print( '>>> move: %s' % winner.state.arr)
        # The winner is the new root
        self.root = winner
        return winner.action, winner.state

    # Termination
    #-------------------------
    def done( self, state):
        return self.get_v_p( state) == (1.0, None)

    # Which leaf to expand.
    # Walk down the tree until the end, using best UCT score.
    #----------------------------------------------------------
    def pick_leaf_to_expand( self):
        node = self.root
        while 1:
            if not node.children: # leaf
                return node
            node = node.get_best_child( self.c_puct)

    # Add children to a leaf, one per possible action.
    #---------------------------------------------------
    def expand_leaf( self, leaf):
        value, policy = self.get_v_p( leaf.state) # Run the net
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
            next_state = self.get_next_state( leaf.state, action_idx=idx)
            new_child = UCTNode( next_state, action=idx, parent=leaf, p=p)
            leaf.children.append( new_child)

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

    #-----------------------------------
    def get_best_child( self, c_puct):
        mmax = -1 * UCTree.LARGE
        winner = None
        for child in self.children:
            score = child.get_uct_score( c_puct)
            if score > mmax:
                mmax = score
                winner = child
        return winner

    # UCT score decides which node gets expanded next.
    # c_puct: How much to rely on hope. Larger means more exploration.
    #-------------------------------------------------------------------
    def get_uct_score( self, c_puct):
        if not self.N: # Leaf
            experience = self.parent.v / self.parent.N
        else:
            experience = self.v / self.N # Our own winrate experience
        hope = self.p * ( math.sqrt(self.parent.N) / (1.0 + self.N) ) # Hope makes us try bad things
        res = experience + c_puct * hope
        return res

    #--------------------------
    def __lt__( self, other):
        return self.p > other.p
