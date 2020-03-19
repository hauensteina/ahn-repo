#!/bin/env python

# A class for A* search.
# Look in astarshift.py to see how it solves a sliding puzzle.
# Python 3
# AHN, Mar 2020

from pdb import set_trace as BP

# Example usage:
# state = initial_state
# astar = Astar( state)
# while not astar.done:
#   astar.search()
# solution = astar.get_solution()
# for state in solution:
#    print( state)

# Your state class needs methods get_actions() and act().
# If you want to print states, you'll need __repr__() .

#==============
class Astar:
    LARGE = 1E9

    # state: The current position.
    #----------------------------------
    def __init__( self, state):
        self.root = AstarNode( state)
        self.leaves = set( [self.root])
        self.done = False
        self.best_leaf_estimate = None
        self.best_leaf_depth = None
        self.best_distance = None
        self.treesize = 1

    # Find the best leaf and expand it.
    #-------------------------------------
    def search( self):
        leaf = self.get_best_leaf()
        if leaf.state.estimate == 0: # solution
            self.done = True
            return
        self.expand_leaf( leaf)

    #--------------------------
    def get_best_leaf( self):
        LAMBDA = 1.0
        winner = None
        self.best_leaf_estimate = Astar.LARGE
        for i,leaf in enumerate(self.leaves):
            est = leaf.state.estimate + LAMBDA*leaf.depth
            if est < self.best_leaf_estimate:
                self.best_leaf_estimate = est
                self.best_leaf_depth = leaf.depth
                self.best_leaf_distance = leaf.state.estimate
                winner = leaf
        return winner

    # Add children to a leaf, one per possible action.
    #---------------------------------------------------
    def expand_leaf( self, leaf):
        self.leaves.remove( leaf)
        # Create a child for each action
        leaf.children = []
        for action in leaf.state.get_actions():
            next_state = leaf.state.act( action)
            new_child = AstarNode( next_state, action, parent=leaf, depth = leaf.depth + 1)
            leaf.children.append( new_child)
            self.treesize += 1
            self.leaves.add( new_child)

    #--------------------------
    def get_solution( self):
        res = []
        node = self.get_best_leaf()
        while 1:
            res.append( node)
            node = node.parent
            if not node: break
        res.reverse()
        return res


#================
class AstarNode:
    # state: State at this node.
    # action: What action got us here.
    # parent: Parent node.
    #---------------------------------------------------------------
    def __init__( self, state, action=None, parent=None, depth=0):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.depth = depth

    #---------------------
    def __repr__( self):
        res = self.state.__repr__()
        res += '\ndepth: %d\n' % self.depth
        res += '\nnum children: %d\n' % (len(self.children) if self.children else 0)
        return res
