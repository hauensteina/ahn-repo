#!/usr/bin/env python

# Store a limited amount of training states and targets in a buffer.
# Older ones age out on overflow.
# Retrieve randomly selected batches from the buffer for training.
# Python 3
# AHN, Mar 2020

from pdb import set_trace as BP
import os
import numpy as np

#====================================
class ExperienceBuf:
    #------------------------------------
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = []

    # Remember [state, target]
    # State and target have already been encoded into numpy arrays.
    #----------------------------------------------------------------
    def remember(self, encoded_state, target):
        self.memory.append((encoded_state, target))
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Get a random batch of (state,target) pairs for training.
    #-----------------------------------------------------------
    def get_batch(self, model, batch_size=10):
        memlen = len(self.memory)
        batch_size = min(memlen, batch_size)
        input_shape = self.memory[0][0].shape
        target_shape = self.memory[0][1].shape
        inputs = np.zeros( (batch_size,) + input_shape, float )
        targets = np.zeros( batch_size, float )
        for i, rrandom in enumerate( np.random.randint(0, memlen, size=batch_size)):
            cur_input  = self.memory[rrandom][0]
            cur_target = self.memory[rrandom][1]
            #print( cur_input)
            #print( cur_target)
            inputs[i] = cur_input
            targets[i] = cur_target
        return inputs, targets
