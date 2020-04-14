
'''
Generate shifting puzzle games for reinforcement learning.
Python 3
AHN, Apr 2020

'''

from pdb import set_trace as BP
import argparse
import math, os, glob, json
import numpy as np
from game import Game
from player import Player
from datetime import datetime
import shortuuid
from shiftmodel import ShiftModel
from state import State, StateJsonEncoder

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Generate training data for reinforcement learning
    Synopsis:
      %s --run
    Description:
      Solve increasingly more difficult shifting puzzles using the model in shiftmodel.py .
      Each (state,v) gets saved to the generator.out folder for use by a separate training process.
    Example:
      %s --run
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--run", required=True, action='store_true')
    args = parser.parse_args()
    model = ShiftModel( size=3)
    gen = Generator( model, c_puct=0.016)
    gen.run()

#==================
class Generator:

    def __init__( self, model,
                  weightsfile='generator.h5',
                  folder='generator.out',
                  movelimit=50,
                  chunksize=100,
                  max_shuffles=1000,
                  playouts=128,
                  #playouts=256,
                  c_puct=0.1,
                  maxfiles=20000 ):
        self.model = model # The model used for self-play
        self.weightsfile = weightsfile # Separate training process stores updated weights here
        self.folder = folder # Folder to store generated training data
        self.movelimit = movelimit # Abort game if still no solution
        self.chunksize = chunksize # If we solved this many, increase shuffles
        self.max_shuffles = max_shuffles # Upper limit for shuffles. We start with 1, then increase.
        self.playouts = playouts  # Number of playouts during self play
        self.c_puct = c_puct # Hope factor
        self.maxfiles = maxfiles # Only keep the newest maxfiles training samples
        self.modeltime = datetime.utcnow()

    def run( self):
        print( '>>> Looking for new weights %s every %d games' % (self.weightsfile, self.chunksize))
        print( '>>> Keeping %d newest files in %s' % (self.maxfiles, self.folder))
        if not os.path.isdir( self.folder):
            os.mkdir( self.folder)
        nshuffles = 1
        gameno = 0
        while nshuffles <= self.max_shuffles:
            movelimit = min( self.movelimit, 2*nshuffles)
            # A training process runs independently and will occasionally
            # save a better model.
            self.load_weights_if_newer()
            failures = 0
            for idx in range( self.chunksize):
                gameno += 1
                state = State.random( self.model.size, nshuffles)
                player = Player( state, self.model, self.playouts, self.c_puct)
                g = Game(player)
                seq, found = g.play( movelimit)
                if not found:
                    failures += 1
                    #self.save_steps( seq) #@@@ re-enable once things converge
                    print( 'Game %d   failed' % gameno)
                else:
                    self.save_steps( seq)
                    print( 'Game %d solved' % gameno)

            if failures == 0: # Our excellent model needs a new challenge
                print( '0/%d failures at %d shuffles' % (self.chunksize,nshuffles))
                nshuffles += 1
                print( '>>> %s increasing to %d shuffles' % (datetime.now(), nshuffles))
            else: # we still need to improve
                print( '%d/%d failures at %d shuffles' % (failures, self.chunksize,nshuffles))
                print( 'staying at %d shuffles' % nshuffles)

            self.delete_old_files()

    def load_weights_if_newer( self):
        modtime = datetime.utcfromtimestamp( os.path.getmtime( self.weightsfile))
        if modtime > self.modeltime:
            self.modeltime = modtime
            self.model.load_weights( self.weightsfile)
            print( '>>> %s loaded new weights file' % datetime.now())

    def save_steps( self, seq):
        'Save individual solution steps as training samples'
        for idx,step in enumerate(seq):
            dist = int(round(State.dist_from_v( step['v'])))
            if dist > 50:
                BP()
                tt=42
            if dist == 0: # solution, no need to train
                continue
            fname = self.folder + '/%d_%04d_%s.json' % (self.model.size, dist, shortuuid.uuid()[:8])
            step = { 'state':step['state'], 'v':step['v'], 'dist':dist }
            jsn = json.dumps( step, cls=StateJsonEncoder)
            with open(fname,'w') as f:
                f.write(jsn)

    def delete_old_files( self):
        'Only keep the newest maxfiles training examples around'
        files = glob.glob("%s/*.json" % self.folder)
        if len(files) < self.maxfiles: return
        files.sort( key=os.path.getmtime)
        delfiles = files[:len(files)-self.maxfiles]
        print( 'Deleting %d old files, leaving %d' % (len(delfiles),self.maxfiles))
        for f in delfiles:
            os.remove( f)


if __name__ == '__main__':
    main()
