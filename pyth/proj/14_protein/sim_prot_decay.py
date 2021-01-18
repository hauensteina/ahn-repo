#!/usr/bin/env python

from pdb import set_trace as BP
import sys,os,json
import argparse
import numpy as np
from random import random as rand01

#-----------------------------
def usage( printmsg=False):
    name = os.path.basename( __file__)
    msg = '''

    Name:
      %s: Simulate the decay of a protein where chains are attched to a core
    Synopsis:
      %s --json <file>
      %s --test
    Description:
      --json <fname.json>
      Read the protein description and simulation parameters from the specified file.
      --test
      Run a simple test case.

      Example json:

        {
        "comment": "Sizes as number of amino acids",
        "core_size": 400,
        "chain_sizes": [400, 600],
        "comment": "Halflives in seconds",
        "chain_halflives": [1E8, 1E8],
        "comment": "Set break_halflives to -1 if there are no breaking points",
        "break_halflives": [604800, 604800],
        "comment": "Histogram bucket borders for size of decayed pieces containing a core",
        "buckets":[401, 801, 1001],
        "comment": "Length of one time step in seconds",
        "dt": 600,
        "comment": "Total duration of simulated time in seconds",
        "T": 604800,
        "comment": "Number of simulated molecules",
        "N": 10000
        }

    Examples:
      %s --test
      %s --json with_bpoints.json

--
''' % (name,name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------------
def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--json")
    parser.add_argument( "--test", action='store_true')
    args = parser.parse_args()

    if args.test:
        unittest()
        exit(0)

    if not args.json:
        usage( True)

    parms = json.load( open( args.json))
    simparms = { k:parms[k] for k in ['core_size', 'chain_sizes', 'chain_halflives', 'break_halflives', 'dt', 'T', 'N'] }
    print()
    print( 'Simulation parameters:')
    print( simparms)
    print()
    decay_products = simulate( **simparms)
    weights = [ sum( x['chain_sizes']) + x['core_size'] for x in decay_products ]
    bins = parms['buckets']
    total_size = parms['core_size'] + sum( parms['chain_sizes'])
    bins = [0] + bins + [total_size+1]
    histo = np.histogram( weights, bins=bins)
    print_histo( histo)

#--------------------------
def print_histo( histo):
    bins = histo[1]
    counts = histo[0]
    print()
    for idx,c in enumerate( counts):
        print( 'Found %d molecules with count >= %d and < %d' % (c, bins[idx], bins[idx+1]))

#----------------------------------------------------------------------------------
def simulate( core_size, chain_sizes, chain_halflives, break_halflives, dt, T, N):
    print( 'Letting %d molecules decay for %d seconds in %d second increments' % (N,T,dt))
    print()
    n_chains = len( chain_sizes)
    chain_survival_probs = [ 0.5 ** ( dt / ch ) if ch > 0 else 1.0 for ch in chain_halflives ]
    break_survival_probs = [ 0.5 ** ( dt / bh ) if bh > 0 else 1.0 for bh in break_halflives ]
    decay_products = []
    for sim_idx in range( N):
        if sim_idx % (N // 10) == 0:
            print( '%d / %d' % (sim_idx, N))
        # Get ourselves a fresh molecule
        molecule = { 'core_size': core_size, 'chain_sizes': chain_sizes[:] }
        for tick, t in enumerate( range( 0, T, dt)):
            # See if any chain broke off completely
            for cidx, chain in enumerate( range( n_chains)):
                if rand01() > break_survival_probs[cidx]: # chain breaks off
                    molecule['chain_sizes'][cidx] = 0
            # See if any chain got shorter
            for cidx, chain in enumerate( range( n_chains)):
                # Lower amino_idx means closer to the core
                for amino_idx in range( molecule['chain_sizes'][cidx]):
                    if rand01() > chain_survival_probs[cidx]: # chain breaks off at amino_idx
                        molecule['chain_sizes'][cidx] = amino_idx
                        break # No need to check further out

        decay_products.append( molecule)
    return decay_products

#----------------
def unittest():
    # Test breakpoint halflife
    parms = {
        'core_size': 10,
        'chain_sizes': [100],
        'chain_halflives': [-1],
        'break_halflives': [100],
        'dt': 10,
        'T': 100,
        'N': 1000
    }
    decay_products = simulate( **parms)
    avg_chain_len = np.mean( [ x['chain_sizes'][0] for x in decay_products ])
    if abs( avg_chain_len - 50) > 5:
        print( 'Test 1 failed: avg_chain_len should be about 50, was %d ' % avg_chain_len)
    else:
        print( 'Test 1 successful')
    print()

    # What if we change dt
    parms = {
        'core_size': 10,
        'chain_sizes': [100],
        'chain_halflives': [-1],
        'break_halflives': [100],
        'dt': 3,
        'T': 100,
        'N': 1000
    }
    decay_products = simulate( **parms)
    avg_chain_len = np.mean( [ x['chain_sizes'][0] for x in decay_products ])
    if abs( avg_chain_len - 50) > 5:
        print( 'Test 2 failed: avg_chain_len should be about 50, was %d ' % avg_chain_len)
    else:
        print( 'Test 2 successful')
    print()

    # Test chain halflife and histogram
    parms = {
        'core_size': 0,
        'chain_sizes': [2],
        'chain_halflives': [100],
        'break_halflives': [-1],
        'dt': 1,
        'T': 100,
        'N': 1000
    }
    decay_products = simulate( **parms)
    weights = [x['chain_sizes'][0] for x in decay_products]
    histo = np.histogram( weights, bins = [0,1,2,3])
    if sum( np.abs( histo[0] - [500,250,250]) > [500 * 0.1, 250 * 0.1, 250 * 0.1]) > 0:
        print( 'Test 3 failed: Histo should be about [500,250,250], was %d %d %d' % (histo[0][0], histo[0][1], histo[0][2]))
    else:
        print( 'Test 3 successful')
    print()

if __name__ == '__main__':
    main()
