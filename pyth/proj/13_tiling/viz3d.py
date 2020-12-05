#!/usr/bin/env python

# Visualize 3x3 tiling puzzles
# AHN, Dec 2020

from pdb import set_trace as BP
import sys,os
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

g_pieces = { # xyz = right back top = ribato
    '3x3x3':
    [
        # The Waiter
        np.array([
            [
                # Rear plane
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ],
            [
                [0,0,0],
                [1,0,0],
                [0,0,0]
            ],
            [   # Front plane
                [1,0,0],
                [1,0,0],
                [1,1,0]
            ]
        ]),
        # The Pillar
        np.array([
            [
                [0,0,0],
                [1,0,0],
                [1,1,1]
            ],
            [
                [0,0,0],
                [0,0,0],
                [1,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ]
        ]),
        # The Cobra
        np.array([
            [
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,1,1]
            ],
            [
                [0,0,1],
                [0,0,1],
                [0,0,1]
            ]
        ]),
        # The Kiss
        np.array([
            [
                [0,1,1],
                [0,0,0],
                [0,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ]
        ]),
        # The L-wiggle
        np.array([
            [
                [1,0,0],
                [0,0,0],
                [0,0,0]
            ],
            [
                [1,1,1],
                [0,0,1],
                [0,0,0]
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ]
        ]),
        # The Mirror-wiggle
        np.array([
            [
                [0,0,0],
                [0,1,1],
                [0,0,0]
            ],
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ],
            [
                [0,1,0],
                [0,1,0],
                [0,0,0]
            ]
        ])
    ]
} # g_pieces

# Turn the ones into piece number
for k in g_pieces.keys():
    g_pieces[k] = [p * (idx + 1) for idx,p in enumerate( g_pieces[k])]

#-----------------------------
def usage( printmsg=False):
    name = os.path.basename( __file__)
    msg = '''

    Description:
      %s: Visuzlize 3D tiling puzzles
    Synopsis:
      %s --case <case_id>
    Example:
      %s --case waiter

--
''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------------
def main():
    if len(sys.argv) == 1: usage( True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--case")
    args = parser.parse_args()
    visualize( g_pieces[args.case])
    plt.show()

#------------------------
def visualize( pieces):
    FIG = plt.figure( figsize=(8,8))
    ax_cube = FIG.add_axes( [0.05, 0.05, 0.8, 0.8], projection='3d')

    # Add button
    ax_add = FIG.add_axes( [0.05, 0.90, 0.1, 0.05] )
    btn_add = Button( ax_add, 'Add')
    btn_add.on_clicked( cb_btn_add)

    # Remove button
    ax_remove = FIG.add_axes( [0.20, 0.90, 0.1, 0.05] )
    btn_remove = Button( ax_remove, 'Remove')
    btn_remove.on_clicked( cb_btn_remove)

    #fig = plt.figure()
    #ax = fig.gca( projection='3d')
    ax_cube.set_aspect('auto')
    #plt.axis( 'off')
    # Manually fix aspect ratio. Uggh.
    ax_cube.set_xlim3d(-0.5, 5.7)
    ax_cube.set_ylim3d(-0.5, 5.7)
    ax_cube.set_zlim3d(-0.5, 4.5)

    ax_cube.set_xlabel( "x")
    ax_cube.set_ylabel( "y")
    ax_cube.set_zlabel( "z")
    ax_cube.grid( False)
    for idx,p in enumerate(pieces):
        cmap = plt.get_cmap('Set1')
        color = cmap.colors[idx]
        viz_piece( p, ax_cube, color)

#----------------------------
def cb_btn_add( event):
    pass

#----------------------------
def cb_btn_remove( event):
    pass

#-----------------------------
def viz_piece( p, ax, color):
    # Rotate to the non-intuitive dimension order voxels wants
    p = np.rot90( p, 1, axes=(2,1))
    p = np.rot90( p, 1, axes=(1,0))
    ax.voxels( p, facecolors=color,edgecolors='gray', shade=True)

main()
