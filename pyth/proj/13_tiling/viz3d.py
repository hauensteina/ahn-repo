#!/usr/bin/env python

# Visualize 3D tiling puzzles
# AHN, Dec 2020

from pdb import set_trace as BP
import sys,os
import pickle
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import tiling_helpers as helpers

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
      %s --file <fname.pickle> --nth <nth solution>
    Examples:
      %s --case 3x3x3
      %s --file algo_x_3d_solutions.pickle --nth 1

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
    parser.add_argument( "--case")
    parser.add_argument( "--file")
    parser.add_argument( "--nth", type=int, default=1)
    args = parser.parse_args()
    if args.case:
        visualize( g_pieces[args.case])
    else:
        with open( args.file, 'rb') as f:
            solutions = pickle.load( f)
            solution = solutions[args.nth - 1]
            visualize( solution)

    plt.show()

#------------------------
def visualize( pieces):
    # Sort pieces by shape to have consistent colors.
    #pieces = sorted( pieces, key = lambda x: helpers.hash_piece_image_3d(x))
    pieces = sorted( pieces, key = helpers.hash_piece_image_3d)
    w = 12.5
    h = 8.0
    S = pieces[0].shape[0]
    space = 0.05
    marg = 0.05
    cube_s = 0.5
    cmap = plt.get_cmap('tab20')
    fig = plt.figure( figsize=(w,h))
    assembled = [False] * len(pieces)

    def cb_click_piece( event):
        ' Click on a piece axes'
        try:
            piece_idx = ax_pieces.index( event.artist)
            p = pieces[piece_idx]
        except:
            return
        if not assembled[piece_idx]:
            color = cmap.colors[piece_idx]
            artist_list = viz_piece_in_cube( p, ax_cube, color)
            assembled[piece_idx] = artist_list
        else:
            artist_list = assembled[piece_idx].values()
            for a in artist_list:
                a.remove()
            assembled[piece_idx] = False

        fig.canvas.draw()

    fig.canvas.mpl_connect( 'pick_event', cb_click_piece)

    # The whole cube
    cube_x = marg * h/w
    cube_width = cube_s * h/w
    ax_cube = fig.add_axes( [cube_x, 4.5 * marg, cube_width, cube_s], projection='3d')
    ax_cube.set_xlabel( "x")
    ax_cube.set_ylabel( "y")
    ax_cube.set_zlabel( "z")
    #ax_cube.grid( False)
    plt.axis( 'off')
    # A 4x4 grid for the individual pieces
    ax_pieces = []
    rowheight = (1.0 - 2*marg - 3*space) / 4
    colwidth = rowheight * h/w
    piece_grid_sz = 4
    for r in range( piece_grid_sz):
        r = piece_grid_sz - r - 1 # mpl y goes from bottom to top => flip
        y = marg + r * rowheight
        if r: y += (r) * space
        for c in range( piece_grid_sz):
            x = cube_x + cube_width + marg * h/w + c * colwidth # h/w * 0.8 * cube_s + c * colwidth
            if c: x += (c) * space * h/w
            # Setting picker arg to a tolerance margin enables pick event on axes
            ax = fig.add_axes( [x, y, colwidth, rowheight], projection='3d', picker=5)
            plt.axis( 'off')
            #ax.grid( False)
            ax_pieces.append(ax)


    # # Add button
    # ax_add = FIG.add_axes( [0.05, 0.90, 0.1, 0.05] )
    # btn_add = Button( ax_add, 'Add')
    # btn_add.on_clicked( cb_btn_add)

    # # Remove button
    # ax_remove = FIG.add_axes( [0.20, 0.90, 0.1, 0.05] )
    # btn_remove = Button( ax_remove, 'Remove')
    # btn_remove.on_clicked( cb_btn_remove)

    # The individual pieces
    for idx,p in enumerate(pieces):
        color = cmap.colors[idx]
        if assembled[idx]:
            viz_piece_in_cube( p, ax_cube, color)
        else:
            viz_piece( p, ax_pieces[idx], color)


#-----------------------------
def viz_piece( p, ax, color):
    # Rotate to the non-intuitive dimension order voxels wants
    p = np.rot90( p, 1, axes=(2,1))
    p = np.rot90( p, 1, axes=(1,0))
    res = ax.voxels( p, facecolors=color,edgecolors='gray', shade=True)
    return res

#-----------------------------------------
def viz_piece_in_cube( p, ax_cube, color):
    # Rotate to the non-intuitive dimension order voxels wants
    p = np.rot90( p, 1, axes=(2,1))
    p = np.rot90( p, 1, axes=(1,0))

    S = p.shape[0]
    trans = np.indices( (S+1,S+1,S+1)) * 0.0
    trans -= S / 2
    x,y,z = np.indices( (S+1,S+1,S+1)) + trans
    res = ax_cube.voxels( x,y,z, p, facecolors=color,edgecolors='gray', shade=True)

    if viz_piece_in_cube.firstcall:
        viz_piece_in_cube.firstcall = False
        zlim = ax_cube.get_zlim()
        scale = 0.93
        ax_cube.set_zlim( zlim[0] * scale, zlim[1] * scale )

    return res

viz_piece_in_cube.firstcall = True


main()
