#!/usr/bin/env python

# /********************************************************************
# Filename: 01_findboard.py
# Author: AHN
# Creation Date: Oct 6, 2017
# **********************************************************************/
#
# Try to detect go boards from jpg images, then find the stones
#

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json,shutil,glob
import numpy as np
import scipy.signal
from numpy.random import random
import argparse
import cv2
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahn_opencv_util as ut

#IMG_FOLDER = 'images/9x9_empty'
#IMG_FOLDER = 'images/9x9_stones'
#IMG_FOLDER = 'images/19x19_stones'

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate sgf files from photos of Go boards
    Synopsis:
      %s --img <n> --folder <folder>
    Description:
      Find Go boards in jpeg images.
      --folder:  Path to image folder
      --img:     Which photo to use from the images folder (an integer >= 0)
    Example:
      %s --folder images/9x9_empty --img 0
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#---------------------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--img",    required=False, default=0, type=int)
    parser.add_argument("--folder", required=True, type=str)
    #parser.add_argument("--squares", required=False, action='store_true')
    #parser.add_argument("--run", required=False, action='store_true')
    args = parser.parse_args()

    images = glob.glob(args.folder + '/*.jpg')
    images += glob.glob(args.folder + '/*.JPG')
    frame = cv2.imread(images[args.img])
    #frame = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)

    #----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cnts = get_contours(gray)
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    fcp = frame.copy()
    cv2.drawContours(fcp, cnts, -1, (0,255,0), 1)
    ut.showim(fcp)

    #--------------------------------
    squares = filter_squares(cnts)
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares), -1, (0,255,0), 2)
    #ut.showim(fcp)

    #--------------------------------------------
    centers, board_center = get_board_center(squares)
    ut.plot_points(fcp,[board_center])
    ut.showim(fcp)

    #----------------------------------------------------
    squares1 = cleanup_squares( centers, squares, board_center)
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares1), -1, (0,255,0), 2)
    ut.showim(fcp)

    # Find enclosing 4-polygon. That's the board.
    #-----------------------------------------------------
    points = np.array([p for s in squares1 for p in s])
    board = ut.approx_poly( points, 4).reshape(4,2)

    #---------------------------------------
    board_stretched = enlarge_board(board)
    #fcp = frame.copy()
    #cv2.drawContours(fcp, [board_stretched], -1, (0,255,0), 1)
    #ut.showim(fcp)

    # Zoom in on the board
    #----------------------
    zoomed = ut.four_point_transform( gray, board_stretched)
    ut.showim(zoomed)

    #---------------------------------------
    boardsize = get_boardsize_by_fft(zoomed)
    print( "Board size: %d" % boardsize)


#-----------------------
def get_contours(img):
    # convert the frame to grayscale, blur it, and detect edges
    #blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = ut.auto_canny(img)

    #ut.showim(blurred,cmap='gray')
    #ut.showim(edges)

    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)
    return cnts


# Try to eliminate all contours except those on the board
#----------------------------------------------------------
def filter_squares(cnts):
    squares = []
    for i,c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 1000: continue
        if area < 10: continue
        peri = cv2.arcLength(c, closed=True)
        hullArea = cv2.contourArea(cv2.convexHull(c))
        solidity = area / float(hullArea)
        approx = cv2.approxPolyDP(c, 0.01 * peri, closed=True)
        if len(approx) < 4: continue
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)
        arlim = 0.4
        if aspectRatio < arlim: continue
        if aspectRatio > 1.0 / arlim: continue
        if solidity < 0.45: continue
        squares.append(approx)
    return squares

#-----------------------------------
def get_board_center(square_cnts):
    # Get square centers
    centers = [ (int(M["m10"] / M["m00"]),
                 int(M["m01"] / M["m00"]))
                for M in [cv2.moments(s) for s in square_cnts]]

    # Find center of board
    board_center = np.array (( int(np.median([x[0] for x in centers])),
                               int(np.median([x[1] for x in centers])) ))
    return centers, board_center

# Remove spurious contours outside the board
#--------------------------------------------------
def cleanup_squares(centers, square_cnts, board_center):
    # Store distance from center for each contour
    sqdists = [ {'cnt':sq, 'dist':np.linalg.norm( centers[idx] - board_center)}
                for idx,sq in enumerate(square_cnts) ]
    distsorted = sorted( sqdists, key = lambda x: x['dist'])

    # Remove contours if there is a jump in distance from center
    lastidx = len(distsorted)
    for idx,c in enumerate(distsorted):
        if not idx: continue
        delta = c['dist'] - distsorted[idx-1]['dist']
        #print ('dist:%f delta: %f' % (c['dist'],delta))
        if delta > 20.0:
            lastidx = idx
            break

    res = [x['cnt'] for x in distsorted[:lastidx]]
    return res

#---------------------------
def enlarge_board(board):
    factor = 1.1
    board = ut.order_points(board)
    diag1_stretched = ut.stretch_line( (board[0],board[2]), factor)
    diag2_stretched = ut.stretch_line( (board[1],board[3]), factor)
    res = np.int0([diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1]])
    return res

#---------------------------------------
def get_boardsize_by_fft(zoomed_img):
    CLIP = 5000
    width = zoomed_img.shape[1]
    # 1D fft per row, magnitude per row, average them all into a 1D array, clip
    magspec_clip = np.clip(np.average( np.abs( np.fft.fftshift( np.fft.fft( zoomed_img))), axis=0), 0, CLIP)
    # Smooth it
    smooth_magspec = np.convolve(magspec_clip, np.bartlett(5), 'same')
    # The first frequency peak above 6 should be close to the board size.
    plt.subplot(111)
    plt.plot(range( -width // 2, 1 + width // 2 ), smooth_magspec)
    plt.show()
    highf = smooth_magspec[width // 2 + 6:]
    maxes = scipy.signal.argrelextrema( highf, np.greater)[0] + 6.0
    res = maxes[0] if len(maxes) else 0
    return res


if __name__ == '__main__':
    main()
