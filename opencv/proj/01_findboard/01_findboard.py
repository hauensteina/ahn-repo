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
g_frame=''

np.set_printoptions(linewidth=np.nan)


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
    #width  = frame.shape[1]
    #height = frame.shape[0]
    frame = ut.resize( frame, 500)
    global g_frame
    g_frame = frame

    #----------------------------
    #gray = cv2.equalizeHist( cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ut.showim(gray,'gray')

    cnts = get_contours(gray)
    #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    fcp = frame.copy()
    cv2.drawContours(fcp, cnts, -1, (0,255,0), 1)
    #ut.showim(fcp)

    #--------------------------------
    squares = filter_squares(cnts, frame.shape[1], frame.shape[0])
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares), -1, (0,255,0), 2)
    #ut.showim(fcp)

    #--------------------------------------------
    centers, board_center = get_board_center(squares)
    ut.plot_points(fcp,[board_center])
    #ut.showim(fcp)

    #----------------------------------------------------
    squares1 = cleanup_squares( centers, squares, board_center, frame.shape[1], frame.shape[0])
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares1), -1, (0,255,0), 2)
    #ut.showim(fcp)

    # Find enclosing 4-polygon. That's the board.
    #-----------------------------------------------------
    points = np.array([p for s in squares1 for p in s])
    board = ut.order_points( ut.approx_poly( points, 4).reshape(4,2)).astype('int')
    fcp = frame.copy()
    cv2.drawContours(fcp, [board], -1, (0,255,0), 1)
    #ut.showim(fcp)

    #---------------------------------------
    board_stretched = enlarge_board(board)
    fcp = frame.copy()
    cv2.drawContours(fcp, [board_stretched], -1, (0,255,0), 1)
    #ut.showim(fcp)

    # Zoom in on the board
    #----------------------
    zoomed,M = ut.four_point_transform( gray, board_stretched)
    ut.showim(zoomed,'gray')

    # Coordinates of board corners after the transform
    #----------------------------------------------------
    # This needs a stupid empty dimension added
    board_zoomed = cv2.perspectiveTransform(board.reshape(1,4,2).astype('float32'),M)
    # And now get rid of the extra dim and back to int
    board_zoomed = board_zoomed.reshape(4,2).astype('int')

    # Get board size (9, 13, 19)
    #-----------------------------
    boardsize = get_boardsize_by_fft( zoomed)
    print('Board size: %dx%d' % (boardsize,boardsize))


    # Compute lines on the board
    #-----------------------------
    tl,tr,br,bl = board_zoomed

    left_x   = np.linspace( tl[0], bl[0], boardsize)
    left_y   = np.linspace( tl[1], bl[1], boardsize)
    right_x  = np.linspace( tr[0], br[0], boardsize)
    right_y  = np.linspace( tr[1], br[1], boardsize)
    left_points =  np.array(zip(left_x, left_y)).astype('int')
    right_points = np.array(zip(right_x, right_y)).astype('int')
    h_lines = zip(left_points, right_points)
    # fcp = zoomed.copy()
    # ut.plot_lines( fcp, h_lines)
    # ut.showim(fcp)

    top_x   = np.linspace( tl[0], tr[0], boardsize)
    top_y   = np.linspace( tl[1], tr[1], boardsize)
    bottom_x  = np.linspace( bl[0], br[0], boardsize)
    bottom_y  = np.linspace( bl[1], br[1], boardsize)
    top_points =  np.array(zip(top_x, top_y)).astype('int')
    bottom_points = np.array(zip(bottom_x, bottom_y)).astype('int')
    v_lines = zip(top_points, bottom_points)
    # fcp = zoomed.copy()
    # ut.plot_lines( fcp, v_lines)
    # ut.showim(fcp)

    # Compute intersections of the lines
    #-------------------------------------
    intersections = np.array([ ut.intersection( hl, vl) for hl in h_lines for vl in v_lines]).astype('int')
    fcp = zoomed.copy()
    ut.plot_points( fcp, intersections)
    ut.showim(fcp)

    # Contour image of the zoomed board
    #-------------------------------------
    zoomed_edges = ut.auto_canny(zoomed)
    zoomed_contours, cnts, hierarchy  = cv2.findContours(zoomed_edges, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)
    #ut.showim(zoomed_contours)

    # Cut out areas around the intersections
    #------------------------------------------
    delta_v = abs(int(np.round( 0.5 * (bottom_y[0] - top_y[0]) / (boardsize -1))))
    delta_h = abs(int(np.round( 0.5 * (right_x[0] - left_x[0]) / (boardsize -1))))
    brightness  = np.empty(boardsize * boardsize)
    darkest  = np.empty(boardsize * boardsize)
    brightest  = np.empty(boardsize * boardsize)
    crossness = np.empty(boardsize * boardsize)
    thresh = np.mean( sorted(zoomed.flatten(), reverse=True)[:10])
    for i,p in enumerate(intersections):
        hood = zoomed[p[1]-delta_v:p[1]+delta_v, p[0]-delta_h:p[0]+delta_h ]
        contour_hood = zoomed_contours[p[1]-delta_v:p[1]+delta_v, p[0]-delta_h:p[0]+delta_h ]
        brightness[i] = get_brightness(hood)
        crossness[i] = get_brightness(contour_hood,6)

    # Black stones
    print('brightness')
    print(brightness.reshape((boardsize,boardsize)).astype('int'))
    thresh = 4 * min(brightness)
    isblack = np.array([ 1 if x < thresh else 0 for x in brightness ])

    # Empty intersections
    print('crossness')
    print(crossness.reshape((boardsize,boardsize)).astype('int'))
    isempty = np.array([ 1 if x > 5 else 0 for x in crossness ])

    # White stones
    iswhite = np.array([ 2 if not isempty[i] + isblack[i] else 0  for i,x in enumerate( isempty) ])

    print('position')
    position = iswhite + isblack
    print(position.reshape((boardsize,boardsize)))

    # Draw zoomed board with markers on detected stones
    fcp = zoomed.copy()
    black_points = []
    white_points = []
    for i,x in enumerate(isblack):
        if x: black_points.append(intersections[i])
    for i,x in enumerate(iswhite):
        if x: white_points.append(intersections[i])
    ut.plot_points( fcp, black_points, 255)
    ut.plot_points( fcp, white_points, 0)
    ut.showim(fcp,'gray')


# Get a center crop of an image
#-------------------------------
def get_center_crop(img, frac=4):
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2
    dx = img.shape[0] // frac
    dy = img.shape[1] // frac
    res =  img[cx-dx:cx+dx, cy-dy:cy+dy]
    area = res.shape[0] * res.shape[1]
    return res, area

# Sum brightness at the center, normalize
#-------------------------------------------
def get_brightness( img, frac=4):
    crop, area =  get_center_crop(img, frac)
    ssum = np.sum( crop)
    return ssum / area


#-----------------------
def get_contours(img):
    #img   = cv2.GaussianBlur( img, (7, 7), 0)
    #img   = cv2.medianBlur( img, 7)
    #img = cv2.bilateralFilter(img, 11, 17, 17)
    #edges = cv2.Canny(img, 60, 200)
    edges = ut.auto_canny(img)
    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)
    #cnts = sorted( cnts, key=cv2.contourArea, reverse=True)
    #ut.show_contours( img, cnts)

    # Keep if larger than 0.1% of image
    img_area = img.shape[0] * img.shape[1]
    cnts = [ c for c in cnts if  cv2.contourArea(c) / img_area > 0.001 ]
    # Keep if reasonably not-wiggly
    #cnts = [ c for c in cnts if   cv2.arcLength(c, closed=True) / len(c) > 2.0 ]
    cnts = [ c for c in cnts if   len(c) < 100 or cv2.arcLength(c, closed=True) / len(c) > 2.0 ]

    return cnts


# Try to eliminate all contours except those on the board
#----------------------------------------------------------
def filter_squares(cnts, width, height):
    squares = []
    for i,c in enumerate(cnts):
        area = cv2.contourArea(c)
        #if area > width*height / 2.5: continue
        if area < width*height / 4000.0 : continue
        peri = cv2.arcLength(c, closed=True)
        hullArea = cv2.contourArea(cv2.convexHull(c))
        if hullArea < 0.001: continue
        solidity = area / float(hullArea)
        approx = cv2.approxPolyDP(c, 0.01 * peri, closed=True)
        #if len(approx) < 4: continue  # Not a square
        # not a circle
        #if len(approx) > 6:
        #center,rad = cv2.minEnclosingCircle(c)
        #circularity = area / (rad * rad * np.pi)
        #if circularity < 0.50: continue
        #print (circularity)

        #if len(approx) > 14: continue
        #(x, y, w, h) = cv2.boundingRect(approx)
        #aspectRatio = w / float(h)
        #arlim = 0.4
        #if aspectRatio < arlim: continue
        #if aspectRatio > 1.0 / arlim: continue
        #if solidity < 0.45: continue
        #if solidity < 0.07: continue
        squares.append(c)
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
def cleanup_squares(centers, square_cnts, board_center, width, height):
    # Store distance from center for each contour
    # sqdists = [ {'cnt':sq, 'dist':np.linalg.norm( centers[idx] - board_center)}
    #             for idx,sq in enumerate(square_cnts) ]
    # distsorted = sorted( sqdists, key = lambda x: x['dist'])

    #ut.show_contours( g_frame, square_cnts)
    sqdists = [ {'cnt':sq, 'dist':ut.contour_maxdist( sq, board_center)}
                 for sq in square_cnts ]
    distsorted = sorted( sqdists, key = lambda x: x['dist'])

    # Remove contours if there is a jump in distance
    lastidx = len(distsorted)
    for idx,c in enumerate(distsorted):
        if not idx: continue
        delta = c['dist'] - distsorted[idx-1]['dist']
        #print(c['dist'], delta)
        #ut.show_contours( g_frame, [c['cnt']])
        #print ('dist:%f delta: %f' % (c['dist'],delta))
        if delta > min(width,height) / 10.0:
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
    smooth_magspec = np.convolve(magspec_clip, np.bartlett(7), 'same')
    if not len(smooth_magspec) % 2:
        smooth_magspec = np.append( smooth_magspec, 0.0)
    # The first frequency peak above 9 should be close to the board size.
    half = len(smooth_magspec) // 2
    #plt.subplot(111)
    #plt.plot(range( -half, half+1 ), smooth_magspec)
    #plt.show()
    MINSZ = 9
    highf = smooth_magspec[width // 2 + MINSZ:]
    maxes = scipy.signal.argrelextrema( highf, np.greater)[0] + MINSZ
    res = maxes[0] if len(maxes) else 0
    print(res)
    if res > 19: res = 19
    #elif res > 13: res = 13
    else: res = 9
    return res

if __name__ == '__main__':
    main()
