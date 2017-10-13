#!/usr/bin/env python

# /********************************************************************
# Filename: target.py
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
IMG_FOLDER = 'images/19x19_stones'

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Find Go boards in jpeg images
    Synopsis:
      %s --run --img <n> [--squares]
    Description:
      Find Go boards in jpeg images.
      --run:     Must be given to get around the usage
      --img:     Which photo to use from the images folder
      --squares: Show the individual squares on the board. Else show the whole board.
    Example:
      %s
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

    global IMG_FOLDER

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--img", required=False, default=0, type=int)
    parser.add_argument("--squares", required=False, action='store_true')
    parser.add_argument("--run", required=False, action='store_true')
    args = parser.parse_args()

    images = glob.glob(IMG_FOLDER + '/*.jpg')
    images += glob.glob(IMG_FOLDER + '/*.JPG')
    frame = cv2.imread(images[args.img])  #<<<<<<<<<<<<<<<<<<<<<<
    #frame = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)

    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = gray
    edges = ut.auto_canny(gray)

    #ut.showim(blurred,cmap='gray')
    #ut.showim(edges)

    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    fcp = frame.copy()
    cv2.drawContours(fcp, cnts, -1, (0,255,0), 1)
    ut.showim(fcp)

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

    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares), -1, (0,255,0), 2)
    #ut.showim(fcp)

    # Get square centers
    centers = [ (int(M["m10"] / M["m00"]),
                 int(M["m01"] / M["m00"]))
                for M in [cv2.moments(s) for s in squares]]

    # Find center of board
    board_center = np.array (( int(np.median([x[0] for x in centers])),
                               int(np.median([x[1] for x in centers])) ))
    ut.plot_points(fcp,[board_center])
    ut.showim(fcp)

    # Store distance from center for each contour
    sqdists = [ {'cnt':sq, 'dist':np.linalg.norm( centers[idx] - board_center)}
                for idx,sq in enumerate(squares) ]
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

    squares1 = [x['cnt'] for x in distsorted[:lastidx]]
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares1), -1, (0,255,0), 2)
    ut.showim(fcp)

    # Find enclosing 4-polygon
    points = np.array([p for s in squares1 for p in s])
    board = ut.approx_poly( points, 4).reshape(4,2)
    #BP()

    #fcp = frame.copy()
    #cv2.drawContours(fcp, [board], -1, (0,255,0), 1)
    #ut.showim(fcp)

    # Make the board a little larger
    factor = 1.1
    board = ut.order_points(board)
    diag1_stretched = ut.stretch_line( (board[0],board[2]), factor)
    diag2_stretched = ut.stretch_line( (board[1],board[3]), factor)
    board_stretched = np.int0([diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1]])
    #BP()
    fcp = frame.copy()
    cv2.drawContours(fcp, [board_stretched], -1, (0,255,0), 1)
    #ut.showim(fcp)

    # Zoom in on the board
    zoomed = ut.four_point_transform( gray, board_stretched)
    ut.showim(zoomed)

    # DFT on original
    dft = cv2.dft(gray.astype('float32'), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shifted[:,:,0],dft_shifted[:,:,1]))
    #ut.showim(magnitude_spectrum)

    # DFT on zoomed
    dft = cv2.dft(zoomed.astype('float32'), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shifted[:,:,0],dft_shifted[:,:,1]))
    #ut.showim(magnitude_spectrum)

    # Get three rows from the image
    height = zoomed.shape[0]
    width = zoomed.shape[1]
    r_middle = height // 2
    r_lower  = height // 4
    r_upper  = r_middle + r_lower

    # Get the spectrum for each
    CLIP = 5000

    row = zoomed[r_middle,:]
    fft = np.fft.fft(row)
    fft_shifted = np.fft.fftshift(fft)
    magspec_m = np.clip(np.abs(fft_shifted),0,CLIP)

    row = zoomed[r_lower,:]
    fft = np.fft.fft(row)
    fft_shifted = np.fft.fftshift(fft)
    magspec_l = np.clip(np.abs(fft_shifted),0,CLIP)

    row = zoomed[r_upper,:]
    fft = np.fft.fft(row)
    fft_shifted = np.fft.fftshift(fft)
    magspec_u = np.clip(np.abs(fft_shifted),0,CLIP)

    magspec_log  = np.log(np.average( np.abs( np.fft.fftshift( np.fft.fft( zoomed))), axis=0))
    magspec_clip = np.clip(np.average( np.abs( np.fft.fftshift( np.fft.fft( zoomed))), axis=0), 0, CLIP)

    #BP()
    #plt.subplot(121)
    #plt.plot(range( -width // 2, width // 2 ), magspec_log)
    plt.subplot(121)
    plt.plot(range( -width // 2, width // 2 ), magspec_clip)
    smooth_magspec = np.convolve(magspec_clip, np.bartlett(5), 'same')
    plt.subplot(122)
    plt.plot(range( -width // 2, width // 2 ), smooth_magspec)
    # First peak with frequencey > 6 gives the horizontal line distance, d_h = width / f
    highf = smooth_magspec[width // 2 + 6:]
    maxes = scipy.signal.argrelextrema( highf, np.greater)[0] + 6.0
    BP()

    # plt.subplot(131)
    # plt.plot(range( -width // 2, width // 2 ), magspec_m)
    # plt.subplot(132)
    # plt.plot(range( -width // 2, width // 2 ), magspec_l)
    # plt.subplot(133)
    # plt.plot(range( -width // 2, width // 2 ), magspec_u)
    plt.show()

    # # Get Cepstrum for each
    # magceps_m = np.abs(np.fft.fft(magspec_m))
    # plt.subplot(111)
    # plt.plot(magceps_m)
    # magceps_l = np.abs(np.fft.fft(magspec_l))
    # plt.subplot(111)
    # plt.plot(magceps_l)
    # magceps_u = np.abs(np.fft.fft(magspec_u))
    # plt.subplot(111)
    # plt.plot(magceps_u)
    # plt.show()



if __name__ == '__main__':
    main()
