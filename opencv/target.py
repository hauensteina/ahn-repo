#!/usr/bin/env python

# /********************************************************************
# Filename: target.py
# Author: AHN
# Creation Date: Oct 6, 2017
# **********************************************************************/
#
# Try to detect go boards from jpg images.
# Inspired by Adrian Rosenbrok
# https://www.pyimagesearch.com/2015/05/04/\
#    target-acquired-finding-targets-in-drone-and-quadcopter-video-streams-using-python-and-opencv/


from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json,shutil,glob
import numpy as np
import scipy
from numpy.random import random
import argparse
import cv2
# import matplotlib as mpl
# mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt
# import keras.layers as kl
# import keras.layers.merge as klm
# import keras.models as km
# import keras.optimizers as kopt
# import keras.activations as ka
# import keras.backend as K
# import theano as th

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
#import ahnutil as ut

IMG_FOLDER = 'images/9x9_empty'

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

# Automatic edge detection without parameters
#-----------------------------------
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# Mark points on an image
#----------------------------
def plot_points(img, points, color=(255,0,0)):
    for p in points:
        cv2.circle(img, (p[0],p[1]), 5, color, thickness=-1) #, lineType=8, shift=0)

# Draw lines on an image
#----------------------------
def plot_lines(img, lines, color=(255,0,0)):
    for  p1,p2 in lines:
        #BP()
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=2) #, lineType=8, shift=0)

# Intersection of two lines
#-------------------------------
def intersection(line1, line2):
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    L1 = line(line1[0], line1[1])
    L2 = line(line2[0], line2[1])
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

#----------------------
def linelen(line):
    p1 = line[0]; p2 = line[1]
    return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Display an image
#-------------------------
def showim(img,cmap=None):
    plt.figure(figsize=(12, 10))
    plt.subplot(1,1,1);
    plt.imshow(img,cmap=cmap);
    plt.show()

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
    frame = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)

    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    edges = auto_canny(blurred)

    showim(blurred,cmap='gray')
    showim(edges)

    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    #BP()
    #(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    fcp = frame.copy()
    cv2.drawContours(fcp, cnts, -1, (0,255,0), 1)
    showim(fcp)

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

    #BP()
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares), -1, (0,255,0), 2)
    showim(fcp)

    # Get square centers
    centers = []
    for s in squares:
        M = cv2.moments(s)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((cX,cY))

    # Find center of board
    board_center_x = int(np.median([x[0] for x in centers]))
    board_center_y = int(np.median([x[1] for x in centers]))
    plot_points(fcp,[(board_center_x,board_center_y)])
    showim(fcp)

    # Store distance from center for each contour
    sqdists=[]
    for idx,sq in enumerate(squares):
        sqdists.append({'cnt':sq, 'dist':linelen((centers[idx],(board_center_x, board_center_y)))})
    distsorted = sorted( sqdists, key = lambda x: x['dist'])

    lastidx = len(distsorted)
    for idx,c in enumerate(distsorted):
        if not idx: continue
        delta = c['dist'] - distsorted[idx-1]['dist']
        print ('dist:%f delta: %f' % (c['dist'],delta))
        if delta > 20.0:
            lastidx = idx
            print( 'over')
            break

    squares1 = [x['cnt'] for x in distsorted[:lastidx]]
    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares1), -1, (0,255,0), 2)
    showim(fcp)




if __name__ == '__main__':
    main()
