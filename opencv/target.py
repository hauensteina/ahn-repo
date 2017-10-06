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
      %s
    Description:
      Find Go boards in jpeg images
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
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = gray
    edges = auto_canny(blurred)

    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges.copy(), cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    #BP()
    #(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    squares = []
    for i,c in enumerate(cnts):
        #if cv2.contourArea(c) > 1000: continue
        #if cv2.contourArea(c) < 10: continue
        # Slanted enclosing rect
        rect = cv2.minAreaRect(c)
        # Corners of rect
        box  = cv2.boxPoints(rect)
        if cv2.contourArea(np.int0(box)) > 1000: continue
        if cv2.contourArea(np.int0(box)) < 50: continue
        len0 =  np.linalg.norm(box[1]-box[0])
        len1 =  np.linalg.norm(box[2]-box[1])
        sides = sorted([len0,len1])
        ratio = sides[1] / sides[0]
        if ratio > 1.5: continue
        squares.append(box)
        #print ('%d %f %f %f' % (i, cv2.contourArea(c), cv2.contourArea(np.int0(box)), ratio))

    # Find center of board
    sq = np.stack(squares)
    points = sq.reshape(sq.shape[0]*sq.shape[1],sq.shape[2])
    cX = np.median([x[0] for x in points])
    cY = np.median([x[1] for x in points])
    #M = cv2.moments(edges)
    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])

    points1 = sorted(points, key = lambda p: (p[0]-cX)**2 + (p[1]-cY)**2, reverse=True)

    # Remove outliers
    if len(points1) > 50:
        points1 = points1[40:]
    # Try again
    board = cv2.boxPoints(cv2.minAreaRect(np.stack(points1)))


    #print('=============')
    #print(board)

    if args.squares:
        # Display squares
        fcp = frame.copy()
        for sq in squares:
            sq = np.int0(sq)
            cv2.drawContours(fcp, [sq], 0, (0,255,0), 3)
        cv2.circle(fcp, (cX,cY), 10, (0,0,255), thickness=1, lineType=8, shift=0)
        plt.subplot(1,1,1)
        plt.imshow(fcp)
    else:
        # Display board rectangle
        fcp = frame.copy()
        board = np.int0(board)
        cv2.drawContours(fcp, [board], 0, (0,255,0), 3)
        plt.subplot(1,1,1)
        plt.imshow(fcp)


    # for i,box in enumerate(squares[25:]):
    #     if i >= 25: break
    #     fcp = frame.copy()
    #     box = np.int0(box)
    #     cv2.drawContours(fcp, [box], 0, (0,255,0), 3)
    #     plt.subplot(5,5,i+1)
    #     plt.imshow(fcp)


    #BP()

    # plt.subplot(221),plt.imshow(blurred, cmap = 'gray')
    # plt.title('Blurred'), plt.xticks([]), plt.yticks([])
    # plt.subplot(222),plt.imshow(edges, cmap = 'gray')
    # plt.title('Edges'), plt.xticks([]), plt.yticks([])
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #plt.subplot(111),plt.imshow(edges, cmap = 'gray')
    #plt.subplot(111),plt.imshow(im2, cmap = 'gray')
    #plt.subplot(111),plt.imshow(frame, cmap = 'gray')
    #plt.title('Lines'), plt.xticks([]), plt.yticks([])


    plt.show()



if __name__ == '__main__':
    main()
