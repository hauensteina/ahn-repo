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
import scipy.signal
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

# Get unit vector of vector
#----------------------------
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

#-----------------------------
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

# Find x where f(x) = target where f is an increasing func.
#------------------------------------------------------------
def bisect( f, lower, upper, target, maxiter=10):
    n=0
    while True and n < maxiter:
        n += 1
        res = (upper + lower) / 2.0
        val = f(res)
        if val > target:
            upper = res
        elif val < target:
            lower = res
        else:
            break
    return res

# Enclose a contour with an n edge polygon
#-------------------------------------------
def approx_poly( cnt, n):
    hull = cv2.convexHull( cnt)
    peri = cv2.arcLength( hull, closed=True)
    epsilon = bisect( lambda x: -len(cv2.approxPolyDP(hull, x * peri, closed=True)),
                      0.0, 1.0, -n)
    res  = cv2.approxPolyDP(hull, epsilon*peri, closed=True)
    return res

# Order four points clockwise
#------------------------------
def order_points(pts):
    top_bottom = sorted( pts, key=lambda x: x[1])
    top = top_bottom[:2]
    bottom = top_bottom[2:]
    res = sorted( top, key=lambda x: x[0]) + sorted( bottom, key=lambda x: -x[0])
    return np.array(res).astype(np.float32)
    # # initialzie a list of coordinates that will be ordered
    # # such that the first entry in the list is the top-left,
    # # the second entry is the top-right, the third is the
    # # bottom-right, and the fourth is the bottom-left
    # rect = np.zeros((4, 2), dtype = "float32")

    # # the top-left point will have the smallest sum, whereas
    # # the bottom-right point will have the largest sum
    # s = pts.sum(axis = 1)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]
    # BP()

    # # now, compute the difference between the points, the
    # # top-right point will have the smallest difference,
    # # whereas the bottom-left will have the largest difference
    # diff = np.diff(pts, axis = 1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]

    # # return the ordered coordinates
    # return rect

# Zoom into an image area where pts are the four corners.
# From pyimagesearch by Adrian Rosebrock
#-----------------------------------------
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# Stretch a line by factor, on both ends
#-----------------------------------------
def stretch_line(line, factor):
    p0 = line[0]
    p1 = line[1]
    length = np.linalg.norm(p1-p0)
    v = ((factor-1.0) * length) * unit_vector(p1-p0)
    q1 = p1 + v
    q0 = p0 - v
    return (q0,q1)

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
    #blurred = gray
    edges = auto_canny(gray)

    #showim(blurred,cmap='gray')
    #showim(edges)

    # find contours in the edge map
    im2, cnts, hierarchy  = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)

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

    fcp = frame.copy()
    cv2.drawContours(fcp, np.array(squares), -1, (0,255,0), 2)
    #showim(fcp)

    # Get square centers
    centers = [ (int(M["m10"] / M["m00"]),
                 int(M["m01"] / M["m00"]))
                for M in [cv2.moments(s) for s in squares]]

    # Find center of board
    board_center = np.array (( int(np.median([x[0] for x in centers])),
                               int(np.median([x[1] for x in centers])) ))
    plot_points(fcp,[board_center])
    showim(fcp)

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
    showim(fcp)

    # Find enclosing 4-polygon
    points = np.array([p for s in squares1 for p in s])
    board = approx_poly( points, 4).reshape(4,2)
    #BP()

    #fcp = frame.copy()
    #cv2.drawContours(fcp, [board], -1, (0,255,0), 1)
    #showim(fcp)

    # Make the board a little larger
    factor = 1.1
    board = order_points(board)
    diag1_stretched = stretch_line( (board[0],board[2]), factor)
    diag2_stretched = stretch_line( (board[1],board[3]), factor)
    board_stretched = np.int0([diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1]])
    #BP()
    fcp = frame.copy()
    cv2.drawContours(fcp, [board_stretched], -1, (0,255,0), 1)
    #showim(fcp)

    # Zoom in on the board
    zoomed = four_point_transform( gray, board_stretched)
    showim(zoomed)

    # DFT on original
    dft = cv2.dft(gray.astype('float32'), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shifted[:,:,0],dft_shifted[:,:,1]))
    #showim(magnitude_spectrum)

    # DFT on zoomed
    dft = cv2.dft(zoomed.astype('float32'), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shifted[:,:,0],dft_shifted[:,:,1]))
    #showim(magnitude_spectrum)

    # Get three rows from the image
    height = zoomed.shape[0]
    width = zoomed.shape[1]
    r_middle = height // 2
    r_lower  = height // 4
    r_upper  = r_middle + r_lower

    # Get the spectrum for each
    CLIP = 1000

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
    #BP()

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
