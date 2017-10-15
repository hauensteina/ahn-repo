# /********************************************************************
# Filename: ahn_opencv_util.py
# Author: AHN
# Creation Date: Oct 13, 2017
# **********************************************************************/
#
# Various utility funcs
#

from __future__ import division,print_function

from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
import cv2
from matplotlib import pyplot as plt

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

# Enclose a contour with an n edge polygon
#-------------------------------------------
def approx_poly( cnt, n):
    hull = cv2.convexHull( cnt)
    peri = cv2.arcLength( hull, closed=True)
    epsilon = bisect( lambda x: -len(cv2.approxPolyDP(hull, x * peri, closed=True)),
                      0.0, 1.0, -n)
    res  = cv2.approxPolyDP(hull, epsilon*peri, closed=True)
    return res

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

# Order four points clockwise
#------------------------------
def order_points(pts):
    top_bottom = sorted( pts, key=lambda x: x[1])
    top = top_bottom[:2]
    bottom = top_bottom[2:]
    res = sorted( top, key=lambda x: x[0]) + sorted( bottom, key=lambda x: -x[0])
    return np.array(res).astype(np.float32)

# Draw lines on an image
#----------------------------
def plot_lines(img, lines, color=(255,0,0)):
    for  p1,p2 in lines:
        #BP()
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=2) #, lineType=8, shift=0)

# Mark points on an image
#----------------------------
def plot_points(img, points, color=(255,0,0)):
    for p in points:
        cv2.circle(img, (p[0],p[1]), 5, color, thickness=-1) #, lineType=8, shift=0)

# Resize image such that min(width,height) = M
#------------------
def resize(img, M):
    width  = img.shape[1]
    height = img.shape[0]
    if width < height:
        scale = M/width
    else:
        scale = M/height

    res = cv2.resize(img,(int(width*scale),int(height*scale)))
    return res

# Display contour for debugging
#--------------------------------
def show_contours(img, cnts):
    for c in cnts:
        peri = cv2.arcLength(c, closed=True)
        area = cv2.contourArea(c)
        hullArea = cv2.contourArea(cv2.convexHull(c))
        print ('area,efficiency,straightness: %d %f %f' % (area, np.sqrt(area)/max(1,peri), peri / len(c)))
        fcp = img.copy()
        cv2.drawContours(fcp, [c], -1, (0,255,0), 2)
        showim(fcp)

# Display an image
#-------------------------
def showim(img,cmap=None):
    plt.figure(figsize=(12, 10))
    plt.subplot(1,1,1);
    plt.imshow(img,cmap=cmap);
    plt.show()

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

# Get unit vector of vector
#----------------------------
def unit_vector(vector):
    return vector / np.linalg.norm(vector)
