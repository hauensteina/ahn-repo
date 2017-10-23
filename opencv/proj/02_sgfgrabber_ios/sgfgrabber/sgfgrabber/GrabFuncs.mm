//
//  GrabFuncs.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "Common.h"
#import "GrabFuncs.h"

//using namespace std;
//typedef std::vector vector;
typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
static cv::RNG rng(12345);

@implementation GrabFuncs
//=========================

//=== General utility funcs ===
//=============================

#pragma mark - Utility funcs
//----------------------------
+ (NSString *) opencvVersion
{
    return [NSString stringWithFormat:@"OpenCV version: %s", CV_VERSION];
}

// Resize image such that min(width,height) = sz
//---------------------------------------------------------
void resize(const cv::Mat &src, cv::Mat &dst, int sz)
{
    //cv::Size s;
    int width  = src.cols;
    int height = src.rows;
    float scale;
    if (width < height) scale = sz / (float) width;
    else scale = sz / (float) height;
    cv::resize( src, dst, cv::Size(int(width*scale),int(height*scale)), 0, 0, cv::INTER_AREA);
    //dst=src;
}

// Calculates the median value of a single channel
//--------------------------------
int median( cv::Mat channel )
{
    cv::Mat flat = channel.reshape(1,1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);
    double res = sorted.at<uchar>(sorted.size() / 2);
    return res;
}

//-------------------------------------------------------
void draw_contours( const Contours cont, cv::Mat &dst)
{
    // Draw contours
    for( int i = 0; i< cont.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
        drawContours( dst, cont, i, color, 2, 8);
    }
} // draw_contours()

// Automatic edge detection without parameters
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.5)
{
    double v = median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

//=== Task specific helpers ===
//=============================

#pragma mark - Task Specific Helpers
//---------------------------------------------
Contours get_contours( const cv::Mat &img)
{
    Contours conts;
    std::vector<cv::Vec4i> hierarchy;
    // Edges
    cv::Mat m;
    auto_canny( img, m);
    // Find contours
    findContours( m, conts, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    // Keep if larger than 0.1% of image
//    float img_area = img.rows * img.cols;
//    float lim = img_area * 0.001;
//    Contours large_conts;
//    std::copy_if( conts.begin(), conts.end(), std::back_inserter(large_conts),
//                 [lim](Contour c){return cv::contourArea(c) > lim;} );

    // Keep if reasonably not-wiggly
//    Contours straight_conts;
//    std::copy_if( conts.begin(), conts.end(), std::back_inserter(straight_conts),
//                 [](Contour c){return (c.size() < 100) || (cv::arcLength(c, true) / (float)c.size() > 2.0);} );
    
    return conts;
} // get_contours()

//# Try to eliminate all contours except those on the board
//#----------------------------------------------------------
//def filter_squares(cnts, width, height):
//squares = []
//for i,c in enumerate(cnts):
//area = cv2.contourArea(c)
//#if area > width*height / 2.5: continue
//if area < width*height / 4000.0 : continue
//peri = cv2.arcLength(c, closed=True)
//hullArea = cv2.contourArea(cv2.convexHull(c))
//if hullArea < 0.001: continue
//solidity = area / float(hullArea)
//approx = cv2.approxPolyDP(c, 0.01 * peri, closed=True)
//#if len(approx) < 4: continue  # Not a square
//# not a circle
//#if len(approx) > 6:
//#center,rad = cv2.minEnclosingCircle(c)
//#circularity = area / (rad * rad * np.pi)
//#if circularity < 0.50: continue
//#print (circularity)
//
//#if len(approx) > 14: continue
//#(x, y, w, h) = cv2.boundingRect(approx)
//#aspectRatio = w / float(h)
//#arlim = 0.4
//#if aspectRatio < arlim: continue
//#if aspectRatio > 1.0 / arlim: continue
//#if solidity < 0.45: continue
//#if solidity < 0.07: continue
//squares.append(c)
//return squares
//

// Try to eliminate all contours except those on the board
//---------------------------------------------------------------------
Contours filter_squares( const Contours conts, int width, int height)
{
    Contours large_conts;
    float minArea = width * height / 4000.0;
    std::copy_if( conts.begin(), conts.end(), std::back_inserter(large_conts),
                 [minArea](Contour c){return cv::contourArea(c) > minArea;} );
    return large_conts;
}


//-----------------------------------------
- (UIImage *) findBoard:(UIImage *)img
{
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);
    // Resize
    //cv::Mat small;
    resize( m, m, 500);
    // Grayscale
    cv::cvtColor( m, m, cv::COLOR_BGR2GRAY);
    // Contours
    cv::Mat drawing = cv::Mat::zeros( m.size(), CV_8UC3 );
    Contours contours = get_contours(m);
    Contours onBoard = filter_squares( contours, m.cols, m.rows);
    draw_contours( onBoard, drawing);

//    // Draw on Mat
//    cv::Point pt1( x, y);
//    cv::Point pt2( x+width, y+height);
//    //cv::Scalar col(0,0,255);
//    int r,g,b;
//    GET_RGB( color, r,g,b);
//    cv::rectangle( m, pt1, pt2, cv::Scalar(r,g,b,255)); // int thickness=1, int lineType=8, int shift=0)¶
//
    // Convert back to UIImage
    UIImage *res = MatToUIImage( drawing);
    return res;
} // drawRectOnImage()



@end
