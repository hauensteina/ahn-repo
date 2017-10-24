//
//  GrabFuncs.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "Common.h"
#import "GrabFuncs.h"

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
//------------------------------------------------------
void resize(const cv::Mat &src, cv::Mat &dst, int sz)
{
    //cv::Size s;
    int width  = src.cols;
    int height = src.rows;
    float scale;
    if (width < height) scale = sz / (float) width;
    else scale = sz / (float) height;
    cv::resize( src, dst, cv::Size(int(width*scale),int(height*scale)), 0, 0, cv::INTER_AREA);
}

// Calculates the median value of a single channel
//-------------------------------------
int channel_median( cv::Mat channel )
{
    cv::Mat flat = channel.reshape(1,1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);
    double res = sorted.at<uchar>(sorted.size() / 2);
    return res;
}

// Calculates the median value of a vector of int
//-------------------------------------------------
int int_median( std::vector<int> ints )
{
    std::sort( ints.begin(), ints.end(), [](int a, int b) { return a < b; });
    int res = ints[ints.size() / 2];
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
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33)
{
    double v = channel_median(src);
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
    auto_canny( img, m, 0.5);
    // Find contours
    findContours( m, conts, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    return conts;
} // get_contours()

// Try to eliminate all contours except those on the board
//---------------------------------------------------------------------
Contours filter_contours( const Contours conts, int width, int height)
{
    Contours large_conts;
    float minArea = width * height / 4000.0;
    std::copy_if( conts.begin(), conts.end(), std::back_inserter(large_conts),
                 [minArea](Contour c){return cv::contourArea(c) > minArea;} );
    
    Contours large_hullarea;
    std::copy_if( large_conts.begin(), large_conts.end(), std::back_inserter(large_hullarea),
                 [minArea](Contour c){
                     Contour hull;
                     cv::convexHull( c, hull);
                     return cv::contourArea(hull) > 0.001; });

    return large_hullarea;
}

// Find the center of the board, which is the median of the contours on the board
//----------------------------------------------------------------------------------
cv::Point get_board_center( const Contours conts, std::vector<cv::Point> &centers)
{
    centers.resize( conts.size());
    int i = 0;
    std::generate( centers.begin(), centers.end(), [conts,&i] {
        Contour c = conts[i++];
        cv::Moments M = cv::moments( c);
        int cent_x = int(M.m10 / M.m00);
        int cent_y = int(M.m01 / M.m00);
        return cv::Point(cent_x, cent_y);
    });
    i=0;
    std::vector<int> cent_x( conts.size());
    std::generate( cent_x.begin(), cent_x.end(), [centers,&i] { return centers[i++].x; } );
    i=0;
    std::vector<int> cent_y( conts.size());
    std::generate( cent_y.begin(), cent_y.end(), [centers,&i] { return centers[i++].y; } );
    int x = int_median( cent_x);
    int y = int_median( cent_y);
    return cv::Point(x,y);
}

// Mark a point on an image
//--------------------------------------
void draw_point( cv::Point p, cv::Mat &img)
{
    cv::circle( img, p, 10, cv::Scalar(255,0,0), -1);
}

//-----------------------------------------
- (UIImage *) findBoard:(UIImage *)img
{
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);
    // Resize
    resize( m, m, 500);
    // Grayscale
    cv::cvtColor( m, m, cv::COLOR_BGR2GRAY);
    // Contours
    cv::Mat drawing = cv::Mat::zeros( m.size(), CV_8UC3 );
    Contours contours = get_contours(m);
    Contours onBoard = filter_contours( contours, m.cols, m.rows);
    draw_contours( onBoard, drawing);
    std::vector<cv::Point> centers;
    cv::Point board_center = get_board_center( onBoard, centers);
    draw_point( board_center, drawing);

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
