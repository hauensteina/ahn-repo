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
    int width  = src.rows;
    int height = src.cols;
    float scale;
    if (width < height) scale = sz / (float) width;
    else scale = sz / (float) height;
    cv::resize( src, dst, cv::Size(int(width*scale),int(height*scale)), 0, 0, cv::INTER_AREA);
    //dst=src;
}

// calculates the median value of a single channel
// based on https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp
//----------------------------------
double median( cv::Mat channel )
{
    double m = (channel.rows*channel.cols) / 2;
    int bin = 0;
    double med = -1.0;
    
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    
    for ( int i = 0; i < histSize && med < 0.0; ++i )
    {
        bin += cvRound( hist.at< float >( i ) );
        if ( bin > m && med < 0.0 )
            med = i;
    }
    
    return med;
} // median()

// Automatic edge detection without parameters
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33)
{
    double v = median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

//#-----------------------
//def get_contours(img):
//#img   = cv2.GaussianBlur( img, (7, 7), 0)
//#img   = cv2.medianBlur( img, 7)
//#img = cv2.bilateralFilter(img, 11, 17, 17)
//#edges = cv2.Canny(img, 60, 200)
//edges = ut.auto_canny(img)
//# find contours in the edge map
//im2, cnts, hierarchy  = cv2.findContours(edges, cv2.RETR_LIST,
//                                         cv2.CHAIN_APPROX_SIMPLE)
//#cnts = sorted( cnts, key=cv2.contourArea, reverse=True)
//#ut.show_contours( img, cnts)
//
//# Keep if larger than 0.1% of image
//img_area = img.shape[0] * img.shape[1]
//cnts = [ c for c in cnts if  cv2.contourArea(c) / img_area > 0.001 ]
//# Keep if reasonably not-wiggly
//#cnts = [ c for c in cnts if   cv2.arcLength(c, closed=True) / len(c) > 2.0 ]
//cnts = [ c for c in cnts if   len(c) < 100 or cv2.arcLength(c, closed=True) / len(c) > 2.0 ]
//
//return cnts

////---------------------------------------------------------------------
//bool compVecByLenAsc( vector<cv::Point> &c1,  vector<cv::Point> &c2)
//{
//    bool res = c1.size() < c2.size();
//    return res;
//}

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
    Contours contours = get_contours(m);
    cv::Mat drawing = cv::Mat::zeros( m.size(), CV_8UC3 );
    draw_contours( contours, drawing);

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
