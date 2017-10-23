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
}

// Automatic edge detection without parameters
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33)
{
    double v = median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

//---------------------------------------------------------------------
- (UIImage *) findBoard:(UIImage *)img
{
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);
    // Resize
    //cv::Mat small;
    resize( m, m, 500);
    // Grayscale
    //cv::Mat gray;
    cv::cvtColor( m, m, cv::COLOR_BGR2GRAY);
    // Edges
    cv::Mat edges;
    auto_canny( m, m);

//    // Draw on Mat
//    cv::Point pt1( x, y);
//    cv::Point pt2( x+width, y+height);
//    //cv::Scalar col(0,0,255);
//    int r,g,b;
//    GET_RGB( color, r,g,b);
//    cv::rectangle( m, pt1, pt2, cv::Scalar(r,g,b,255)); // int thickness=1, int lineType=8, int shift=0)¶
//
    // Convert back to UIImage
    UIImage *res = MatToUIImage( m);
    return res;
} // drawRectOnImage()



@end
