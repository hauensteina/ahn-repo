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

//# Resize image such that min(width,height) = M
//#------------------
//def resize(img, M):
//width  = img.shape[1]
//height = img.shape[0]
//if width < height:
//scale = M/width
//else:
//scale = M/height
//
//res = cv2.resize(img,(int(width*scale),int(height*scale)), interpolation = cv2.INTER_AREA)
//return res

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


//---------------------------------------------------------------------
- (UIImage *) findBoard:(UIImage *)img
{
    NSLog(@"drawrect");
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);
    // Resize
    cv::Mat small;
    resize( m, small, 500);
    // Grayscale
    cv::Mat gray;
    cv::cvtColor( m, gray, cv::COLOR_BGR2GRAY);

//    // Draw on Mat
//    cv::Point pt1( x, y);
//    cv::Point pt2( x+width, y+height);
//    //cv::Scalar col(0,0,255);
//    int r,g,b;
//    GET_RGB( color, r,g,b);
//    cv::rectangle( m, pt1, pt2, cv::Scalar(r,g,b,255)); // int thickness=1, int lineType=8, int shift=0)¶
//
    // Convert back to UIImage
    UIImage *res = MatToUIImage( gray);
    return res;
} // drawRectOnImage()



@end
