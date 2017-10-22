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

//---------------------------------------------------------------------
- (UIImage *) drawRectOnImage:(UIImage *)img
                        color:(UIColor *)color
                            x:(int)x y:(int)y
                        width:(int)width height:(int)height
{
    NSLog(@"drawrect");
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);

    // Draw on Mat
    cv::Point pt1( x, y);
    cv::Point pt2( x+width, y+height);
    //cv::Scalar col(0,0,255);
    int r,g,b;
    GET_RGB( color, r,g,b);
    cv::rectangle( m, pt1, pt2, cv::Scalar(r,g,b,255)); // int thickness=1, int lineType=8, int shift=0)¶

    // Convert back to UIImage
    UIImage *res = MatToUIImage( m);
    return res;
} // drawRectOnImage()



@end
