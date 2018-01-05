//
//  GrabFuncs.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

// This class is the only place where Objective-C and C++ mix.
// All other files are either pure Obj-C or pure C++.

#import <Foundation/Foundation.h>

@interface CppInterface : NSObject

//----------------------------
// Individual steps for debugging
- (UIImage *) f00_blobs:(std::vector<cv::Mat>)imgQ;
//- (UIImage *) f01_straight;
- (UIImage *) f02_vert_lines;
- (UIImage *) f03_vert_lines_2;
- (UIImage *) f04_vert_params;
- (UIImage *) f05_horiz_lines;
- (UIImage *) f06_corners;
- (UIImage *) f07_zoom_in;
- (UIImage *) f08_show_threshed;
- (UIImage *) f09_intersections;
- (UIImage *) f10_features;
- (UIImage *) f11_classify;
- (UIImage *) real_time_flow:(UIImage *)img;
@property int sldDbg;

@end
