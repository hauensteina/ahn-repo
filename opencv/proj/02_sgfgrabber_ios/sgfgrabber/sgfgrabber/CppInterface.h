//
//  CppInterface.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

// This class is the only place where Objective-C and C++ mix.
// All other files are either pure Obj-C or pure C++.

#import <Foundation/Foundation.h>

@interface CppInterface : NSObject

// Individual steps for debugging
//---------------------------------
- (UIImage *) f00_blobs:(std::vector<cv::Mat>)imgQ;
- (UIImage *) f01_vert_lines;
- (UIImage *) f02_horiz_lines;
- (UIImage *) f03_corners;
- (UIImage *) f04_zoom_in;
- (UIImage *) f05_dark_places;
- (UIImage *) f06_mask_dark;
- (UIImage *) f07_white_holes;
- (UIImage *) f08_features;
- (UIImage *) f09_classify;
- (UIImage *) real_time_flow:(UIImage *)img;
@property int sldDbg;

@end
