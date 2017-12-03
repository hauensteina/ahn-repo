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
- (UIImage *) f00_blobs:(UIImage *)img;
- (UIImage *) f01_straight;
- (UIImage *) f02_horiz_lines;
- (UIImage *) f03_vert_lines;
- (UIImage *) f04_vert_lines_2;
- (UIImage *) f05_vert_params;
- (UIImage *) f06_corners;
- (UIImage *) f07_zoom_in;
- (UIImage *) f08_repeat_on_zoomed;
- (UIImage *) findBoard:(UIImage *)img;
@property float sld_low;
@property int canny_hi;

@end
