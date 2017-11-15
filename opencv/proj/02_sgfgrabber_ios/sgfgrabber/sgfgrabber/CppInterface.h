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
- (UIImage *) f00_adaptive_thresh:(UIImage *)img;
- (UIImage *) f01_closing;
- (UIImage *) f02_flood;
- (UIImage *) f03_find_board;
- (UIImage *) f04_zoom_in;
- (UIImage *) f05_find_intersections;
- (UIImage *) f06_hough_grid;
- (UIImage *) f07_clean_grid_h;
- (UIImage *) f08_clean_grid_v;
- (UIImage *) f09_classify;
//- (int) f05_get_boardsize;
//- (UIImage *) f06_get_intersections;
//- (UIImage *) f07_classify;
// All in one for production
- (UIImage *) findBoard:(UIImage *)img;
@property float sld_low;
@property int canny_hi;

@end
