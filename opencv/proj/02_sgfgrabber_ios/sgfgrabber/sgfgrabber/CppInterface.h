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
//- (UIImage *) f02_horizontals;
- (UIImage *) f03_vert_lines;
- (UIImage *) f04_vert_lines_2;
- (UIImage *) f05_vert_params;
//- (UIImage *) f04_clean_horiz_lines;
//- (UIImage *) f03_find_board;
//- (UIImage *) f04_zoom_in;
//- (UIImage *) f05_find_intersections;
////- (UIImage *) f06_find_lines;
//- (UIImage *) f07_show_horiz_lines;
//- (UIImage *) f08_show_vert_lines;
////- (UIImage *) f09_clean_horiz_lines;
//- (UIImage *) f10_clean_vert_lines;
//- (UIImage *) f11_classify;
//- (int) f05_get_boardsize;
//- (UIImage *) f06_get_intersections;
//- (UIImage *) f07_classify;
// All in one for production
- (UIImage *) findBoard:(UIImage *)img;
@property float sld_low;
@property int canny_hi;

@end
