//
//  GrabFuncs.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface GrabFuncs : NSObject

//----------------------------
+ (NSString *) opencvVersion;
// Individual steps for debugging
- (UIImage *) f00_adaptive_thresh:(UIImage *)img;
- (UIImage *) f01_closing;
- (UIImage *) f02_flood;
- (UIImage *) f03_find_board;
- (UIImage *) f04_zoom_in;
- (UIImage *) f05_black_blobs;
//- (int) f05_get_boardsize;
//- (UIImage *) f06_get_intersections;
//- (UIImage *) f07_classify;
// All in one for production
- (UIImage *) findBoard:(UIImage *)img;
@property float sld_low;
@property int canny_hi;

@end
