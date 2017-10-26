//
//  GrabFuncs.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface GrabFuncs : NSObject

@property int canny_low;
@property int canny_hi;
//@property int thresh;

//----------------------------
+ (NSString *) opencvVersion;
// Individual steps for debugging
- (UIImage *) f00_adaptive_thresh:(UIImage *)img;
- (UIImage *) f01_opening;
- (UIImage *) f02_flood;
- (UIImage *) f03_find_board;
// All in one for production
- (UIImage *) findBoard:(UIImage *)img;
@end
