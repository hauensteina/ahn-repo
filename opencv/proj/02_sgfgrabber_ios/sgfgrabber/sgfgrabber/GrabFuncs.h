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
- (UIImage *) f00_contours:(UIImage *)img;
- (UIImage *) f01_filtered_contours;
- (UIImage *) f02_inside_contours;
- (UIImage *) f03_find_board;
@end
