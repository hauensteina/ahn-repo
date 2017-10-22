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
- (UIImage *) drawRectOnImage:(UIImage *)img
                            x:(int)x y:(int)y
                        width:(int)width height:(int)height;

@end
