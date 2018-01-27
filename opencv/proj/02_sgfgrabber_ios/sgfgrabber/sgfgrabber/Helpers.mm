//
//  Helpers.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-27.
//  Copyright © 2018 AHN. All rights reserved.
//

// Collection of app specific standalone Objective-C++ functions

// Don't change the order of these two,
// and don't move them down
#import "Ocv.hpp"
#import <opencv2/imgcodecs/ios.h>

#import <iostream>
#import <vector>
#import <regex>

#import <UIKit/UIKit.h>

#import "Common.hpp"

// Load image from file
//---------------------------------------------
void load_img( NSString *fname, cv::Mat &m)
{
    UIImage *img = [UIImage imageNamed:fname];
    UIImageToMat(img, m);
}


