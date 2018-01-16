//
//  TopViewController.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-14.
//  Copyright © 2017 AHN. All rights reserved.
//

#import "TopViewController.h"
#import "MainVC.h"
#import "LeftViewController.h"
#import "RightViewController.h"

@interface TopViewController ()

//@property (assign, nonatomic) NSUInteger type;

@end

@implementation TopViewController

//--------------
- (void)setup
{
    //self.type = type;
    
    self.leftViewController = [LeftViewController new];
    //self.rightViewController = [RightViewController new];
    
    self.leftViewWidth = 250.0;
    self.leftViewBackgroundImage = [UIImage imageNamed:@"imageLeft"];
    self.leftViewBackgroundColor = [UIColor colorWithRed:0.8 green:0.8 blue:0.8 alpha:0.95];
    self.rootViewCoverColorForLeftView = [UIColor colorWithRed:0.0 green:1.0 blue:0.0 alpha:0.05];
    
    
    
//    self.rightViewWidth = 100.0;
//    self.rightViewBackgroundImage = [UIImage imageNamed:@"imageRight"];
//    self.rightViewBackgroundColor = [UIColor colorWithRed:0.65 green:0.5 blue:0.65 alpha:0.95];
//    self.rootViewCoverColorForRightView = [UIColor colorWithRed:1.0 green:0.0 blue:1.0 alpha:0.05];
    
    // -----

    UIColor *grayCoverColor = [UIColor colorWithRed:0.1 green:0.0 blue:0.0 alpha:0.3];
    //UIColor *greenCoverColor = [UIColor colorWithRed:0.0 green:0.1 blue:0.0 alpha:0.3];
    //UIColor *purpleCoverColor = [UIColor colorWithRed:0.1 green:0.0 blue:0.1 alpha:0.3];
    UIBlurEffectStyle regularStyle;

    if (UIDevice.currentDevice.systemVersion.floatValue >= 10.0) {
        regularStyle = UIBlurEffectStyleRegular;
    }
    else {
        regularStyle = UIBlurEffectStyleLight;
    }

    // -----

    self.leftViewPresentationStyle = LGSideMenuPresentationStyleSlideAbove;
    self.rootViewCoverColorForLeftView = grayCoverColor;
    
    //self.rightViewPresentationStyle = LGSideMenuPresentationStyleSlideAbove;
    //self.rootViewCoverColorForRightView = purpleCoverColor;
    
} // setup()

//--------------------------------------------------------
- (void)leftViewWillLayoutSubviewsWithSize:(CGSize)size
{
    [super leftViewWillLayoutSubviewsWithSize:size];

    if (!self.isLeftViewStatusBarHidden) {
        self.leftView.frame = CGRectMake(0.0, 20.0, size.width, size.height-20.0);
    }
}

//---------------------------------------------------------
- (void)rightViewWillLayoutSubviewsWithSize:(CGSize)size
{
    [super rightViewWillLayoutSubviewsWithSize:size];

    if (!self.isRightViewStatusBarHidden ||
        (self.rightViewAlwaysVisibleOptions & LGSideMenuAlwaysVisibleOnPadLandscape &&
         UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad &&
         UIInterfaceOrientationIsLandscape(UIApplication.sharedApplication.statusBarOrientation))) {
        self.rightView.frame = CGRectMake(0.0, 20.0, size.width, size.height-20.0);
    }
}

//---------------------------------
- (BOOL)isLeftViewStatusBarHidden
{
    return super.isLeftViewStatusBarHidden;
}

//-----------------------------------
- (BOOL)isRightViewStatusBarHidden
{
    return super.isRightViewStatusBarHidden;
}

//---------------
- (void)dealloc
{
    NSLog(@"TopViewController deallocated");
}

@end
