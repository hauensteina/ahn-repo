//
//  AppDelegate.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-20.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "MainVC.h"

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;
@property (nonatomic,strong)  UINavigationController *nav;
@property (nonatomic, strong) MainVC *mainVC;


@end

