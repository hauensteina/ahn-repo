//
//  Common.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-22.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <UIKit/UIKit.h>

UIFont *g_fntBtn;

// Init globals. Called from Appdelegate.
//----------------------------------------
void g_init()
{
    g_fntBtn = [UIFont fontWithName:@"HelveticaNeue" size: 20];
}

