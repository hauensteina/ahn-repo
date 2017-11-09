//
//  Common.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-22.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Common.h"

UIFont *g_fntBtn;

// Init globals. Called from Appdelegate.
//----------------------------------------
void g_init()
{
    g_fntBtn = [UIFont fontWithName:@"HelveticaNeue" size: 20];
}

// Replacement for annoying [NSString stringWithFormat ...
//---------------------------------------------------------
NSString* nsprintf (NSString *format, ...)
{
    va_list args;
    va_start(args, format);
    NSString *msg =[[NSString alloc] initWithFormat:format
                                          arguments:args];
    return msg;
}

// Prepend path to our documents folder
//---------------------------------------------
NSString* getFullPath( NSString *fname)
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:fname];
    return filePath;
}


