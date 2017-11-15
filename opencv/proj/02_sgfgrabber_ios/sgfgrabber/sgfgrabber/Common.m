//
//  Common.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-22.
//  Copyright © 2017 AHN. All rights reserved.
//

// Generally useful convenience funcs

#import <UIKit/UIKit.h>
#import "Common.h"


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

// Concatenate two NSStrings
//-----------------------------
NSString *nscat (id a, id b)
{
    return [NSString stringWithFormat:@"%@%@",a,b];
}

//=============
// File Stuff
//=============

// Prepend path to our documents folder
//---------------------------------------------
NSString* getFullPath( NSString *fname)
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:fname];
    return filePath;
}


