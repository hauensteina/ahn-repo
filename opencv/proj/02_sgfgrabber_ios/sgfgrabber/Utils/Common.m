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

//==========
// Drawing
//==========
// Draw a massive rectangle on a view
//--------------------------------------------------------------------------------
void drawRect( UIView *view, UIColor *color, int x, int y, int width, int height)
{
    UIView *myBox  = [[UIView alloc] initWithFrame:CGRectMake(x, y, width, height)];
    myBox.backgroundColor = color;
    [view addSubview:myBox];
}

//==========
// Strings
//==========

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
// UI Helpers
//=============

// Popup with message and OK button
//-----------------------------------------------
void popup (NSString *msg, NSString *title)
{
    UIAlertController * alert = [UIAlertController
                                 alertControllerWithTitle:title
                                 message:msg
                                 preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction* yesButton = [UIAlertAction
                                actionWithTitle:@"OK"
                                style:UIAlertActionStyleDefault
                                handler:^(UIAlertAction * action) {
                                }];
    [alert addAction:yesButton];
    
    UIViewController *vc = [[[[UIApplication sharedApplication] delegate] window] rootViewController];
    [vc presentViewController:alert animated:YES completion:nil];
} // popup()

// Alert with several choices
//---------------------------------------------------------------------------------------
void choicePopup (NSArray *choices, NSString *title, void(^callback)(UIAlertAction *))
{
    UIAlertController *alert = [UIAlertController
                                 alertControllerWithTitle:title
                                 message:@""
                                 preferredStyle:UIAlertControllerStyleAlert];
    for (NSString *str in choices) {
        UIAlertAction *button =  [UIAlertAction
                                  actionWithTitle:str
                                  style:UIAlertActionStyleDefault
                                  handler:callback];
        [alert addAction:button];
    }
    
    UIViewController *vc = [[[[UIApplication sharedApplication] delegate] window] rootViewController];
    [vc presentViewController:alert animated:YES completion:nil];
} // choicePopup()

//=============
// File Stuff
//=============

// Prepend path to documents folder
//---------------------------------------------
NSString* getFullPath( NSString *fname)
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:fname];
    return filePath;
}

// Change filename extension
//----------------------------------------------------------
NSString* changeExtension( NSString *fname, NSString *ext)
{
    NSString *res = [fname stringByDeletingPathExtension];
    res = nscat( res, ext);
    return res;
}

// Find a file in the main bundle
//----------------------------------
NSString* findInBundle( NSString *basename, NSString *ext)
{
    NSBundle* myBundle = [NSBundle mainBundle];
    NSString* path = [myBundle pathForResource:basename ofType:ext];
    return path;
}

// List files in folder, filter by extension, sort
//------------------------------------------------------------------------
NSArray* globFiles( NSString *path_, NSString *prefix, NSString *ext)
{
    id fm = [NSFileManager defaultManager];
    NSString *path = getFullPath( path_);
    NSArray *files =
    [fm contentsOfDirectoryAtPath:path error:nil];
    NSPredicate *predicate = [NSPredicate
                              predicateWithFormat:@"SELF like[c] %@", nsprintf( @"%@*%@", prefix, ext)];
    files = [files filteredArrayUsingPredicate:predicate];
    files = [files sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
    return files;
} // globFiles()

// Make a folder below the document dir
//----------------------------------------
void makeDir( NSString *dir)
{
    NSString *path = getFullPath( dir);
    [[NSFileManager defaultManager] createDirectoryAtPath: path
                              withIntermediateDirectories: YES
                                               attributes: nil
                                                    error: nil];
}

// Remove file below document dir
//--------------------------------
void rmFile( NSString *fname)
{
    NSString *fullfname = getFullPath( fname);
    NSError *error;
    [[NSFileManager defaultManager]  removeItemAtPath:fullfname error:&error];
}

// Check whether folder exists
//-----------------------------------
bool dirExists( NSString *path_)
{
    NSString *path = getFullPath( path_);
    BOOL isDir;
    BOOL fileExists = [[NSFileManager defaultManager]  fileExistsAtPath:path isDirectory:&isDir];
    return (fileExists && isDir);
}

// Check whether file exists
//-----------------------------------
bool fileExists( NSString *path_)
{
    NSString *path = getFullPath( path_);
    BOOL isDir;
    BOOL fileExists = [[NSFileManager defaultManager]  fileExistsAtPath:path isDirectory:&isDir];
    return (fileExists && !isDir);
}

