//
//  Common.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

// Generally useful convenience funcs to be included by Obj-C and Obj-C++ files.
// Pure C++ files should not include this

#import <UIKit/UIKit.h>
#import "Common.hpp"

#ifdef __cplusplus
extern "C" {
#endif
    
    NSString *nsprintf (NSString *format, ...);
    NSString *nscat (id a, id b);
    // Prepend path to documents folder
    NSString* getFullPath( NSString *fname);
    // Find a file in the main bundle
    NSString* findInBundle( NSString *basename, NSString *ext);
    // Popup notification
    void popup (NSString *str, NSString *title);

#ifdef __cplusplus
}
#endif





