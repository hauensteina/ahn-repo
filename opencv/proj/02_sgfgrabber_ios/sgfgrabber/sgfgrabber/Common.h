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
    NSString* getFullPath( NSString *fname);


#ifdef __cplusplus
}
#endif





