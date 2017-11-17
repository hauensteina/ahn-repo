//
//  Globals.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

#ifndef Globals_h
#define Globals_h

#import "AppDelegate.h"

extern UIFont *g_fntBtn;
extern AppDelegate *g_app;

#ifdef __cplusplus
extern "C" {
#endif
    
    void g_init(void); // Init globals. Called from Appdelegate.
    
    
#ifdef __cplusplus
}
#endif

#endif /* Globals_h */
