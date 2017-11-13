//
//  Common.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <UIKit/UIKit.h>

#define SIGN(x) (x)>=0?1:-1

#define RGB(rgbValue) [UIColor \
colorWithRed:((float)((rgbValue & 0xFF0000) >> 16))/255.0 \
green:((float)((rgbValue & 0xFF00) >> 8))/255.0 \
blue:((float)(rgbValue & 0xFF))/255.0 alpha:1.0]

#define GET_RGB(col,r,g,b) \
do { \
CGFloat rr,gg,bb,aa; \
[col getRed: &rr green: &gg blue: &bb alpha: &aa];  \
r = int(rr * 255); g = int(gg * 255); b = int(bb * 255); \
} while(0)

#define SCREEN_BOUNDS [UIScreen mainScreen].bounds
#define SCREEN_WIDTH  ([UIScreen mainScreen].bounds.size.width)
#define SCREEN_HEIGHT ([UIScreen mainScreen].bounds.size.height)

#define CLEAR  [UIColor clearColor]
#define WHITE  [UIColor whiteColor]
#define BLACK  [UIColor blackColor]
#define YELLOW [UIColor yellowColor]
#define RED    [UIColor redColor]
#define BLUE   [UIColor blueColor]
#define GREEN  [UIColor greenColor]
#define GRAY   [UIColor grayColor]
#define DARKRED    RGB(0xd00000)
#define DARKGREEN  RGB(0x007000)
#define DARKBLUE   RGB(0x4481A7)

extern UIFont *g_fntBtn;

#ifdef __cplusplus
extern "C" {
#endif
    
    void g_init(void); // Init globals. Called from Appdelegate.
    NSString *nsprintf (NSString *format, ...);
    NSString *nscat (id a, id b);
    NSString* getFullPath( NSString *fname);


#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#import <iostream>

// Append a vector to another
//--------------------------------------------------------
template <typename T>
void vapp( std::vector<T> &v1, const std::vector<T> &v2)
{
    v1.insert( v1.end(), v2.begin(), v2.end());
}

#endif




