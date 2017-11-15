//
//  Common.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-22.
//  Copyright © 2017 AHN. All rights reserved.
//

//======================================
// Generally useful convenience funcs
//======================================


#import <UIKit/UIKit.h>
#import "Common.h"

cplx I(0.0, 1.0);

//===============
// String Stuff
//===============

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

//======
// Math
//======

//---------------------------------------------------
void _fft(cplx buf[], cplx out[], int n, int step)
{
    if (step < n) {
        _fft( out, buf, n, step * 2);
        _fft( out + step, buf + step, n, step * 2);
        
        for (int i = 0; i < n; i += 2 * step) {
            cplx t = exp( -I * PI * (cplx(i) / cplx(n))) * out[ i + step];
            buf[ i / 2]     = out[i] + t;
            buf[ (i + n)/2] = out[i] - t;
        }
    }
}

//---------------------------
void fft(cplx buf[], int n)
{
    cplx out[n];
    for (int i = 0; i < n; i++) out[i] = buf[i];
    
    _fft( buf, out, n, 1);
}
