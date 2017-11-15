//
//  Common.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

//=========================================================================
// Generally useful convenience funcs to be included from pure cpp files.
// Obj-C and Obj-C++ files include this file via Common.h.
//=========================================================================


#ifndef Common_cpp_hpp
#define Common_cpp_hpp

//=================================
// Common between C++ and Obj C
//=================================

#define ILOOP(n) for (int i=0; i < (n); i++ )
#define JLOOP(n) for (int j=0; j < (n); j++ )
#define KLOOP(n) for (int k=0; k < (n); k++ )
#define RLOOP(n) for (int r=0; r < (n); r++ )
#define CLOOP(n) for (int c=0; c < (n); c++ )

#define ISLOOP(n) for (int i=0; i < ((n).size()); i++ )
#define JSLOOP(n) for (int j=0; j < ((n).size()); j++ )
#define KSLOOP(n) for (int k=0; k < ((n).size()); k++ )
#define RSLOOP(n) for (int r=0; r < ((n).size()); r++ )
#define CSLOOP(n) for (int c=0; c < ((n).size()); c++ )

#define SIGN(x) (x)>=0?1:-1
#define PI M_PI

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


//=================
// C++ only below
//=================

#ifdef __cplusplus
#import <iostream>
#import <complex>
#import <vector>

typedef std::complex<double> cplx;
extern cplx I;

void fft(cplx buf[], int n);


//===================
// Templates
//===================

// Misc
//=======

// Swap two things
//---------------------------
template <typename T>
void swap( T &x1, T &x2)
{
    T tmp = x1; x1 = x2; x2 = tmp;
}

//# Find x where f(x) = target where f is an increasing func.
//------------------------------------------------------------
template<typename Func>
float bisect( Func f, float lower, float upper, int target, int maxiter=10)
{
    int n=0;
    float res=0.0;
    while (n++ < maxiter) {
        res = (upper + lower) / 2.0;
        int val = int(f(res));
        if (val > target) upper = res;
        else if (val < target) lower = res;
        else break;
    } // while
    return res;
}

// Vector
//==========

// Flatten a vector of vectors into a vector
// [[1,2,3],[4,5,6],...] -> [1,2,3,4,5,6,...]
//--------------------------------------------
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v)
{
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}

// Append a vector to another
//--------------------------------------------------------
template <typename T>
void vapp( std::vector<T> &v1, const std::vector<T> &v2)
{
    v1.insert( v1.end(), v2.begin(), v2.end());
}

// Calculate the median value of a vector
//----------------------------------------------
template <typename T>
T vec_median( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    std::sort( vec.begin(), vec.end(), [](T a, T b) { return a < b; });
    return vec[vec.size() / 2];
}

// Calculates the avg value of a vector
//----------------------------------------------
template <typename T>
T vec_avg( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    double ssum = 0;
    ISLOOP (vec) { ssum += vec[i]; }
    return T(ssum / vec.size());
}

// Calculates the avg value of a vector, with acces func
//----------------------------------------------------------
template <typename T, typename Func>
float vec_avg( std::vector<T> vec, Func at)
{
    if (!vec.size()) return 0;
    double ssum = 0;
    ISLOOP (vec) { ssum += at(vec, i); }
    return ssum / vec.size();
}


// Get the min value of a vector
//----------------------------------------------
template <typename T>
T vec_min( std::vector<T> vec )
{
    T res = *(std::min_element(vec.begin(), vec.end()));
    return res;
}

// Gets the max value of a vector
//----------------------------------------------
template <typename T>
T vec_max( std::vector<T> vec )
{
    T res = *(std::max_element(vec.begin(), vec.end()));
    return res;
}

// Calculates the avg delta of a vector
//----------------------------------------------
template <typename T>
T vec_avg_delta( const std::vector<T> &vec)
{
    std::vector<T> deltas;
    ISLOOP (vec) {
        if (!i) continue;
        deltas.push_back( vec[i] - vec[i-1]);
    }
    return vec_avg( deltas);
}

// Calculates the median delta of a vector
//----------------------------------------------
template <typename T>
T vec_median_delta( const std::vector<T> &vec)
{
    std::vector<T> deltas;
    ISLOOP (vec) {
        if (!i) continue;
        deltas.push_back( vec[i] - vec[i-1]);
    }
    return vec_median( deltas);
}


// Partition a vector of elements by class func.
// Return parts as vec of vec.
//---------------------------------------------------------------------
template<typename Func, typename T>
std::vector<std::vector<T> >
partition( std::vector<T> elts, int nof_classes, Func getClass)
{
    // Extract parts
    std::vector<std::vector<T> > res( nof_classes);
    ILOOP (elts.size()) {
        res[getClass( elts[i])].push_back( elts[i]);
    }
    return res;
} // partition()

// Debugging Helpers
//======================

// Print a vector
//----------------------------------
template <typename T>
void print_vec( std::vector<T> v)
{
    std::cout << "( ";
    ISLOOP (v) {
        std::cout << v[i] << ' ';
    }
    std::cout << ")";
}


#endif /* __cplusplus */
#endif /* Common_cpp_hpp */
