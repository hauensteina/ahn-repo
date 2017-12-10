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


#ifndef Common_hpp
#define Common_hpp

//=================================
// Common between C++ and Obj C
//=================================

#define ILOOP(n) for (int i=0; i < (n); i++ )
#define JLOOP(n) for (int j=0; j < (n); j++ )
#define KLOOP(n) for (int k=0; k < (n); k++ )
#define RLOOP(n) for (int r=0; r < (n); r++ )
#define CLOOP(n) for (int c=0; c < (n); c++ )

// Make sure size is signed
#define SZ(x) int((x).size())

#define ISLOOP(n) for (int i=0; i < ((n).size()); i++ )
#define JSLOOP(n) for (int j=0; j < ((n).size()); j++ )
#define KSLOOP(n) for (int k=0; k < ((n).size()); k++ )
#define RSLOOP(n) for (int r=0; r < ((n).size()); r++ )
#define CSLOOP(n) for (int c=0; c < ((n).size()); c++ )

#define SIGN(x) ((x)>=0?1:-1)
#define ROUND(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))
#define RAT(a,b) ((b)!=0?((a)/(b)):0)
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

#define PLOG(format,args...) printf("" format "",## args)

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

// Count inversions in a vector of int
//-----------------------------------------------
inline int count_inversions ( std::vector<int> a)
{
    int count = 0;
    ISLOOP (a) {
        for (int j = i+1; j < SZ(a); j++) {
            if (a[i] > a[j]) count++;
        }
    }
    return count;
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

// Append a vector in place
//----------------------------------------------------
template <typename T>
void vec_append( std::vector<T> a, std::vector<T> b)
{
    a.insert(a.end(), b.begin(), b.end());
}

// Append elt to vector, remove elts from front until length <= N
//------------------------------------------------------------------
template <typename T>
void ringpush( std::vector<T>& v, const T &elt, int N)
{
    v.push_back( elt);
    int erase_n = SZ(v) - N;
    if (erase_n > 0) {
        v.erase( v.begin(), v.begin() + erase_n);
    }
}

// Flatten a vector of vectors into a vector
// [[1,2,3],[4,5,6],...] -> [1,2,3,4,5,6,...]
//--------------------------------------------
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v)
{
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); 
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

// Median value of a vector, with acces func
//----------------------------------------------
template <typename T, typename Func>
T vec_median( std::vector<T> vec, Func at)
{
    if (!vec.size()) return T();
    std::sort( vec.begin(), vec.end(),
              [at](T a, T b) { return at(a) < at(b); });
    return vec[vec.size() / 2];
}

// Median value of a vector
//---------------------------------
template <typename T>
T vec_median( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    std::sort( vec.begin(), vec.end(), [](T a, T b) { return a < b; });
    return vec[vec.size() / 2];
}

// Bottom quartile
//---------------------------------
template <typename T>
T vec_q1( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    std::sort( vec.begin(), vec.end(), [](T a, T b) { return a < b; });
    return vec[vec.size() / 4];
}

// Top quartile
//---------------------------------
template <typename T>
T vec_q3( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    std::sort( vec.begin(), vec.end(), [](T a, T b) { return a < b; });
    return vec[(3 * vec.size()) / 4];
}


// Variance (sigma**2) of a vector.
// Welford's algorithm.
//----------------------------------
template <typename T>
T vec_var( std::vector<T> samples)
{
    double M = 0;
    double oldM = 0;
    double S = 0;
    int N = SZ(samples);
    if (!N) return 0;
    for (int k=1; k <= N; k++) {
        double x = samples[k-1];
        oldM = M;
        M += (x-M) / k;
        S += (x-M)*(x-oldM);
    }
    return S / N;
}

// Variance (sigma**2) of a vector.
//-------------------------------------
template <typename T>
T vec_var_ref( std::vector<T> samples)
{
    double mean = 0;
    double sqmean = 0;
    ISLOOP (samples) {
        mean += samples[i];
        sqmean += samples[i] * samples[i];
    }
    mean /= SZ(samples);
    sqmean /= SZ(samples);
    return sqmean - mean;
}

// Sum of square dist from median
//-------------------------------------
template <typename T>
T vec_var_med( std::vector<T> samples)
{
    double med = vec_median( samples);
    double sqmean = 0;
    ISLOOP (samples) {
        double tt = samples[i] - med;
        sqmean += tt*tt;
    }
    sqmean /= SZ(samples);
    return sqmean;
}

// Avg value of a vector
//------------------------------
template <typename T>
T vec_avg( std::vector<T> vec)
{
    if (!vec.size()) return T(0);
    double ssum = 0;
    ISLOOP (vec) { ssum += vec[i]; }
    return T(ssum / vec.size());
}

// Avg value of a vector, with acces func
//---------------------------------------------
template <typename T, typename Func>
float vec_avg( std::vector<T> vec, Func at)
{
    if (!vec.size()) return 0;
    double ssum = 0;
    ISLOOP (vec) { ssum += at(vec[i]); }
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

// The deltas of a vector
//----------------------------------------------
template <typename T>
std::vector<T> vec_delta( const std::vector<T> &vec)
{
    std::vector<T> deltas;
    ISLOOP (vec) {
        if (!i) continue;
        deltas.push_back( vec[i] - vec[i-1]);
    }
    return deltas;
}

// The ratios of a vector
//-------------------------------------------------
template <typename T>
std::vector<T> vec_rat( const std::vector<T> &vec)
{
    std::vector<T> rats;
    ISLOOP (vec) {
        if (!i) continue;
        rats.push_back( vec[i-1] != 0 ? vec[i] / (float) vec[i-1] : 0);
    }
    return rats;
}

// Extract a vector of floats from a vector of some type
//---------------------------------------------------------------------
template <typename T, typename F>
std::vector<float> vec_extract(  const std::vector<T> &vec, F getter)
{
    std::vector<float> res(SZ(vec));
    ISLOOP (vec) {
        res[i] = getter( vec[i]);
    }
    return res;
}

// Find index of closest element in a vec
//---------------------------------------------------------------------
template <typename T>
int vec_closest(  const std::vector<T> &vec, T num)
{
    float mindist = 1E9;
    int minidx = -1;
    ISLOOP (vec) {
        if (fabs( vec[i] - num) < mindist) {
            mindist = fabs( vec[i] - num);
            minidx = i;
        }
    }
    return minidx;
}

// Index of larget element in vector
//-----------------------------------------
template <typename T>
int argmax( const std::vector<T> &vec)
{
    auto maxiter = std::max_element( vec.begin(), vec.end());
    return int(maxiter - vec.begin());
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

// Debugger Helpers
//======================

// Print a vector
void print_vecf( std::vector<float> v);
void print_veci( std::vector<int> v);


#endif /* __cplusplus */
#endif /* Common_hpp */
