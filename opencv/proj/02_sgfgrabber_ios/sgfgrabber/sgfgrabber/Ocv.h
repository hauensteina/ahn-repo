//
//  Ocv.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

//=======================
// OpenCV helper funcs
//=======================

#ifndef Ocv_hpp
#define Ocv_hpp


#ifdef __cplusplus

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <iostream>
#import <vector>

#import "Common.h"

typedef cv::Point2f Point2f;
typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
typedef cv::Point Line[2];
typedef std::vector<cv::Point2f> Points2f;


extern cv::RNG rng;

// Matrix
//==========
// Get the type string of a matrix
std::string mat_typestr( const cv::Mat &m);

// Lines
//=======
// Average line segs by fitting a line thu the endpoints
cv::Vec4f avg_lines( const std::vector<cv::Vec4f> &lines );
// Average polar lines after setting rho to zero and conv to seg
cv::Vec4f avg_slope_line( const std::vector<cv::Vec2f> &plines );
// Get a line segment representation of a polar line (rho, theta)
void polarToSegment( const cv::Vec2f &pline, cv::Vec4f &result);
// Line segment to polar, with positive rho
void segmentToPolar( const cv::Vec4f &line_, cv::Vec2f &pline);
// Fit a line through points, L2 norm
cv::Vec4f fit_line( const Points &p);
// Length of a line segment
float line_len( cv::Point p, cv::Point q);
// Distance between point and line segment
float dist_point_line( cv::Point p, const cv::Vec4f &line);
// Distance between point and polar line
float dist_point_line( cv::Point p, const cv::Vec2f &pline);

// Type Conversions
//====================
// Vector of int points to float
void points2float( const Points &pi, Points2f &pf);
// Vector of float points to int
void points2int( const Points2f &pf, Points &pi);

// Debugging
//=============
// Print matrix type
void print_mat_type( const cv::Mat &m);
// Print uint8 matrix
void printMatU( const cv::Mat &m);
// Print float matrix
void printMatF( const cv::Mat &m);

// Misc
//========
std::string opencvVersion();
void test_mcluster();


//===================
// Templates below
//===================

// Contours
//=============

// Draw one contour (e.g. the board)
//------------------------------------
template <typename Points_>
void draw_contour( cv::Mat &img, const Points_ &cont,
                 cv::Scalar color = cv::Scalar(255,0,0), int thickness = 1)
{
    cv::drawContours( img, std::vector<Points_>( 1, cont), -1, color, thickness, 8);
}

// Clustering
//=============

// Cluster a vector of elements by func.
// Return clusters as vec of vec.
// Assumes feature is a single float.
//---------------------------------------------------------------------
template<typename Func, typename T>
std::vector<std::vector<T> >
cluster (std::vector<T> elts, int nof_clust, Func getFeature)
{
    if (elts.size() < 2) return std::vector<std::vector<T> >();
    std::vector<float> features;
    std::vector<float> centers;
    ILOOP (elts.size()) { features.push_back( getFeature( elts[i])); }
    std::vector<int> labels;
    cv::kmeans( features, nof_clust, labels,
               cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    // Extract parts
    std::vector<std::vector<T> > res( nof_clust, std::vector<T>());
    ILOOP (elts.size()) {
        res[labels[i]].push_back( elts[i]);
    }
    return res;
} // cluster()

// Cluster a vector of elements by func.
// Return clusters as vec of vec.
// Assumes feature is a vec of float of ndims.
//-----------------------------------------------------------------------
template<typename Func, typename T>
std::vector<std::vector<T> >
mcluster (std::vector<T> elts, int nof_clust, int ndims, double &compactness, Func getFeatVec)
{
    if (elts.size() < 2) return std::vector<std::vector<T> >();
    std::vector<float> featVec;
    // Append all vecs into one large one
    ILOOP (elts.size()) {
        //size_t n1 = featVec.size();
        vapp( featVec, getFeatVec( elts[i]));
        //size_t n2 = featVec.size();
    }
    // Reshape into a matrix with one row per feature vector
    //cv::Mat m = cv::Mat(featVec).reshape( 0, sizeof(elts) );
    //assert (featVec.size() == 361*ndims);
    cv::Mat m = cv::Mat(featVec).reshape( 0, int(elts.size()));
    
    // Cluster
    std::vector<int> labels;
    cv::Mat centers;
    compactness = cv::kmeans( m, nof_clust, labels,
                             cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0),
                             3, cv::KMEANS_PP_CENTERS, centers);
    // Extract parts
    std::vector<std::vector<T> > res( nof_clust, std::vector<T>());
    ILOOP (elts.size()) {
        res[labels[i]].push_back( elts[i]);
    }
    return res;
} // mcluster()


#endif /* __clusplus */

#endif /* Ocv_hpp */
