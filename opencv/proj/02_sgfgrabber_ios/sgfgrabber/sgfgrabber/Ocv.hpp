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

#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs/ios.h>
#include <iostream>
#include <vector>

#include "Common.hpp"

typedef cv::Point2f Point2f;
typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
typedef cv::Point Line[2];
typedef std::vector<cv::Point2f> Points2f;


extern cv::RNG rng;

// Point
//=========
// Get average x of a bunch of points
float avg_x (const Points &p);
// Get average y of a bunch of points
float avg_y (const Points &p);
// Return unit vector of p
cv::Point2f unit_vector( cv::Point p);

// Matrix
//==========
// Get the type string of a matrix
std::string mat_typestr( const cv::Mat &m);
// Calculate the median value of a single channel
int channel_median( cv::Mat channel );

// Contour
//=========
// Enclose a contour with an n edge polygon
Points approx_poly( Points cont, int n);
// Draw contour in random colors
void draw_contours( const Contours cont, cv::Mat &dst);

// Line
//=======
// Stretch a line by factor, on both ends
Points stretch_line(Points line, float factor );
// Stretch a line by factor, on both ends
cv::Vec4f stretch_line(cv::Vec4f line, float factor );
// Angle between line segments
float angle_between_lines( cv::Point pa, cv::Point pe,
                          cv::Point qa, cv::Point qe);
// Intersection of two lines defined by point pairs
Point2f intersection( cv::Vec4f line1, cv::Vec4f line2);
// Intersection of polar lines (rho, theta)
Point2f intersection( cv::Vec2f line1, cv::Vec2f line2);
// Average line segs by fitting a line thu the endpoints
cv::Vec4f avg_lines( const std::vector<cv::Vec4f> &lines );
// Average polar lines after setting rho to zero and conv to seg
cv::Vec4f avg_slope_line( const std::vector<cv::Vec2f> &plines );
// Get a line segment representation of a polar line (rho, theta)
void polar2segment( const cv::Vec2f &pline, cv::Vec4f &result);
// Line segment to polar, with positive rho
void segment2polar( const cv::Vec4f &line_, cv::Vec2f &pline);
// Fit a line through points, L2 norm
cv::Vec4f fit_line( const Points &p);
// Length of a line segment
float line_len( cv::Point p, cv::Point q);
// Distance between point and line segment
float dist_point_line( cv::Point p, const cv::Vec4f &line);
// Distance between point and polar line
float dist_point_line( cv::Point p, const cv::Vec2f &pline);

// Quad
//========
// Stretch quadrangle by factor
Points2f stretch_quad( Points quad, float factor);
cv::Mat zoom_quad( const cv::Mat &img, cv::Mat &warped, Points2f pts);
// Return whole image as a quad
Points whole_img_quad( const cv::Mat &img);
// Find smallest quad among a few
Points smallest_quad( std::vector<Points> quads);
// Average the corners of quads
Points avg_quad( std::vector<Points> quads);

// Image
//========
// Resize image such that min(width,height) = sz
void resize(const cv::Mat &src, cv::Mat &dst, int sz);
// Automatic edge detection without parameters (from PyImageSearch)
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33);
// Dilate then erode for some iterations
void morph_closing( cv::Mat &m, cv::Size sz, int iterations, int type = cv::MORPH_RECT );
// Get a center crop of an image
int get_center_crop( const cv::Mat &img, cv::Mat &dst, float frac=4);
// Average over a center crop of img
float center_avg( const cv::Mat &img, float frac=4);
// Normalize mean and variance, per channel
void normalize_image( const cv::Mat &src, cv::Mat &dst);

// Drawing
//==========
// Draw a point on an image
void draw_point( cv::Point p, cv::Mat &img, int r=1, cv::Scalar col = cv::Scalar(255,0,0));
// Draw several points on an image
void draw_points( Points p, cv::Mat &img, int r=1, cv::Scalar col = cv::Scalar(255,0,0));
// Draw a line segment
void draw_line( const cv::Vec4f &line, cv::Mat &dst, cv::Scalar col = cv::Scalar(255,0,0));
// Draw several line segments
void draw_lines( const std::vector<cv::Vec4f> &lines, cv::Mat &dst,
                cv::Scalar col = cv::Scalar(255,0,0));
// Draw a polar line (rho, theta)
void draw_polar_line( cv::Vec2f pline, cv::Mat &dst,
                     cv::Scalar col = cv::Scalar(255,0,0));
// Draw several polar lines (rho, theta)
void draw_polar_lines( std::vector<cv::Vec2f> plines, cv::Mat &dst,
                      cv::Scalar col = cv::Scalar(255,0,0));


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
void test_segment2polar();

//===================
// Templates below
//===================

// Point
//=========

// Intersection of two line segments AB CD
//-----------------------------------------------------------------
template <typename Point_>
Point2f intersection( Point_ A, Point_ B, Point_ C, Point_ D)
{
    // Line AB represented as a1x + b1y = c1
    double a1 = B.y - A.y;
    double b1 = A.x - B.x;
    double c1 = a1*(A.x) + b1*(A.y);
    
    // Line CD represented as a2x + b2y = c2
    double a2 = D.y - C.y;
    double b2 = C.x - D.x;
    double c2 = a2*(C.x)+ b2*(C.y);
    
    double determinant = a1*b2 - a2*b1;
    
    if (determinant == 0) { // The lines are parallel.
        return Point_(10E9, 10E9);
    }
    else
    {
        double x = (b2*c1 - b1*c2)/determinant;
        double y = (a1*c2 - a2*c1)/determinant;
        return Point_( x, y);
    }
}

// Get center of a bunch of points
//-----------------------------------------------------------------
template <typename Points_>
cv::Point2f get_center( const Points_ ps)
{
    double avg_x = 0, avg_y = 0;
    ISLOOP (ps) {
        avg_x += ps[i].x;
        avg_y += ps[i].y;
    }
    return cv::Point2f( avg_x / ps.size(), avg_y / ps.size());
}

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

// Order four points clockwise
//----------------------------------------
template <typename POINTS>
POINTS order_points( POINTS &points)
{
    POINTS top_bottom = points;
    std::sort( top_bottom.begin(), top_bottom.end(), [](cv::Point2f a, cv::Point2f b){ return a.y < b.y; });
    POINTS top( top_bottom.begin(), top_bottom.begin()+2 );
    POINTS bottom( top_bottom.end()-2, top_bottom.end());
    std::sort( top.begin(), top.end(), [](cv::Point2f a, cv::Point2f b){ return a.x < b.x; });
    std::sort( bottom.begin(), bottom.end(), [](cv::Point2f a, cv::Point2f b){ return b.x < a.x; });
    POINTS res = top;
    res.insert(res.end(), bottom.begin(), bottom.end());
    return res;
}

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