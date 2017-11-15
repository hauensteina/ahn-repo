//
//  Ocv.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

//=======================
// OpenCV helper funcs
//=======================



#import "Ocv.h"
#import "Common.h"

cv::RNG rng(12345); // random number generator

// Point
//========

// Get average x of a bunch of points
//-----------------------------------------
float avg_x (const Points &p)
{
    double ssum = 0.0;
    ISLOOP (p) { ssum += p[i].x; }
    return ssum / p.size();
}

// Get average y of a bunch of points
//-----------------------------------------
float avg_y (const Points &p)
{
    double ssum = 0.0;
    ISLOOP (p) { ssum += p[i].y; }
    return ssum / p.size();
}

// Return unit vector of p
//------------------------------------
cv::Point2f unit_vector( cv::Point p)
{
    float norm = cv::norm(p);
    return cv::Point2f(p.x / (float)norm, p.y / (float)norm);
}

// Matrix
//===========

// Get the type string of a matrix
//------------------------------------------
std::string mat_typestr( const cv::Mat &m)
{
    int type = m.type();
    std::string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}

// Calculate the median value of a single channel
//---------------------------------------------------
int channel_median( cv::Mat channel )
{
    cv::Mat flat = channel.reshape(1,1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);
    double res = sorted.at<uchar>(sorted.size() / 2);
    return res;
}

// Contour
//===========

// Enclose a contour with an n edge polygon
//--------------------------------------------
Points approx_poly( Points cont, int n)
{
    Points hull = cont;
    //cv::convexHull( cont, hull);
    float peri = cv::arcLength( hull, true);
    float epsilon = bisect(
                           [hull,peri](float x) {
                               Points approx;
                               cv::approxPolyDP( hull, approx, x*peri, true);
                               return -approx.size();
                           },
                           0.0, 1.0, -n);
    Points res;
    cv::approxPolyDP( hull, res, epsilon*peri, true);
    return res;
}

// Draw contour in random colors
//-------------------------------------------------------
void draw_contours( const Contours cont, cv::Mat &dst)
{
    // Draw contours
    for( int i = 0; i< cont.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
        drawContours( dst, cont, i, color, 2, 8);
    }
} // draw_contours()


// Line
//=========

// Stretch a line by factor, on both ends
//--------------------------------------------------
Points stretch_line(Points line, float factor )
{
    cv::Point p0 = line[0];
    cv::Point p1 = line[1];
    float length = line_len( p0, p1);
    cv::Point v = ((factor-1.0) * length) * unit_vector(p1-p0);
    Points res = {p0-v , p1+v};
    return res;
}

// Stretch a line by factor, on both ends
//----------------------------------------------------
cv::Vec4f stretch_line(cv::Vec4f line, float factor )
{
    const cv::Point p0( line[0], line[1]);
    const cv::Point p1( line[2], line[3]);
    float length = line_len( p0, p1);
    const cv::Point v = ((factor-1.0) * length) * unit_vector(p1-p0);
    cv::Vec4f res;
    res[0] = (p0-v).x;
    res[1] = (p0-v).y;
    res[2] = (p1+v).x;
    res[3] = (p1+v).y;
    return res;
}

//----------------------------------------------------
float angle_between_lines( cv::Point pa, cv::Point pe,
                          cv::Point qa, cv::Point qe)
{
    cv::Point2f v1 = unit_vector( cv::Point( pe - pa) );
    cv::Point2f v2 = unit_vector( cv::Point( qe - qa) );
    float dot = v1.x * v2.x + v1.y * v2.y;
    if (dot < -1) dot = -1;
    if (dot > 1) dot = 1;
    return std::acos(dot);
}

// Intersection of two lines defined by point pairs
//----------------------------------------------------------
Point2f intersection( cv::Vec4f line1, cv::Vec4f line2)
{
    return intersection( cv::Point2f( line1[0], line1[1]),
                        cv::Point2f( line1[2], line1[3]),
                        cv::Point2f( line2[0], line2[1]),
                        cv::Point2f( line2[2], line2[3]));
}

// Intersection of polar lines (rho, theta)
//---------------------------------------------------------
Point2f intersection( cv::Vec2f line1, cv::Vec2f line2)
{
    cv::Vec4f seg1, seg2;
    polar2segment( line1, seg1);
    polar2segment( line2, seg2);
    return intersection( seg1, seg2);
}


// Average a bunch of line segments by
// fitting a line through all the endpoints
//---------------------------------------------------------
cv::Vec4f avg_lines( const std::vector<cv::Vec4f> &lines )
{
    // Get the points
    Points2f points;
    ILOOP (lines.size()) {
        cv::Point2f p1(lines[i][0], lines[i][1]);
        cv::Point2f p2(lines[i][2], lines[i][3]);
        points.push_back( p1);
        points.push_back( p2);
    }
    // Put a line through them
    cv::Vec4f lparms;
    cv::fitLine( points, lparms, CV_DIST_L2, 0.0, 0.01, 0.01);
    cv::Vec4f res;
    res[0] = lparms[2];
    res[1] = lparms[3];
    res[2] = lparms[2] + lparms[0];
    res[3] = lparms[3] + lparms[1];
    return res;
} // avg_lines()

// Take a bunch of polar lines, set rho to zero, turn into
// segments, return avg line segment
//-----------------------------------------------------------
cv::Vec4f avg_slope_line( const std::vector<cv::Vec2f> &plines )
{
    std::vector<cv::Vec4f> segs;
    cv::Vec2f pline;
    ISLOOP (plines) {
        pline = plines[i];
        pline[0] = 0;
        cv::Vec4f seg;
        polar2segment( pline, seg);
        segs.push_back(seg);
    }
    return avg_lines( segs);
}

// Get a line segment representation of a polar line (rho, theta)
//----------------------------------------------------------------
void polar2segment( const cv::Vec2f &pline, cv::Vec4f &result)
{
    float rho = pline[0], theta = pline[1];
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    result[0] = cvRound(x0 + 1000*(-b));
    result[1] = cvRound(y0 + 1000*(a));
    result[2] = cvRound(x0 - 1000*(-b));
    result[3] = cvRound(y0 - 1000*(a));
}

// Line segment to polar, with positive rho
//-----------------------------------------------------------------
void segment2polar( const cv::Vec4f &line_, cv::Vec2f &pline)
{
    cv::Vec4f line = line_;
    // Always go left to right
    if (line[2] < line[0]) {
        swap( line[0], line[2]);
        swap( line[1], line[3]);
    }
    float dx = line[2] - line[0];
    float dy = line[3] - line[1];
    if (fabs(dx) > fabs(dy)) { // horizontal
        if (dx < 0) { dx *= -1; dy *= -1; }
    }
    else { // vertical
        if (dy > 0) { dx *= -1; dy *= -1; }
    }
    float theta = atan2( dy, dx) + CV_PI/2;
    float rho = fabs(dist_point_line( cv::Point(0,0), line));
    pline[0] = rho;
    pline[1] = theta;
}

// Fit a line through points, L2 norm
//--------------------------------------
cv::Vec4f fit_line( const Points &p)
{
    cv::Vec4f res,tt;
    cv::fitLine( p, tt, CV_DIST_L2, 0.0, 0.01, 0.01);
    res[0] = tt[2];
    res[1] = tt[3];
    res[2] = tt[2] + tt[0];
    res[3] = tt[3] + tt[1];
    return res;
}

// Length of a line segment
//---------------------------------------------------------
float line_len( cv::Point p, cv::Point q)
{
    return cv::norm( q-p);
}

// Distance between point and line segment
//----------------------------------------------------------
float dist_point_line( cv::Point p, const cv::Vec4f &line)
{
    float x = p.x;
    float y = p.y;
    float x0 = line[0];
    float y0 = line[1];
    float x1 = line[2];
    float y1 = line[3];
    float num = (y0-y1)*x + (x1-x0)*y + (x0*y1 - x1*y0);
    float den = sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    return num / den;
}

// Distance between point and polar line
//----------------------------------------------------------
float dist_point_line( cv::Point p, const cv::Vec2f &pline)
{
    cv::Vec4f line;
    polar2segment( pline, line);
    return dist_point_line( p, line);
}

// Quad
//=======
// Stretch quadrangle by factor
//--------------------------------------------------
Points2f stretch_quad( Points quad, float factor)
{
    quad = order_points( quad);
    Points diag1_stretched = stretch_line( { quad[0],quad[2] }, factor);
    Points diag2_stretched = stretch_line( { quad[1],quad[3] }, factor);
    Points2f res = { diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1] };
    return res;
}
// Zoom into a quadrangle
//--------------------------------------------------------
cv::Mat zoom_quad( const cv::Mat &img, cv::Mat &warped, Points2f pts)
{
    Points2f rect = order_points(pts);
    Points2f dst = {
        cv::Point(0,0),
        cv::Point(img.cols - 1, 0),
        cv::Point(img.cols - 1, img.cols - 1),
        cv::Point(0, img.cols - 1) };
    
    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat res;
    cv::warpPerspective(img, warped, M, cv::Size(img.cols, img.rows));
    return M;
}
// Return whole image as a quad
//---------------------------------------------
Points whole_img_quad( const cv::Mat &img)
{
    Points res = { cv::Point(1,1), cv::Point(img.cols-2,1),
        cv::Point(img.cols-2,img.rows-2), cv::Point(1,img.rows-2) };
    return res;
}

// Find smallest quad among a few
//--------------------------------------------
Points smallest_quad( std::vector<Points> quads)
{
    Points res(4);
    int minidx=0;
    float minArea = 1E9;
    ILOOP (quads.size()) {
        Points b = quads[i];
        float area = cv::contourArea(b);
        if (area < minArea) { minArea = area; minidx = i;}
    }
    return quads[minidx];
}

// Average the corners of quads
//---------------------------------------------------
Points avg_quad( std::vector<Points> quads)
{
    Points res(4);
    ILOOP (quads.size()) {
        Points b = quads[i];
        res[0] += b[0];
        res[1] += b[1];
        res[2] += b[2];
        res[3] += b[3];
    }
    res[0] /= (float)quads.size();
    res[1] /= (float)quads.size();
    res[2] /= (float)quads.size();
    res[3] /= (float)quads.size();
    return res;
}


// Image
//=========
// Automatic edge detection without parameters (from PyImageSearch)
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma)
{
    double v = channel_median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

// Resize image such that min(width,height) = sz
//------------------------------------------------------
void resize(const cv::Mat &src, cv::Mat &dst, int sz)
{
    //cv::Size s;
    int width  = src.cols;
    int height = src.rows;
    float scale;
    if (width < height) scale = sz / (float) width;
    else scale = sz / (float) height;
    cv::resize( src, dst, cv::Size(int(width*scale),int(height*scale)), 0, 0, cv::INTER_AREA);
}

// Dilate then erode for some iterations
//---------------------------------------------------------------------------------------
void morph_closing( cv::Mat &m, cv::Size sz, int iterations, int type)
{
    cv::Mat element = cv::getStructuringElement( type, sz);
    for (int i=0; i<iterations; i++) {
        cv::dilate( m, m, element );
        cv::erode( m, m, element );
    }
}

// Get a center crop of an image
//-------------------------------------------------------------------
int get_center_crop( const cv::Mat &img, cv::Mat &dst, float frac)
{
    float cx = img.cols / 2.0;
    float cy = img.rows / 2.0;
    float dx = img.cols / frac;
    float dy = img.rows / frac;
    dst = cv::Mat( img, cv::Rect( round(cx-dx), round(cy-dy), round(2*dx), round(2*dy)));
    int area = dst.rows * dst.cols;
    return area;
}

// Normalize mean and variance, per channel
//--------------------------------------------------------
void normalize_image( const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat planes[4];
    cv::split( src, planes);
    cv::Scalar mmean, sstddev;
    
    cv::meanStdDev( planes[0], mmean, sstddev);
    planes[0].convertTo( planes[0], CV_32FC1, 1 / sstddev.val[0] , -mmean.val[0] / sstddev.val[0]);
    
    cv::meanStdDev( planes[1], mmean, sstddev);
    planes[1].convertTo( planes[1], CV_32FC1, 1 / sstddev.val[0] , -mmean.val[0] / sstddev.val[0]);
    
    cv::meanStdDev( planes[2], mmean, sstddev);
    planes[2].convertTo( planes[2], CV_32FC1, 1 / sstddev.val[0] , -mmean.val[0] / sstddev.val[0]);
    
    // ignore channel 4, that's alpha
    cv::merge( planes, 3, dst);
}

// Drawing
//==========

// Draw a point
//--------------------------------------------------------------------
void draw_point( cv::Point p, cv::Mat &img, int r, cv::Scalar col)
{
    cv::circle( img, p, r, col, -1);
}

// Draw several points
//----------------------------------------------------------------
void draw_points( Points p, cv::Mat &img, int r, cv::Scalar col)
{
    ISLOOP( p) draw_point( p[i], img, r, col);
}

// Draw a line segment
//-------------------------------------------------------------------------------------------
void draw_line( const cv::Vec4f &line, cv::Mat &dst, cv::Scalar col)
{
    cv::Point pt1, pt2;
    pt1.x = cvRound(line[0]);
    pt1.y = cvRound(line[1]);
    pt2.x = cvRound(line[2]);
    pt2.y = cvRound(line[3]);
    cv::line( dst, pt1, pt2, col, 1, CV_AA);
}

// Draw several line segments
//--------------------------------------------------------------------
void draw_lines( const std::vector<cv::Vec4f> &lines, cv::Mat &dst,
                cv::Scalar col)
{
    ISLOOP (lines) draw_line( lines[i], dst, col);
}


// Draw a polar line (rho, theta)
//----------------------------------------------------------
void draw_polar_line( cv::Vec2f pline, cv::Mat &dst,
                     cv::Scalar col)
{
    cv::Vec4f seg;
    polar2segment( pline, seg);
    cv::Point pt1( seg[0], seg[1]), pt2( seg[2], seg[3]);
    cv::line( dst, pt1, pt2, col, 1, CV_AA);
}

// Draw several polar lines (rho, theta)
//-------------------------------------------------------------------
void draw_polar_lines( std::vector<cv::Vec2f> plines, cv::Mat &dst,
                      cv::Scalar col)
{
    ISLOOP (plines) { draw_polar_line( plines[i], dst, col); }
}


// Type Conversions
//===================

// Vector of int points to float
//--------------------------------------------------
void points2float( const Points &pi, Points2f &pf)
{
    pf = Points2f( pi.begin(), pi.end());
}

// Vector of float points to int
//--------------------------------------------------
void points2int( const Points2f &pf, Points &pi)
{
    pi = Points( pf.begin(), pf.end());
}


// Misc
//========

//----------------------------
std::string opencvVersion()
{
    std::ostringstream out;
    out << "OpenCV version: " << CV_VERSION;
    return out.str();
}


// How to use mcluster()
//------------------------
void test_mcluster()
{
    std::vector<float> v1 = { 1, 2 };
    std::vector<float> v2 = { 3, 4  };
    std::vector<float> v3 = { 10, 20 };
    std::vector<float> v4 = { 11, 21 };
    std::vector<float> v5 = { 30, 40 };
    std::vector<float> v6 = { 31, 41 };
    std::vector<std::vector<float> > samples;
    samples.push_back( v1);
    samples.push_back( v2);
    samples.push_back( v3);
    samples.push_back( v4);
    samples.push_back( v5);
    samples.push_back( v6);
    
    double compactness;
    auto res = mcluster( samples, 3, 2, compactness,
                        [](std::vector<float>s) {return s;} );
    CSLOOP (res) {
        std::cout << "Cluster " << c << ":\n";
        std::vector<std::vector<float> > clust = res[c];
        ISLOOP (clust) {
            print_vec( clust[i]);
        }
        std::cout << "\n";
    }
    return;
}

// How to use segmentToPolar()
//------------------------------
void test_segment2polar()
{
    cv::Vec4f line;
    cv::Vec2f hline;
    
    // horizontal
    line = cv::Vec4f( 0, 1, 2, 1.1);
    segment2polar( line, hline);
    if (hline[0] < 0) {
        NSLog(@"Oops 1");
    }
    line = cv::Vec4f( 0, 1, 2, 0.9);
    segment2polar( line, hline);
    if (hline[0] < 0) {
        NSLog(@"Oops 2");
    }
    // vertical down up
    line = cv::Vec4f( 1, 1, 1.1, 3);
    segment2polar( line, hline);
    if (hline[0] < 0) {
        NSLog(@"Oops 3");
    }
    line = cv::Vec4f( 1, 1 , 0.9, 3);
    segment2polar( line, hline);
    if (hline[0] < 0) {
        NSLog(@"Oops 4");
    }
}

// Debuggging
//=============

// Print matrix type
//---------------------------------------
void print_mat_type( const cv::Mat &m)
{
    std::cout << mat_typestr( m) << std::endl;
    printf("\n========================\n");
}

// Print uint8 matrix
//---------------------------------
void printMatU( const cv::Mat &m)
{
    RLOOP (m.rows) {
        printf("\n");
        CLOOP (m.cols) {
            printf("%4d",m.at<uint8_t>(r,c) );
        }
    }
    printf("\n========================\n");
}

// Print float matrix
//---------------------------------
void printMatF( const cv::Mat &m)
{
    RLOOP (m.rows) {
        printf("\n");
        CLOOP (m.cols) {
            printf("%8.2f",m.at<float>(r,c) );
        }
    }
    printf("\n========================\n");
}

