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



#include "Ocv.hpp"
#include "Common.hpp"

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

// Get average x of a bunch of points
//-----------------------------------------
float median_x (const Points &p)
{
    std::vector<float> v(p.size());
    ISLOOP (p) { v[i] = p[i].x; }
    std::sort( v.begin(), v.end(), [](float a, float b) { return a < b; });
    return v[v.size()/2];
}

// Get average x of a bunch of points
//-----------------------------------------
float median_y (const Points &p)
{
    std::vector<float> v(p.size());
    ISLOOP (p) { v[i] = p[i].y; }
    std::sort( v.begin(), v.end(), [](float a, float b) { return a < b; });
    return v[v.size()/2];
}

// Return unit vector of p
//------------------------------------
cv::Point2f unit_vector( cv::Point p)
{
    float norm = cv::norm(p);
    return cv::Point2f(p.x / (float)norm, p.y / (float)norm);
}

// Sort points by x coord and remove duplicates.
//-------------------------------------------------
void rem_dups_x( Points &pts, float tol)
{
    std::sort( pts.begin(), pts.end(),
              [](cv::Point a, cv::Point b){ return a.x < b.x; } );
    
    Points res;
    res.push_back( pts[0]);
    ISLOOP (pts) {
        if (i==0) continue;
        float d = cv::norm( pts[i] - pts[i-1]);
        if (d > tol) {
            res.push_back( pts[i]);
        }
    }
    pts = res;
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
    std::vector<int> v(channel.rows*channel.cols);
    int i=0;
    RLOOP( channel.rows) {
        CLOOP( channel.cols) {
            v[i++] = channel.at<uint8_t>(r,c);
        }
    }
    return vec_median( v);
}

// Calculate the q1 value of a single channel
//---------------------------------------------------
int channel_q1( cv::Mat channel )
{
    std::vector<int> v(channel.rows*channel.cols);
    int i=0;
    RLOOP( channel.rows) {
        CLOOP( channel.cols) {
            v[i++] = channel.at<uint8_t>(r,c);
        }
    }
    return vec_q1( v);
}

// Contour
//===========

// Enclose a contour with an n edge polygon
//--------------------------------------------
Points approx_poly( Points cont, int n)
{
    Points hull; // = cont;
    cv::convexHull( cont, hull);
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
    seg1 = polar2segment( line1);
    seg2 = polar2segment( line2);
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
        cv::Vec4f seg = polar2segment( pline);
        segs.push_back(seg);
    }
    return avg_lines( segs);
}

// Return a line segment with median theta 
//-----------------------------------------------------------
cv::Vec4f median_slope_line( const std::vector<cv::Vec2f> &plines )
{
    cv::Vec2f med_theta = vec_median( plines, [](cv::Vec2f line) { return line[1]; });
    //med_theta[0] = 0;
    cv::Vec4f seg = polar2segment( med_theta);
    return seg;
}

// Get a line segment representation of a polar line (rho, theta)
//-------------------------------------------------------------------
cv::Vec4f polar2segment( const cv::Vec2f &pline)
{
    cv::Vec4f result;
    float rho = pline[0], theta = pline[1];
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    result[0] = cvRound(x0 + 1000*(-b));
    result[1] = cvRound(y0 + 1000*(a));
    result[2] = cvRound(x0 - 1000*(-b));
    result[3] = cvRound(y0 - 1000*(a));
    return result;
}

// Line segment to polar, with positive rho
//-----------------------------------------------------------------
cv::Vec2f segment2polar( const cv::Vec4f &line_)
{
    cv::Vec2f pline;
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
    float theta = atan2( dy, dx) + PI/2;
    float rho = fabs(dist_point_line( cv::Point(0,0), line));
    pline[0] = rho;
    pline[1] = theta;
    return pline;
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
    cv::Vec4f line = polar2segment( pline);
    return dist_point_line( p, line);
}

// y given x for polar line
//----------------------------------------
float y_from_x( float x, cv::Vec2f pline)
{
    float res = (pline[0] - x * cos( pline[1])) / sin( pline[1]);
    return res;
}

// x given y for polar line
//----------------------------------------
float x_from_y( float y, cv::Vec2f pline)
{
    float res = (pline[0] - y * sin( pline[1])) / cos( pline[1]);
    return res;
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

// Rotate image by angle. Adjusts image size.
// From PyImageSearch.
//----------------------------------------------------------------
void rot_img( const cv::Mat &img, float angle, cv::Mat &dst)
{
    angle *= -1;
    int h = img.rows;
    int w = img.cols;
    float cX = w / 2;
    float cY = h / 2;
    // Get ro mat
    cv::Mat m = cv::getRotationMatrix2D( Point2f( cX, cY), -angle * 180/PI, 1.0);
    //std::string ttype = mat_typestr( m);
    // Get new img size
    float ccos = fabs( m.at<double>( 0,0));
    float ssin = fabs( m.at<double>( 0,1));
    int nW = int( (h * ssin) + (w * ccos));
    int nH = int( (h * ccos) + (w * ssin));
    // Adjust rot mat for translation
    m.at<double>(0, 2) += (nW / 2) - cX;
    m.at<double>(1, 2) += (nH / 2) - cY;
    // Rotate
    cv::warpAffine( img, dst, m, cv::Size( nW, nH));
}
            
// Get main horizontal direction of a grid of points (in rad)
//-------------------------------------------------------------
float direction (const cv::Mat &img, const Points &ps)
{
    // Draw the points
    cv::Mat canvas = cv::Mat::zeros( cv::Size(img.cols, img.rows), CV_8UC1 );
    ISLOOP (ps) {
        draw_point( ps[i], canvas,1, cv::Scalar(255));
    }
    // Put lines through them
    std::vector<cv::Vec2f> lines;
    const int votes = 10;
    HoughLines(canvas, lines, 1, PI/180, votes, 0, 0 );
    
    // Separate horizontal, vertical, and other lines
    std::vector<std::vector<cv::Vec2f> > horiz_vert_other_lines;
    horiz_vert_other_lines = partition( lines, 3,
                                       [](cv::Vec2f &line) {
                                           const float thresh = 10.0;
                                           float theta = line[1] * (180.0 / PI);
                                           if (fabs(theta - 180) < thresh) return 1;
                                           else if (fabs(theta) < thresh) return 1;
                                           else if (fabs(theta-90) < thresh) return 0;
                                           else return 2;
                                       });
    // Find median theta of vertical lines
    cv::Vec2f med = vec_median( horiz_vert_other_lines[0],
                              [](cv::Vec2f &a) { return a[1]; } );
    return med[1];
}

// Inverse threshold at median
//-----------------------------------------------------------
void inv_thresh_median( const cv::Mat &gray, cv::Mat &dst)
{
    float med = channel_median( gray);
    cv::threshold( gray, dst, med, 1, CV_THRESH_BINARY_INV);
}

// Inverse threshold at q1
//-----------------------------------------------------------
void inv_thresh_q1( const cv::Mat &gray, cv::Mat &dst)
{
    float q1 = channel_q1( gray);
    cv::threshold( gray, dst, q1, 1, CV_THRESH_BINARY_INV);
}

// Inverse threshold at average
//-----------------------------------------------------------
void inv_thresh_avg( const cv::Mat &gray, cv::Mat &dst)
{
    double avg = cv::sum( gray).val[0];
    avg /= (gray.rows * gray.cols);
    cv::threshold( gray, dst, avg, 1, CV_THRESH_BINARY_INV);
}

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
    float cx = ROUND(img.cols / 2.0);
    float cy = ROUND(img.rows / 2.0);
    float dx = ROUND(img.cols / frac);
    float dy = ROUND(img.rows / frac);
    dst = cv::Mat( img, cv::Rect( cx-dx, cy-dy, 2*dx+1, 2*dy+1));
    int area = dst.rows * dst.cols;
    return area;
}

// Average over a center crop of img
//------------------------------------------------------
float center_avg( const cv::Mat &img, float frac)
{
    cv::Mat crop;
    int area = get_center_crop( img, crop, frac);
    double ssum = cv::sum(crop)[0];
    return ssum / area;
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

// Normalize mean and variance for one uint channel,
// scale back to 0..255
//--------------------------------------------------------
void normalize_plane( const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat normed;
    cv::Scalar mmean, sstddev;
    cv::meanStdDev( src, mmean, sstddev);
    src.convertTo( normed, CV_32FC1, 1 / sstddev.val[0] , -mmean.val[0] / sstddev.val[0]);
    double mmin, mmax;
    cv::minMaxLoc( normed, &mmin, &mmax);
    double delta = mmax - mmin;
    double scale = 255.0 / delta;
    double trans = -mmin * scale;
    normed.convertTo( dst, CV_8UC1, scale , trans);
}

// Make sure rect does not extend beyond img
//--------------------------------------------------
void clip_rect( cv::Rect &rect, const cv::Mat &img)
{
    if (rect.x < 0) rect.x = 0;
    if (rect.y < 0) rect.y = 0;
    if (rect.x > img.cols-1) rect.x = img.cols-1;
    if (rect.y > img.rows-1) rect.y = img.rows-1;
    if (rect.x + rect.width > img.cols) {
        rect.width = img.cols - rect.x;
    }
    if (rect.y + rect.height > img.rows) {
        rect.height = img.rows - rect.y;
    }
}

// Normalize nxn submatrices, with mean and var from larger submatrix.
//------------------------------------------------------------------------
void normalize_plane_local( const cv::Mat &src, cv::Mat &dst, int radius)
{
    cv::Mat normed;
    cv::Scalar mmean, sstddev;
    cv::Mat fltmat( src.rows, src.cols, CV_32FC1);
    const int FAC = 4;

    int r = 0;
    while (r < src.rows) {
        int c = 0;
        while (c < src.cols) {
            cv::Rect inner_rect( c-radius, r-radius, 2*radius+1, 2*radius+1);
            clip_rect( inner_rect, src);
            cv::Rect outer_rect( c - FAC*radius, r - FAC*radius, 2*FAC*radius+1, 2*FAC*radius+1);
            clip_rect( outer_rect, src);
            cv::meanStdDev( src( outer_rect), mmean, sstddev);
            src(inner_rect).convertTo( normed, CV_32FC1, 1 / sstddev.val[0] , -mmean.val[0] / sstddev.val[0]);
            //PLOG(">>>>>>>>>> mean %f sigma %f\n", mmean.val[0], sstddev.val[0]);
            normed.copyTo( fltmat( inner_rect) );
            c += 2*radius+1;
        }
        r += 2*radius + 1;
    }
    double mmin, mmax;
    cv::minMaxLoc( fltmat, &mmin, &mmax);
    double delta = mmax - mmin;
    double scale = 255.0 / delta;
    double trans = -mmin * scale;
    fltmat.convertTo( dst, CV_8UC1, scale , trans);
}


// Drawing
//==========

// Draw a point
//--------------------------------------------------------------------
void draw_point( cv::Point p, cv::Mat &img, int r, cv::Scalar col)
{
    cv::circle( img, p, r, col, -1);
}
void draw_point( cv::Point2f p, cv::Mat &img, int r, cv::Scalar col)
{
    cv::Point ip( ROUND(p.x), ROUND(p.y));
    cv::circle( img, ip, r, col, -1);
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
    cv::Vec4f seg = polar2segment( pline);
    cv::Point pt1( seg[0], seg[1]), pt2( seg[2], seg[3]);
    cv::line( dst, pt1, pt2, col, 1, CV_AA);
}

// Get a changing color
//----------------------------------
cv::Scalar get_color( bool reset)
{
    static int idx = 0;
    if (reset) { idx =0; return cv::Scalar(); }
    cv::Scalar cols[] = {
        cv::Scalar( 255,0,0),
        cv::Scalar( 0,255,0),
        cv::Scalar( 0,0,255),
        cv::Scalar( 255,255,0),
        cv::Scalar( 255,0,255),
        cv::Scalar( 0,255,255)
    };
    cv::Scalar res = cols[idx];
    idx++; idx %= 6;
    return res;
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
Points2f points2float( const Points &pi)
{
    return Points2f( pi.begin(), pi.end());
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
            print_vecf( clust[i]);
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
    hline = segment2polar( line);
    if (hline[0] < 0) {
        std::cerr << "Oops 1\n";
    }
    line = cv::Vec4f( 0, 1, 2, 0.9);
    hline = segment2polar( line);
    if (hline[0] < 0) {
        std::cerr << "Oops 2\n";
    }
    // vertical down up
    line = cv::Vec4f( 1, 1, 1.1, 3);
    hline = segment2polar( line);
    if (hline[0] < 0) {
        std::cerr << "Oops 3\n";
    }
    line = cv::Vec4f( 1, 1 , 0.9, 3);
    hline = segment2polar( line);
    if (hline[0] < 0) {
        std::cerr << "Oops 4\n";
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

// Print double matrix
//---------------------------------
void printMatD( const cv::Mat &m)
{
    RLOOP (m.rows) {
        printf("\n");
        CLOOP (m.cols) {
            printf("%8.2f",m.at<double>(r,c) );
        }
    }
    printf("\n========================\n");
}

