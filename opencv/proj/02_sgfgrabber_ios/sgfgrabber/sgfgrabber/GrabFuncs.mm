//
//  GrabFuncs.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

//#include <math.h>
#include <type_traits>
#import <opencv2/opencv.hpp>
//#import <opencv2/core/ptr.inl.hpp>
#import <opencv2/imgcodecs/ios.h>
//#import <opencv2/imgproc/imgproc.hpp>
#import "Common.h"
#import "GrabFuncs.h"

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


typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
typedef cv::Point Line[2];
typedef std::vector<cv::Point2f> Points2f;
static cv::RNG rng(12345);
double PI = M_PI;
typedef std::complex<double> cplx;
cplx I(0.0, 1.0);
const cv::Size TMPL_SZ(16,16);

#define STRETCH_FACTOR 1.1

@interface GrabFuncs()
//=======================
@property cv::Mat small; // resized image, in color
@property cv::Mat gray;  // Grayscale version of small
@property cv::Mat m;     // Mat with image we are working on
@property Contours cont; // Current set of contours
@property Points board;  // Current hypothesis on where the board is
@property Points board_zoomed; // board corners after zooming in
@property int board_sz; // board size, 9 or 19
@property Points intersections; // locations of line intersections (81,361)
@property int delta_v; // approx vertical line dist
@property int delta_h; // approx horiz line dist
@property Points stone_or_empty; // places where we suspect stones or empty
@property std::vector<Points> horizontal_clusters; // Result of Hough lines and clustering
@property std::vector<Points> vertical_clusters;
@property std::vector<cv::Vec4f> horizontal_lines;
@property float wavelen_h;
@property std::vector<cv::Vec4f> vertical_lines;
@property float wavelen_v;

@property cv::Mat tmpl_black;
@property cv::Mat tmpl_white;
@property cv::Mat tmpl_top_left;
@property cv::Mat tmpl_top_right;
@property cv::Mat tmpl_bot_right;
@property cv::Mat tmpl_bot_left;
@property cv::Mat tmpl_top;
@property cv::Mat tmpl_right;
@property cv::Mat tmpl_bot;
@property cv::Mat tmpl_left;
@property cv::Mat tmpl_inner;
@property cv::Mat tmpl_hoshi;
@end

@implementation GrabFuncs
//=========================

//----------------------
- (instancetype)init
{
    self = [super init];
    if (self) {
        [self jpg:@"black.jpg" toTmpl:_tmpl_black];
        [self jpg:@"white.jpg" toTmpl:_tmpl_white];
        [self jpg:@"top_left.jpg" toTmpl:_tmpl_top_left];
        [self jpg:@"top_right.jpg" toTmpl:_tmpl_top_right];
        [self jpg:@"bot_right.jpg" toTmpl:_tmpl_bot_right];
        [self jpg:@"bot_left.jpg" toTmpl:_tmpl_bot_left];
        [self jpg:@"top.jpg" toTmpl:_tmpl_top];
        [self jpg:@"right.jpg" toTmpl:_tmpl_right];
        [self jpg:@"bot.jpg" toTmpl:_tmpl_bot];
        [self jpg:@"left.jpg" toTmpl:_tmpl_left];
        [self jpg:@"inner.jpg" toTmpl:_tmpl_inner];
        [self jpg:@"hoshi.jpg" toTmpl:_tmpl_hoshi];
    }
    return self;
}

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

//---------------------------------------
void print_mat_type( const cv::Mat &m)
{
    std::cout << mat_typestr( m) << std::endl;
    printf("\n========================\n");
}

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


//--------------------------------------------------
void PointsToFloat( const Points &pi, Points2f &pf)
{
    pf = Points2f( pi.begin(), pi.end());
}

//--------------------------------------------------
void PointsToInt( const Points2f &pf, Points &pi)
{
    pi = Points( pf.begin(), pf.end());
}

// Draw one contour (e.g. the board)
//------------------------------------
template <typename Points_>
void drawContour( cv::Mat &img, const Points_ &cont,
                 cv::Scalar color = cv::Scalar(255,0,0), int thickness = 1)
{
    cv::drawContours( img, std::vector<Points_>( 1, cont), -1, color, thickness, 8);
}

// Prepare image for use as a similarity template
//-------------------------------------------------
void templify( cv::Mat &m)
{
    cv::resize(m,m,TMPL_SZ);
    cv::normalize( m, m, 0 , 255, CV_MINMAX, CV_8UC1);
    //m.convertTo( m, CV_64FC1);
    //m.convertTo( m, CV_8UC1);
    //cv::Mat sq = m.mul(m);
    //double ssum = cv::sum(sq)[0];
    //ssum = sqrt(ssum);
    //m /= ssum;
}

// Compare two templified images of same size
//-------------------------------------------------------
double cmpTmpl( const cv::Mat &m1, const cv::Mat &m2)
{
//    cv::Mat prod = m1.mul(m2);
//    double d = cv::sum(prod)[0];
//    return d;
    cv::Mat diff = m1 - m2;
    cv::Mat sq = diff.mul(diff);
    double d = cv::sum(sq)[0];
    return -d;
}

// Convert jpg file to 16x16 template cv::Mat
//-------------------------------------------------
- (void) jpg:(NSString *)fname toTmpl:(cv::Mat&)m
{
    UIImage *img = [UIImage imageNamed:fname];
    UIImageToMat(img, m);
    templify( m);
}

#pragma mark - General utility funcs
//======================================

//----------------------------
+ (NSString *) opencvVersion
{
    return [NSString stringWithFormat:@"OpenCV version: %s", CV_VERSION];
}

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

// Partition a vector of elements by class func.
// Return parts as vec of vec
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
// Assumes feature is a vec of float.
//-----------------------------------------------------------------------
template<typename Func, typename T>
std::vector<std::vector<T> >
mcluster (std::vector<T> elts, int nof_clust, int ndims, Func getFeatVec)
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
    cv::kmeans( m, nof_clust, labels,
               cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    // Extract parts
    std::vector<std::vector<T> > res( nof_clust, std::vector<T>());
    ILOOP (elts.size()) {
        res[labels[i]].push_back( elts[i]);
    }
    if (res[0].size() == 7 || res[1].size() == 7) {
        int xx=42;
    }
    return res;
} // mcluster()

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

    auto res = mcluster( samples, 3, 2, [](std::vector<float>s) {return s;} );
    CSLOOP (res) {
        std::cout << "Cluster " << c << ":\n";
        std::vector<std::vector<float> > clust = res[c];
        ISLOOP (clust) {
            print_vec( clust[i]);
        }
        std::cout << "\n";
    }
    int tt=42;
    return;
}

// Average a bunch of line segments.
// Put a line through all the endpoints.
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

// Take a bunch of Hough lines, set rho to zero, turn into
// segments, return avg line segment
//-----------------------------------------------------------
cv::Vec4f avg_slope_line( const std::vector<cv::Vec2f> &hlines )
{
    std::vector<cv::Vec4f> segs;
    cv::Vec2f hline;
    ISLOOP (hlines) {
        hline = hlines[i];
        hline[0] = 0;
        cv::Vec4f seg;
        polarToSegment( hline, seg);
        segs.push_back(seg);
    }
    return avg_lines( segs);
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

// Distance between point and Hough line
//----------------------------------------------------------
float dist_point_line( cv::Point p, const cv::Vec2f &hline)
{
    cv::Vec4f line;
    polarToSegment( hline, line);
    return dist_point_line( p, line);
}

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

// Intersection of two line segments
template <typename Point_>
//-----------------------------------------------------------------
cv::Point2f intersection( Point_ A, Point_ B, Point_ C, Point_ D)
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
} // intersection()

// Intersection of two lines defined by point pairs
//----------------------------------------------------------
cv::Point2f intersection( cv::Vec4f line1, cv::Vec4f line2)
{
    return intersection( cv::Point2f( line1[0], line1[1]),
                        cv::Point2f( line1[2], line1[3]),
                        cv::Point2f( line2[0], line2[1]),
                        cv::Point2f( line2[2], line2[3]));
}

// Intersection of Hough lines (rho, theta)
//---------------------------------------------------------
cv::Point2f intersection( cv::Vec2f line1, cv::Vec2f line2)
{
    cv::Vec4f seg1, seg2;
    polarToSegment( line1, seg1);
    polarToSegment( line2, seg2);
    return intersection( seg1, seg2);
}

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
    //return hull;
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

// Calculates the median value of a single channel
//-------------------------------------
int channel_median( cv::Mat channel )
{
    cv::Mat flat = channel.reshape(1,1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);
    double res = sorted.at<uchar>(sorted.size() / 2);
    return res;
}

// Calculates the median value of a vector
//----------------------------------------------
template <typename T>
T vec_median( std::vector<T> vec )
{
    std::sort( vec.begin(), vec.end(), [](T a, T b) { return a < b; });
    T res = vec[vec.size() / 2];
    return res;
}

// Gets the min value of a vector
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


// Calculates the avg value of a vector
//----------------------------------------------
template <typename T>
T vec_avg( std::vector<T> vec )
{
    double ssum = 0;
    ISLOOP (vec) { ssum += vec[i]; }
    return T(ssum / vec.size());
}


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

// Automatic edge detection without parameters
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33)
{
    double v = channel_median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

// Mark a point on an image
//--------------------------------------
void draw_point( cv::Point p, cv::Mat &img, int r=1, cv::Scalar col = cv::Scalar(255,0,0))
{
    cv::circle( img, p, r, col, -1);
}
//-------------------------------------------------------------------------------------------
void draw_points( Points p, cv::Mat &img, int r=1, cv::Scalar col = cv::Scalar(255,0,0))
{
    ISLOOP( p) draw_point( p[i], img, r, col);
}


#pragma mark - Pipeline Helpers
//==================================

// Reject board if opposing lines not parallel
// or adjacent lines not at right angles
//----------------------------------------------
bool board_valid( Points board, float screenArea)
{
    if (board.size() != 4) return false;
    float area = cv::contourArea(board);
    if (area / screenArea > 0.95) return false;
    if (area / screenArea < 0.20) return false;

    float par_ang1   = (180.0 / M_PI) * angle_between_lines( board[0], board[1], board[3], board[2]);
    float par_ang2   = (180.0 / M_PI) * angle_between_lines( board[0], board[3], board[1], board[2]);
    float right_ang1 = (180.0 / M_PI) * angle_between_lines( board[0], board[1], board[1], board[2]);
    float right_ang2 = (180.0 / M_PI) * angle_between_lines( board[0], board[3], board[3], board[2]);
    //float horiz_ang   = (180.0 / M_PI) * angle_between_lines( board[0], board[1], cv::Point(0,0), cv::Point(1,0));
    //NSLog(@"%f.2, %f.2, %f.2, %f.2", par_ang1,  par_ang2,  right_ang1,  right_ang2 );
    //if (abs(horiz_ang) > 20) return false;
    if (abs(par_ang1) > 10) return false;
    if (abs(par_ang2) > 10) return false;
    if (abs(right_ang1 - 90) > 10) return false;
    if (abs(right_ang2 - 90) > 10) return false;
    return true;
}

//---------------------------------------------------------------------------------------
void morph_closing( cv::Mat &m, cv::Size sz, int iterations, int type = cv::MORPH_RECT )
{
    cv::Mat element = cv::getStructuringElement( type, sz);
                                                //cv::Size( 2*size + 1, 2*size+1 ),
                                                //cv::Point( size, size ) );
    for (int i=0; i<iterations; i++) {
        cv::dilate( m, m, element );
        cv::erode( m, m, element );
    }
}

// Find a nonzero point near the middle, flood from there,
// eliminate all else.
//--------------------------------------------------------
void flood_from_center( cv::Mat &m)
{
    // Find some nonzero point close to the center
    cv::Mat locations;
    cv::findNonZero(m, locations);
    std::vector<float> distvec(locations.rows);
    std::vector<int> idxvec(locations.rows);
    cv::Point center( m.cols / 2, m.rows / 2);
    // Sort points by dist from center
    for (int i=0; i<locations.rows; i++) {
        cv::Point p = locations.at<cv::Point>(i,0);
        distvec[i] = line_len(p, center);
        idxvec[i] = i;
    }
    if (!distvec.size()) return;
    std::sort( idxvec.begin(), idxvec.end(), [distvec](int a, int b) {
        return distvec[a] < distvec[b];
    });
    // Floodfill from nonzero point closest to center
    cv::Point seed = locations.at<cv::Point>(idxvec[0],0);
    cv::floodFill(m, seed, cv::Scalar(200));
    
    // Keep only filled area
    cv::threshold(m, m, 199, 255, cv::THRESH_BINARY);
}


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

//--------------------------------------------------
Points2f scale_board( Points board, float factor)
{
    board = order_points( board);
    Points diag1_stretched = stretch_line( { board[0],board[2] }, factor);
    Points diag2_stretched = stretch_line( { board[1],board[3] }, factor);
    Points2f res = { diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1] };
    return res;
}

// Make our 4-polygon a little larger
//-------------------------------------
Points2f enlarge_board( Points board)
{
    return scale_board( board, STRETCH_FACTOR);
}

// Zoom into an image area where pts are the four corners.
// From pyimagesearch by Adrian Rosebrock
//--------------------------------------------------------
cv::Mat four_point_transform( const cv::Mat &img, cv::Mat &warped, Points2f pts)
{
    Points2f rect = order_points(pts);
    cv::Point tl = pts[0];
    cv::Point tr = pts[1];
    cv::Point br = pts[2];
    cv::Point bl = pts[3];
    // compute the width of the new image, which will be the
    // maximum distance between bottom-right and bottom-left
    // x-coordiates or the top-right and top-left x-coordinates
    float widthA = sqrt(((br.x - bl.x)*(br.x - bl.x)) + ((br.y - bl.y)*(br.y - bl.y)));
    float widthB = sqrt(((tr.x - tl.x)*(tr.x - tl.x)) + ((tr.y - tl.y)*(tr.y - tl.y)));
    int maxWidth = fmax(int(widthA), int(widthB));
    // compute the height of the new image, which will be the
    // maximum distance between the top-right and bottom-right
    // y-coordinates or the top-left and bottom-left y-coordinates
    float heightA = sqrt(((tr.x - br.x)*(tr.x - br.x)) + ((tr.y - br.y)*(tr.y - br.y)));
    float heightB = sqrt(((tl.x - bl.x)*(tl.x - bl.x)) + ((tl.y - bl.y)*(tl.y - bl.y)));
    int maxHeight = fmax(int(heightA), int(heightB));
    // now that we have the dimensions of the new image, construct
    // the set of destination points to obtain a "birds eye view",
    // (i.e. top-down view) of the image, again specifying points
    // in the top-left, top-right, bottom-right, and bottom-left
    // order
    
    Points2f dst = {
        cv::Point(0,0),
        cv::Point(maxWidth - 1, 0),
        cv::Point(maxWidth - 1, maxHeight - 1),
        cv::Point(0, maxHeight - 1) };

    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat res;
    cv::warpPerspective(img, warped, M, cv::Size(maxWidth,maxHeight));
    return M;
} // four_point_transform()

// Linear map of four corners to whole screen width
//--------------------------------------------------------
cv::Mat board_transform( const cv::Mat &img, cv::Mat &warped, Points2f pts)
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
} // board_transform()


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

//--------------------------------------------------------------------
int get_boardsize_by_fft( const cv::Mat &zoomed_img)
{
    cv::Mat flimg;
    zoomed_img.convertTo( flimg, CV_64FC1);
    int width = zoomed_img.cols;
    int height = zoomed_img.rows;
    cplx crow[width];
    double magsum[width];
    ILOOP (width) { magsum[i]=0; }
    // Sum the ffts of each row
    ILOOP (height) {
        double *row = flimg.ptr<double>(i);
        KLOOP (width) { crow[k] = cplx( row[k],0); }
        fft( crow, width);
        KLOOP (width) { magsum[k] += std::abs(crow[k]); }
    }
    double ssum = 0;
    ILOOP (50) { ssum += magsum[width/2-i-1]; }
    // Smooth
    double old = magsum[0];
    double alpha = 0.2;
    ILOOP (width) { magsum[i] = (1-alpha)*magsum[i] + alpha*old; old = magsum[i]; }
    
    // Find max
    old = magsum[7];
    std::vector<int> argmaxes;
    std::vector<float> maxes;
    for (int i = 7; i < 30; i++ ) {
        double cur = magsum[i];
        double nnext = magsum[i+1];
        if (cur > old && cur > nnext) {
            argmaxes.push_back(i);
            maxes.push_back(cur);
        }
        old = magsum[i];
    }
    if (!argmaxes.size()) { return 9;}
    ILOOP (argmaxes.size()) {
        if (argmaxes[i] < 16 && maxes[i] > 50000) {
            return 9;
        }
        if ((argmaxes[i] >= 18 && argmaxes[i] <= 20)
            && maxes[i] > 50000) {
            return 19;
        }
    }
    return 9;
    
} // get_boardsize_by_fft

// Get a line segment representation of a Hough line (rho, theta)
//----------------------------------------------------------------
void polarToSegment( const cv::Vec2f &hline, cv::Vec4f &result)
{
    float rho = hline[0], theta = hline[1];
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    result[0] = cvRound(x0 + 1000*(-b));
    result[1] = cvRound(y0 + 1000*(a));
    result[2] = cvRound(x0 - 1000*(-b));
    result[3] = cvRound(y0 - 1000*(a));
}

// Draw a Hough line (rho, theta)
//-------------------------------------------------------
void drawPolarLine( cv::Vec2f line, cv::Mat &dst,
                   cv::Scalar col = cv::Scalar(255,0,0))
{
    cv::Vec4f seg;
    polarToSegment( line, seg);
    cv::Point pt1( seg[0], seg[1]), pt2( seg[2], seg[3]);
    cv::line( dst, pt1, pt2, col, 1, CV_AA);
}

//--------------------------------------------------------------
void drawPolarLines( std::vector<cv::Vec2f> lines, cv::Mat &dst,
                    cv::Scalar col = cv::Scalar(255,0,0))
{
    ISLOOP (lines) { drawPolarLine( lines[i], dst, col); }
}

// Get the middle screen x val for a somewhat vertical Hough line.
// Used to find leftmost, rightmost lines.
//----------------------------------------------------------------------------------
float polarMiddleValV( const cv::Vec2f &hline, int rows)
{
    float rho = hline[0];
    float theta = hline[1];
    float y = rows / 2.0;
    float x = (rho - y * sin(theta)) / cos(theta);
    return x;
}

// Get the middle screen y val for a somewhat horizontal Hough line.
// Used to find top, bottom lines.
//----------------------------------------------------------------------------------
float polarMiddleValH( const cv::Vec2f &hline, int cols)
{
    float rho = hline[0];
    float theta = hline[1];
    float x = cols / 2.0;
    float y = (rho - x * cos(theta)) / sin(theta);
    return y;
}

//-----------------------------------------------------------------------------------------
void drawLine( const cv::Vec4f &line, cv::Mat &dst, cv::Scalar col = cv::Scalar(255,0,0))
{
    cv::Point pt1, pt2;
    pt1.x = cvRound(line[0]);
    pt1.y = cvRound(line[1]);
    pt2.x = cvRound(line[2]);
    pt2.y = cvRound(line[3]);
    cv::line( dst, pt1, pt2, col, 1, CV_AA);
}

//--------------------------------------------------------------
void drawLines( const std::vector<cv::Vec4f> &lines, cv::Mat &dst,
               cv::Scalar col = cv::Scalar(255,0,0))
{
    ISLOOP (lines) drawLine( lines[i], dst, col);
}

// Return whole screen as board
//-----------------------------------------
Points whole_screen( const cv::Mat &img)
{
    Points res = { cv::Point(1,1), cv::Point(img.cols-2,1),
        cv::Point(img.cols-2,img.rows-2), cv::Point(1,img.rows-2) };
    return res;
}

// Find board in binary image (after threshold or canny)
//-------------------------------------------------------------
Points find_board( const cv::Mat &binImg, cv::Mat &boardImg)
{
    Contours conts;
    cv::findContours( binImg, conts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // only keep the biggest one
    std::sort( conts.begin(), conts.end(), [](Contour a, Contour b){ return cv::contourArea(a) > cv::contourArea(b); });
    //conts.erase( conts.begin()+1, conts.end());
    if (!conts.size()) return whole_screen( binImg);
    boardImg = cv::Mat::zeros( binImg.size(), CV_8UC1 );
    cv::drawContours( boardImg, conts, 0, cv::Scalar(255), 3);
    // Find lines
    std::vector<cv::Vec4f> lines;
    HoughLinesP(boardImg, lines, 1, CV_PI/180, 150, 0, 0 );
    // Find vertical and horizontal lines
    std::vector<std::vector<cv::Vec4f> > horiz_vert_lines;
    horiz_vert_lines = partition( lines, 2,
                                 [](cv::Vec4f &line) {
                                     if (fabs( line[0] - line[2]) > fabs( line[1] - line[3])) return 0;
                                     else return 1;
                                 });
    std::vector<cv::Vec4f> horiz_lines = horiz_vert_lines[0];
    std::vector<cv::Vec4f> vert_lines = horiz_vert_lines[1];
    // Separate top from bottom
    std::vector<std::vector<cv::Vec4f> > top_bottom_lines;
    top_bottom_lines = cluster( horiz_lines, 2,
                               [](cv::Vec4f &line) {
                                   return (line[1] + line[3]) / 2.0;
                               });
    if (!top_bottom_lines.size()) return whole_screen( binImg);
    // Separate left from right
    std::vector<std::vector<cv::Vec4f> > left_right_lines;
    left_right_lines = cluster( vert_lines, 2,
                               [](cv::Vec4f &line) {
                                   return (line[0] + line[2]) / 2.0;
                               });
    if (!left_right_lines.size()) return whole_screen( binImg);

    // Average vertical lines
    cv::Vec4f vert_1 = avg_lines( left_right_lines[0]);
    cv::Vec4f vert_2 = avg_lines( left_right_lines[1]);
    // Average horiz lines
    cv::Vec4f horiz_1 = avg_lines( top_bottom_lines[0]);
    cv::Vec4f horiz_2 = avg_lines( top_bottom_lines[1]);
    
    // Corners are the intersections
    cv::Point2f c1 = intersection( vert_1, horiz_1);
    cv::Point2f c2 = intersection( vert_1, horiz_2);
    cv::Point2f c3 = intersection( vert_2, horiz_1);
    cv::Point2f c4 = intersection( vert_2, horiz_2);
    Points corners = { cv::Point(c1), cv::Point(c2), cv::Point(c3), cv::Point(c4) };
    Points board = order_points( corners);
    if (board.size() != 4) return whole_screen( binImg);
    return board;
} // find_board()

// Make a better board estimate from several
//--------------------------------------------
Points best_board( std::vector<Points> boards)
{
    Points res(4);
    int minidx=0;
    float minArea = 1E9;
    ILOOP (boards.size()) {
        Points b = boards[i];
        float area = cv::contourArea(b);
        if (area < minArea) { minArea = area; minidx = i;}
//        res[0] += b[0];
//        res[1] += b[1];
//        res[2] += b[2];
//        res[3] += b[3];
    }
//    res[0] /= (float)boards.size();
//    res[1] /= (float)boards.size();
//    res[2] /= (float)boards.size();
//    res[3] /= (float)boards.size();
    return boards[minidx];
}

#pragma mark - Processing Pipeline for debugging
//=================================================

- (UIImage *) f00_adaptive_thresh:(UIImage *)img
{
    UIImageToMat( img, _m);
    resize( _m, _m, 350);
    cv::cvtColor( _m, _gray, cv::COLOR_BGR2GRAY);
    adaptiveThreshold( _gray, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                      3, // neighborhood_size
                      4); // constant to add. 2 to 6 is the viable range
    UIImage *res = MatToUIImage( _m);
    return(res);
}

//--------------------------
- (UIImage *) f01_closing
{
    //int erosion_size = 2;
    int iterations = 1;
    morph_closing( _m, cv::Size(3,1), iterations);
    morph_closing( _m, cv::Size(1,3), iterations);

    UIImage *res = MatToUIImage( _m);
    return res;
}

//----------------------
- (UIImage *) f02_flood
{
    flood_from_center( _m);
    UIImage *res = MatToUIImage( _m);
    return res;
}

//-----------------------------
- (UIImage *) f03_find_board
{
    cv::Mat drawing, boardImg;
    _board = find_board( _m, boardImg);
    if (!_board.size()) { return MatToUIImage( _m); }
    cv::cvtColor( boardImg, drawing, cv::COLOR_GRAY2RGB);
    cv::drawContours( drawing, std::vector<Points> (1, _board), -1, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//----------------------------
- (UIImage *) f04_zoom_in
{
    if (!_board.size()) { return MatToUIImage( _m); }
    // Zoom out a little
    Points2f board_stretched = enlarge_board( _board);
    cv::Mat transform = board_transform( _gray, _gray, board_stretched);
    Points2f b,tt;
    PointsToFloat( _board, b);
    cv::perspectiveTransform( b, tt, transform);
    PointsToInt( tt, _board_zoomed);
    
    UIImage *res = MatToUIImage( _gray);
    return res;
}

// Save small crops around intersections for use as template
//-------------------------------------------------------------------------------
void save_intersections( const cv::Mat img,
                        const Points2f &intersections, int delta_v, int delta_h)
{
    ILOOP( intersections.size())
    {
        float x = intersections[i].x;
        float y = intersections[i].y;
        cv::Rect rect( x - delta_h/2.0, y - delta_v/2.0, delta_h, delta_v );
        if (0 <= rect.x &&
            0 <= rect.width &&
            rect.x + rect.width <= img.cols &&
            0 <= rect.y &&
            0 <= rect.height &&
            rect.y + rect.height <= img.rows)
        {
            cv::Mat hood = cv::Mat( img, rect);
            NSString *fname = nsprintf(@"hood_%03d.jpg",i);
            fname = getFullPath( fname);
            cv::imwrite( [fname UTF8String], hood);
        }
    } // ILOOP
} // save_intersections()

//-------------------------------------------------------------
void find_stones( const cv::Mat &img, Points &result)
{
    cv::Mat mtmp;
    // Find circles
    std::vector<cv::Vec3f> circles;
    cv::GaussianBlur( img, mtmp, cv::Size(5, 5), 2, 2 );
    cv::HoughCircles( mtmp, circles, CV_HOUGH_GRADIENT,
                     1, // acumulator res == image res; Larger means less acc res
                     img.rows/30, // minimum distance between circles
                     260, // upper canny thresh; half of this is the lower canny
                     12, // less means more circles. The higher ones come first in the result
                     0,   // min radius
                     25 ); // max radius
    if (!circles.size()) return;

    // Keep the ones where radius close to avg radius
    std::vector<float> rads;
    ISLOOP (circles){ rads.push_back( circles[i][2]); }
    float avg_r = vec_median( rads);
    
    std::vector<cv::Vec3f> good_circles;
    //const float TOL_LO = 2.0;
    const float TOL_HI = 0.5;
    ISLOOP (circles)
    {
        cv::Vec3f c = circles[i];
        if ( c[2] > avg_r && (c[2] - avg_r) / avg_r < TOL_HI) {
            good_circles.push_back( circles[i]);
        }
        else if ( c[2] <= avg_r) {
            good_circles.push_back( circles[i]);
        }
    }
    ISLOOP (good_circles) { result.push_back( cv::Point( circles[i][0], circles[i][1]) ); }

    // Draw circles for debug
//    cv::cvtColor( mtmp, mtmp, cv::COLOR_GRAY2RGB);
//    ISLOOP (good_circles)
//    {
//        cv::Vec3f c = good_circles[i];
//        cv::Point center( cvRound(c[0]), cvRound(c[1]));
//        int radius = cvRound( c[2]);
//        // draw the circle center
//        cv::circle( mtmp, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
//        // draw the circle outline
//        cv::circle( mtmp, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
//    }
} // find_black_stones()

//-------------------------------------------------------------
void find_empty_places( const cv::Mat &img, Points &border) //, Points &inner)
{
    // Prepare image for template matching
    cv::Mat mtmp;
    cv::adaptiveThreshold( img, mtmp, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          11,  // neighborhood_size
                          8); // 8 or ten, need to try both. 8 better for 19x19
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate( mtmp, mtmp, element );
    
    // Define the templates
    const int tsz = 15;
    uint8_t cross[tsz*tsz] = {
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
    };
    uint8_t right[tsz*tsz] = {
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0
    };
    uint8_t left[tsz*tsz] = {
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
    };
    uint8_t top[tsz*tsz] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0
    };
    uint8_t bottom[tsz*tsz] = {
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    };
    cv::Mat mcross  = 255 * cv::Mat(tsz, tsz, CV_8UC1, cross);
    cv::Mat mright  = 255 * cv::Mat(tsz, tsz, CV_8UC1, right);
    cv::Mat mleft   = 255 * cv::Mat(tsz, tsz, CV_8UC1, left);
    cv::Mat mtop    = 255 * cv::Mat(tsz, tsz, CV_8UC1, top);
    cv::Mat mbottom = 255 * cv::Mat(tsz, tsz, CV_8UC1, bottom);
    
    // Match
    double thresh = 90;
    //inner.clear();
    //border.clear();
    //matchTemplate( mtmp, mcross, inner, thresh);
    matchTemplate( mtmp, mright, border, thresh);
    matchTemplate( mtmp, mleft, border, thresh);
    matchTemplate( mtmp, mtop, border, thresh);
    matchTemplate( mtmp, mbottom, border, thresh);
    matchTemplate( mtmp, mcross, border, thresh);
    //NSLog (@"template thresh at %.2f inner %ld border %ld", thresh, innersize, bordersize );
} // find_empty_places()

// Template maching for empty intersections
//------------------------------------------------------------------------------
void matchTemplate( const cv::Mat &img, const cv::Mat &templ, Points &result, double thresh)
{
    cv::Mat matchRes;
    cv::Mat mtmp;
    int tsz = templ.rows;
    cv::copyMakeBorder( img, mtmp, tsz/2, tsz/2, tsz/2, tsz/2, cv::BORDER_REPLICATE, cv::Scalar(0));
    cv::matchTemplate( mtmp, templ, matchRes, CV_TM_SQDIFF);
    cv::normalize( matchRes, matchRes, 0 , 255, CV_MINMAX, CV_8UC1);
    cv::adaptiveThreshold( matchRes, mtmp, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          //11,  // neighborhood_size
                          11,  // neighborhood_size
                          thresh); // threshold; less is more
    // Find the blobs. They are the empty places.
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = 255;
    params.minDistBetweenBlobs = 2;
    params.filterByConvexity = false;
    params.filterByInertia = false;
    params.filterByCircularity = false;
    params.minCircularity = 0.0;
    params.maxCircularity = 100;
    params.minArea = 0;
    params.maxArea = 100;
    cv::Ptr<cv::SimpleBlobDetector> d = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    d->detect( mtmp, keypoints);
    //result = Points();
    ILOOP (keypoints.size()) { result.push_back(keypoints[i].pt); }
    
} // matchTemplate()

// Find rough guess at stones and empty intersections
// Also determines board size (9,13,19)
//------------------------------------------------------
- (UIImage *) f05_find_intersections
{
    Points pts, crosses;
    find_empty_places( _gray, pts); // , crosses); // has to be first
    find_stones( _gray, pts);
    // Use only inner ones
    Points2f innerboard = scale_board( _board_zoomed, 1.01);
    _stone_or_empty = Points();
    ISLOOP (pts) {
        cv::Point2f p( pts[i]);
        if (cv::pointPolygonTest( innerboard, p, false) > 0) {
            _stone_or_empty.push_back( p);
        }
    }
    _board_sz = 19;
//    Points inner_empty;
//    ISLOOP (crosses) {
//        cv::Point2f p( crosses[i]);
//        if (cv::pointPolygonTest( innerboard, p, false) > 0) {
//            inner_empty.push_back( p);
//        }
//    }
//    size_t n = _stone_or_empty.size() + inner_empty.size();
//    _board_sz = 19;
//    if (n <= 9*9) { _board_sz = 9; }
//    else if (n <= 250) { _board_sz = 13; }
    // Show results
    cv::Mat canvas;
    cv::cvtColor( _gray, canvas, cv::COLOR_GRAY2RGB);
    ISLOOP ( _stone_or_empty) {
        draw_point( _stone_or_empty[i], canvas, 2);
    }
    UIImage *res = MatToUIImage( canvas);
    return res;
}

// Find all intersections from corners and boardsize
//--------------------------------------------------------------------------------
template <typename Points_>
void get_intersections( const Points_ &corners, int boardsz,
                       Points_ &result, float &delta_v, float &delta_h)
{
    if (corners.size() != 4) return;
    
    cv::Point2f tl = corners[0];
    cv::Point2f tr = corners[1];
    cv::Point2f br = corners[2];
    cv::Point2f bl = corners[3];
    
    std::vector<float> left_x;
    std::vector<float> left_y;
    std::vector<float> right_x;
    std::vector<float> right_y;
    ILOOP (boardsz) {
        left_x.push_back(  tl.x + i * (bl.x - tl.x) / (float)(boardsz-1));
        left_y.push_back(  tl.y + i * (bl.y - tl.y) / (float)(boardsz-1));
        right_x.push_back( tr.x + i * (br.x - tr.x) / (float)(boardsz-1));
        right_y.push_back( tr.y + i * (br.y - tr.y) / (float)(boardsz-1));
    }
    std::vector<float> top_x;
    std::vector<float> top_y;
    std::vector<float> bot_x;
    std::vector<float> bot_y;
    ILOOP (boardsz) {
        top_x.push_back( tl.x + i * (tr.x - tl.x) / (float)(boardsz-1));
        top_y.push_back( tl.y + i * (tr.y - tl.y) / (float)(boardsz-1));
        bot_x.push_back( bl.x + i * (br.x - bl.x) / (float)(boardsz-1));
        bot_y.push_back( bl.y + i * (br.y - bl.y) / (float)(boardsz-1));
    }
    delta_v = abs(int(round( 0.5 * (bot_y[0] - top_y[0]) / (boardsz -1))));
    delta_h = abs(int(round( 0.5 * (right_x[0] - left_x[0]) / (boardsz -1))));
    
    result = Points_();
    RLOOP (boardsz) {
        CLOOP (boardsz) {
            cv::Point2f p = intersection( cv::Point2f( left_x[r], left_y[r]), cv::Point2f( right_x[r], right_y[r]),
                                         cv::Point2f( top_x[c], top_y[c]), cv::Point2f( bot_x[c], bot_y[c]));
            result.push_back(p);
        }
    }
} // get_intersections()


// Get MSE of the dots relative to the grid defined by corners and boardsize
//-----------------------------------------------------------------------------
template <typename Points_>
float grid_err( const Points_ &corners, const Points_ &dots, int boardsize)
{
    Points_ gridpoints;
    float delta_v, delta_h;
    get_intersections( corners, boardsize, gridpoints, delta_v, delta_h);
    double err = 0;
    ILOOP (dots.size()) {
        float mind = 10E9;
        JLOOP (gridpoints.size()) {
            float d = cv::norm( dots[i] - gridpoints[j]);
            if (d < mind) {
                mind = d;
            }
        }
        //NSLog(@"mind:%f", mind);
        err += mind * mind;
    }
    err = sqrt(err);
    return err;
}

// Eliminate points that don't fit the rhythm of lines
//--------------------------------------------------------------------
void outlier_points( const std::vector<Points> &clusters,
                    float &phase,
                    float &wavelength,
                    float &slope,
                    Points &clean_points, Points &dirty_points, float epsilon = 0.25)
{
    std::vector<cv::Vec4f> lines;
    // Lines through the clusters
    ISLOOP (clusters) {
        lines.push_back( fit_line( clusters[i]));
    }
    // Slopes of the lines
    std::vector<float> slopes;
    ISLOOP( lines) {
        cv::Vec4f line = lines[i];
        float dx = line[2] - line[0];
        float dy = line[3] - line[1];
        if (fabs(dx) > fabs(dy)) { // horizontal
            if (dx < 0) { dx *= -1; dy *= -1; }
        }
        else { // vertical
            if (dy > 0) { dx *= -1; dy *= -1; }
        }
        float theta = atan2( dy, dx);
        //NSLog(@"dx dy theta %.2f %.2f %.2f", dx, dy, theta );
        slopes.push_back( theta);
    }
    NSLog(@"==========");
    slope = vec_median( slopes);
    // A polar line with the median slope
    cv::Vec2f median_hline(0, slope + CV_PI/2.0);

    // For each cluster, get the median dist from the median slope line
    std::vector<float> distances( clusters.size());
    ISLOOP (clusters) {
        std::vector<float> ds;
        JSLOOP (clusters[i]) {
            cv::Point p = clusters[i][j];
            float d = dist_point_line( p, median_hline);
            ds.push_back( d);
        }
        distances[i] = vec_median( ds);
    }
    // Get the rhythm (wavelength of line distances)
    std::sort( distances.begin(), distances.end(), [](float a, float b){ return a < b; });
    std::vector<float> delta_dists;
    ISLOOP (distances) {
        if (!i) continue;
        delta_dists.push_back( distances[i] - distances[i-1]);
    }
    wavelength = vec_median( delta_dists);
    // Get the phase
    std::vector<float> phases;
    ISLOOP (distances) {
        phases.push_back( fmod( distances[i],wavelength));
    }
    phase = vec_median( phases);
    // Filter
    ISLOOP (clusters) {
        JSLOOP (clusters[i]) {
            cv::Point p = clusters[i][j];
            float d = dist_point_line( p, median_hline);
            float rem = fmod( d, wavelength) ;
            float delta = fabs( rem - phase);
            if ( delta < epsilon * wavelength) {
                clean_points.push_back( p);
            }
            else {
                dirty_points.push_back(p);
            }
        }
    }
} // outlier_points()

// Find grid by putting lines through detected stones and intersections
//------------------------------------------------------------------------
- (UIImage *) f06_hough_grid
{
    // Find Hough lines in the detected intersections and stones
    cv::Mat canvas = cv::Mat::zeros( _gray.size(), CV_8UC1 );
    ILOOP (_stone_or_empty.size()) {
        draw_point( _stone_or_empty[i], canvas,1, cv::Scalar(255));
    }
    std::vector<cv::Vec2f> lines;
    std::vector<std::vector<cv::Vec2f> > horiz_vert_other_lines;
    std::vector<int> vote_thresholds = { 10, 9, 8, 7, 6, 5 };

    ISLOOP (vote_thresholds) {
        int votes = vote_thresholds[i];
        NSLog( @"trying %d hough line votes", votes );
        HoughLines(canvas, lines, 1, CV_PI/180, votes, 0, 0 );

        // Separate horizontal, vertical, and other lines
        horiz_vert_other_lines = partition( lines, 3,
                                           [](cv::Vec2f &line) {
                                               const float thresh = 10.0;
                                               float theta = line[1] * (180.0 / CV_PI);
                                               if (fabs(theta - 180) < thresh) return 1;
                                               else if (fabs(theta) < thresh) return 1;
                                               else if (fabs(theta-90) < thresh) return 0;
                                               else return 2;
                                           });
        // Sort by Rho (distance of line from origin)
        std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
        std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
        if (hlines.size() >= 2 && vlines.size() >= 2) break;
    }
    std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
    std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
    cv::Vec4f hslope = avg_slope_line( hlines);
    cv::Vec4f vslope = avg_slope_line( vlines);

    // Cluster points into board_size clusters by dist from hslope, vslope
    _horizontal_clusters = cluster( _stone_or_empty, _board_sz,
                                  [hslope](cv::Point &p) { return dist_point_line(p, hslope); });
    _vertical_clusters = cluster( _stone_or_empty, _board_sz,
                                  [vslope](cv::Point &p) { return dist_point_line(p, vslope); });
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    drawLine(topline, drawing);
//    drawPolarLine( topline, drawing);
//    ISLOOP (_intersections) { draw_point( _intersections[i], drawing, 2, cv::Scalar(0,255,0)); }
    ISLOOP( _horizontal_clusters) {
        Points cl = _horizontal_clusters[i];
        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
        draw_points( cl, drawing, 2, color);
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f06_hough_grid()

// Find wavelength and phase of the grid, remove outliers.
//------------------------------------------------------------------------
- (UIImage *) f07_clean_grid_h
{
    Points clean_points, dirty_points;
    float phase, slope;
    outlier_points( _horizontal_clusters, phase, _wavelen_h,
                   slope,
                   clean_points, dirty_points,
                 0.25); // epsilon should be between 0 and 0.5
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    // Draw the family of lines from phase, wavelength, slope
    float theta = slope + CV_PI/2;
    float rho = phase;
    _horizontal_lines = std::vector<cv::Vec4f>();
    while (rho < _gray.rows) {
        cv::Vec2f hline( rho, theta);
        cv::Vec4f line;
        polarToSegment( hline, line);
        _horizontal_lines.push_back( line);
        drawLine( line, drawing, cv::Scalar(0,0,255));
        rho += _wavelen_h;
    }
    draw_points( clean_points, drawing, 2, cv::Scalar(0,255,0));
    draw_points( dirty_points, drawing, 2, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}
//------------------------------------------------------------------------
- (UIImage *) f08_clean_grid_v
{
    Points clean_points, dirty_points;
    float phase, slope;
    outlier_points( _vertical_clusters, phase, _wavelen_v,
                   slope,
                   clean_points, dirty_points,
                   0.25); // epsilon should be between 0 and 0.5
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    // Draw the family of lines from phase, wavelength, slope
    float theta = slope + CV_PI/2;
    float rho = phase;
    _vertical_lines = std::vector<cv::Vec4f>();
    while (rho < _gray.cols) {
        cv::Vec2f hline( rho, theta);
        cv::Vec4f line;
        polarToSegment( hline, line);
        _vertical_lines.push_back( line);
        drawLine( line, drawing, cv::Scalar(0,0,255));
        rho += _wavelen_v;
    }
    draw_points( clean_points, drawing, 2, cv::Scalar(0,255,0));
    draw_points( dirty_points, drawing, 2, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}


//------------------------------------
- (UIImage *) fxx_get_intersections
{
    if (!_board.size()) { return MatToUIImage( _m); }
    cv::Mat drawing; // = cv::Mat::zeros( _gray.size(), CV_8UC3 );
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    
    Points intersections;
    float delta_v, delta_h;
    get_intersections( _board_zoomed, _board_sz, intersections, delta_v, delta_h);
    ILOOP (intersections.size()) {
        draw_point( intersections[i], drawing, 1);
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f06_get_intersections()

// Get a center crop of an image
//-------------------------------------------------------------------
int get_center_crop( const cv::Mat &img, cv::Mat &dst, float frac=4)
{
    float cx = img.cols / 2.0;
    float cy = img.rows / 2.0;
    float dx = img.cols / frac;
    float dy = img.rows / frac;
    dst = cv::Mat( img, cv::Rect( round(cx-dx), round(cy-dy), round(2*dx), round(2*dy)));
    int area = dst.rows * dst.cols;
    return area;
}

// Sum brightness at the center, normalize
//------------------------------------------------------
float get_brightness( const cv::Mat &img, float frac=4)
{
    cv::Mat crop;
    int area = get_center_crop( img, crop, frac);
    float ssum = cv::sum(crop)[0];
    return ssum / area;
}

//------------------------------------------------
void logveci( NSString *str, std::vector<int> v)
{
    NSString *msg = nsprintf( @"%@", str);
    ISLOOP (v) {
        if (i%19 == 0) msg = nscat( msg, @"\n");
        msg = nscat( msg, nsprintf( @"%5d ", v[i]));
    }
    NSLog(@"%@\n", msg);
}

//------------------------------------------------
void logvecf( NSString *str, std::vector<float> v)
{
    NSString *msg = nsprintf( @"%@", str);
    ISLOOP (v) {
        if (i%19 == 0) msg = nscat( msg, @"\n");
        msg = nscat( msg, nsprintf( @"%7.2f ", v[i]));
    }
    NSLog(@"%@\n", msg);
}

// Type to hold a feature vector at a board position
//=====================================================
typedef struct feat {
    std::string key;
    int x,y;     // Pixel pos
    std::vector<float> features;
} Feat;

// Compute features from a neighborhood of point
//---------------------------------------------------------------------------------------------
void get_features( const cv::Mat &img, cv::Point p, float wavelen_h, float wavelen_v,
                  std::string key, Feat &f)
{
    cv::Rect rect( p.x - wavelen_h/4.0, p.y - wavelen_v/4.0, wavelen_h/2.0, wavelen_v/2.0 );
    if (0 <= rect.x &&
        0 <= rect.width &&
        rect.x + rect.width <= img.cols &&
        0 <= rect.y &&
        0 <= rect.height &&
        rect.y + rect.height <= img.rows)
    {
        cv::Mat hood = cv::Mat( img, rect);
        float area = hood.rows * hood.cols;
        cv::Scalar ssum = cv::sum( hood);
        float brightness_r = ssum[0] / area;
        float brightness_g = ssum[1] / area;
        float brightness_b = ssum[2] / area;
        float v = sqrt (brightness_r*brightness_r + brightness_g*brightness_g + brightness_b*brightness_b);
        f.features.push_back( brightness_r);
        f.features.push_back( brightness_g);
        f.features.push_back( brightness_b);
    }
    else {
        NSLog( @"get_features failed at key %s", key.c_str());
    }
    f.key = key;
    f.x = p.x;
    f.y = p.y;
    
} // get_features()

//----------------------------------
std::string rc_key (int r, int c)
{
    char buf[100];
    sprintf( buf, "%d_%d", r,c);
    return std::string (buf);
}

// Get a list of features for the intersections in a subgrid
// with r,c as upper left corner.
//------------------------------------------------------------
bool get_subgrid_features( int top_row, int left_col, int boardsize,
                 std::map<std::string, Feat> &features,
                 std::vector<Feat> &subgrid)
{
    //NSLog( @"get_subgrid_features for %d %d", top_row, left_col);
    RLOOP (boardsize) {
        CLOOP (boardsize) {
            std::string key = rc_key( top_row + r, left_col + c);
            if (!features.count( key)) {
                //NSLog( @"no intersection at %d %d", top_row + r, left_col + c);
                return false;
            }
            if (!features[key].features.size()) {
                //NSLog( @"no features at %d %d", top_row + r, left_col + c);
                return false;
            }
            subgrid.push_back( features[key]);
        }
    }
    return true;
}

// Normalize mean and variance, per channel
//========================================================
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


// Classify intersections into b,w,empty
//----------------------------------------
- (UIImage *) f09_classify
{
    //test_mcluster();
    // Get pixel pos for each potential board intersection
    std::map<std::string, cv::Point> intersections;
    RSLOOP (_horizontal_lines) {
        CSLOOP (_vertical_lines) {
            intersections[rc_key(r,c)] = intersection( _horizontal_lines[r], _vertical_lines[c]);
        }
    }
    
    // Normalize color image
    cv::Mat img;
    normalize_image( _small, img);

    // Compute features for each potential board intersection
    std::map<std::string, Feat> features;
    for (const auto &x : intersections) {
        std::string key = x.first;
        cv::Point p = x.second;
        Feat f;
        //get_features( _gray, p, _wavelen_h, _wavelen_v, key, f);
        
        
//
//        _small.convertTo( _small, CV_32FC3);
//        cv::Mat img;
//        _small.convertTo( img, CV_32FC4, sstddev, -mmean/sstddev);
//        //_small -= mmean;
//        //_small = _small / sstddev;

        get_features( img, p, _wavelen_h, _wavelen_v, key, f);
        features[key] = f;
    }
    // Try all possible grids @@@
    RSLOOP (_horizontal_lines) {
        CSLOOP (_vertical_lines) {
            //NSLog(@">>>>>>>>>>> %d %d", r, c);
            std::vector<Feat> subgrid;
            if (!get_subgrid_features( r, c, _board_sz, features, subgrid)) continue;
            //NSLog( @"$$$$$$$$$$$$$ Complete grid at %d %d", r, c);
            // Cluster the features into two classes
            std::vector<std::vector<Feat> > clusters;
            clusters = mcluster( subgrid,
                                2, // clusters
                                3, // feature dims
                                [](Feat f) { return f.features; });
            NSLog( @"%ld %ld clusters at %d %d",
                  clusters[0].size(), clusters[1].size(),
                  r, c);
        }
    }
    NSLog( @"==================");


//
    cv::Mat drawing; // = cv::Mat::zeros( _gray.size(), CV_8UC3 );
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    // Contour image of the zoomed board
//    cv::Mat zoomed_edges;
//    //cv::Canny( _gray, zoomed_edges, _canny_low, _canny_hi);
//    cv::Canny( _gray, zoomed_edges, 30, 70);
//    //auto_canny( _gray, zoomed_edges);
//    //cv::findContours( zoomed_edges, _cont, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
//    // Cut out areas around the intersections
//    std::vector<float> brightness;
//    std::vector<float> crossness;
//    std::vector<int> isempty;
//    ILOOP( _intersections.size()) {
//        float x = _intersections[i].x;
//        float y = _intersections[i].y;
//        cv::Rect rect( x -_delta_h/2.0, y - _delta_v/2.0, _delta_h, _delta_v );
//        if (0 <= rect.x &&
//            0 <= rect.width &&
//            rect.x + rect.width <= _gray.cols &&
//            0 <= rect.y &&
//            0 <= rect.height &&
//            rect.y + rect.height <= _gray.rows)
//        {
//            cv::Mat hood = cv::Mat( _gray, rect);
//            cv::Mat contour_hood = cv::Mat( zoomed_edges, rect);
//            brightness.push_back( get_brightness( hood));
//            crossness.push_back( get_brightness(contour_hood,6));
//            cv::rectangle( drawing, rect, cv::Scalar(255,0,0));
//
////            // Template approach
////            templify(hood);
////            double sim_white = cmpTmpl(hood,_tmpl_white);
////            double sim_inner = cmpTmpl(hood,_tmpl_inner);
////            if (sim_inner > sim_white) { isempty.push_back(2); }
////            else { isempty.push_back(0); }
//        }
//    }
//    //logvecf( @"brightness:", brightness);
//    //logvecf( @"crossness:",  crossness);
//
//    // Black stones
//    float thresh = *(std::min_element( brightness.begin(), brightness.end())) * 4;
//    std::vector<int> isblack( brightness.size(), 0);
//    ILOOP (brightness.size()) {
//        if (brightness[i] < thresh) {
//            isblack[i] = 1;
//            draw_point( _intersections[i], drawing, 1);
//        }
//    }
//    logveci( @"isblack:", isblack);
//    ILOOP (isempty.size()) {
//        if (isblack[i]) isempty[i] = 0;
//    }
//    logveci( @"isempty:", isempty);
//    std::vector<int> board;
//    ILOOP (isblack.size()) {
//        if (isblack[i]) board.push_back(1);
//        else if (isempty[i]) board.push_back(0);
//        else board.push_back(2);
//    }
//    logveci(<#NSString *str#>, <#std::vector<int> v#>)( @"board:", board);

//    // Empty places
//    std::vector<int> isempty( crossness.size(), 0);
//    ILOOP (crossness.size()) {
//        if (crossness[i] > 5) {
//            isempty[i] = 1;
//            draw_point( _intersections[i], drawing, 1);
//        }
//    }

//# Empty intersections
//    print('crossness')
//    print(crossness.reshape((boardsize,boardsize)).astype('int'))
//    isempty = np.array([ 1 if x > 5 else 0 for x in crossness ])
//
//# White stones
//    iswhite = np.array([ 2 if not isempty[i] + isblack[i] else 0  for i,x in enumerate( isempty) ])
//
//    print('position')
//    position = iswhite + isblack
//    print(position.reshape((boardsize,boardsize)))
    
    
    UIImage *res = MatToUIImage( drawing);
    //UIImage *res = MatToUIImage( zoomed_edges);
    return res;

} // f07_classify()

#pragma mark - Real time implementation
//========================================

// f00_*, f01_*, ... all in one go
//--------------------------------------------
- (UIImage *) findBoard:(UIImage *) img
{
    const int N_BOARDS = 8;
    static std::vector<Points> boards; // Some history for averaging
    UIImageToMat( img, _m, false);
    resize( _m, _small, 350);
    //cv::cvtColor( small, small, cv::COLOR_BGR2RGB);
    cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(_gray, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          3, // neighborhood_size
                          4); // constant to add. 2 to 6 is the viable range
    // Make sure the board in the center is in one piece
    int iterations = 1;
    morph_closing( _m, cv::Size(3,1), iterations);
    morph_closing( _m, cv::Size(1,3), iterations);
    // Find the biggest connected piece in the center
    flood_from_center( _m);
    // Find Hough lines and construct the board from them
    cv::Mat boardImg;
    _board = find_board( _m, boardImg);
    if ( board_valid( _board, cv::contourArea(whole_screen(_small)))) {
        boards.push_back( _board);
        if (boards.size() > N_BOARDS) { boards.erase( boards.begin()); }
        _board = best_board( boards);
        drawContour( _small, _board, cv::Scalar(255,0,0,255));
//        _cont = std::vector<Points>( 1, _board);
//        cv::drawContours( small, _cont, -1, cv::Scalar(255,0,0,255));
    }
    //cv::cvtColor( small, small, cv::COLOR_RGB2BGR);
    UIImage *res = MatToUIImage( _small);
    return res;
}

@end





























