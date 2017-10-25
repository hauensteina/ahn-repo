//
//  GrabFuncs.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "Common.h"
#import "GrabFuncs.h"

typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
static cv::RNG rng(12345);


@interface GrabFuncs()
//=======================
@property cv::Mat m; // Mat with image we are working on
@property Contours cont; // Current set of contours
@property Points board;  // Current hypothesis on where the board is
@end

@implementation GrabFuncs
//=========================


#pragma mark - General utility funcs
//======================================

//----------------------
- (instancetype)init
{
    self = [super init];
    if (self) {
        _canny_hi = 120;
        _canny_low = 70;
    }
    return self;
}

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

//# Find x where f(x) = target where f is an increasing func.
//#------------------------------------------------------------
//def bisect( f, lower, upper, target, maxiter=10):
//n=0
//while True and n < maxiter:
//n += 1
//res = (upper + lower) / 2.0
//val = f(res)
//if val > target:
//upper = res
//elif val < target:
//lower = res
//else:
//break
//return res

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

//# Order four points clockwise
//#------------------------------
//def order_points(pts):
//top_bottom = sorted( pts, key=lambda x: x[1])
//top = top_bottom[:2]
//bottom = top_bottom[2:]
//res = sorted( top, key=lambda x: x[0]) + sorted( bottom, key=lambda x: -x[0])
//return np.array(res).astype(np.float32)

// Order four points clockwise
//----------------------------------------
Points order_points( Points &points)
{
    Points top_bottom = points;
    std::sort( top_bottom.begin(), top_bottom.end(), [](cv::Point a, cv::Point b){ return a.y < b.y; });
    Points top( top_bottom.begin(), top_bottom.begin()+2 );
    Points bottom( top_bottom.end()-2, top_bottom.end());
    std::sort( top.begin(), top.end(), [](cv::Point a, cv::Point b){ return a.x < b.x; });
    std::sort( bottom.begin(), bottom.end(), [](cv::Point a, cv::Point b){ return b.x < a.x; });
    Points res = top;
    res.insert(res.end(), bottom.begin(), bottom.end());
    return res;
    
}

// Length of a line segment
//---------------------------------------------------------
float line_len( cv::Point p, cv::Point q)
{
    return cv::norm( q-p);
}

// Return unit vector of p
//----------------------------------
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

//# Enclose a contour with an n edge polygon
//#-------------------------------------------
//def approx_poly( cnt, n):
//hull = cv2.convexHull( cnt)
//peri = cv2.arcLength( hull, closed=True)
//epsilon = bisect( lambda x: -len(cv2.approxPolyDP(hull, x * peri, closed=True)),
//                 0.0, 1.0, -n)
//res  = cv2.approxPolyDP(hull, epsilon*peri, closed=True)
//return res

// Enclose a contour with an n edge polygon
//--------------------------------------------
Points approx_poly( Points cont, int n)
{
    Points hull;
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

// Calculates the median value of a vector of int
//-------------------------------------------------
int int_median( std::vector<int> ints )
{
    std::sort( ints.begin(), ints.end(), [](int a, int b) { return a < b; });
    int res = ints[ints.size() / 2];
    return res;
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
void draw_point( cv::Point p, cv::Mat &img)
{
    cv::circle( img, p, 10, cv::Scalar(255,0,0), -1);
}


#pragma mark - Pipeline Helpers
//==================================

//---------------------------------------------
Contours get_contours( const cv::Mat &img, int low, int hi)
{
    Contours conts;
    std::vector<cv::Vec4i> hierarchy;
    // Edges
    cv::Mat m;
    //auto_canny( img, m, 0.7);
    cv::Canny( img, m, low, hi);
    // Find contours
    findContours( m, conts, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    return conts;
} // get_contours()

// Try to eliminate small and wiggly contours
//---------------------------------------------------------------------
Contours filter_contours( const Contours conts, int width, int height)
{
    Contours large_conts;
    float minArea = width * height / 4000.0;
    std::copy_if( conts.begin(), conts.end(), std::back_inserter(large_conts),
                 [minArea](Contour c){return cv::contourArea(c) > minArea;} );
    
    Contours large_hullarea;
    std::copy_if( large_conts.begin(), large_conts.end(), std::back_inserter(large_hullarea),
                 [minArea](Contour c){
                     Contour hull;
                     cv::convexHull( c, hull);
                     return cv::contourArea(hull) > 0.001; });

    return large_hullarea;
}

// Find the center of the board, which is the median of the contours on the board
//----------------------------------------------------------------------------------
cv::Point get_board_center( const Contours conts)
{
    Points centers;
    centers.resize( conts.size());
    int i = 0;
    std::generate( centers.begin(), centers.end(), [conts,&i] {
        Contour c = conts[i++];
        cv::Moments M = cv::moments( c);
        int cent_x = int(M.m10 / M.m00);
        int cent_y = int(M.m01 / M.m00);
        return cv::Point(cent_x, cent_y);
    });
    i=0;
    std::vector<int> cent_x( conts.size());
    std::generate( cent_x.begin(), cent_x.end(), [centers,&i] { return centers[i++].x; } );
    i=0;
    std::vector<int> cent_y( conts.size());
    std::generate( cent_y.begin(), cent_y.end(), [centers,&i] { return centers[i++].y; } );
    int x = int_median( cent_x);
    int y = int_median( cent_y);
    return cv::Point(x,y);
}

//# Remove spurious contours outside the board
//#--------------------------------------------------
//def cleanup_squares(centers, square_cnts, board_center, width, height):
//# Store distance from center for each contour
//# sqdists = [ {'cnt':sq, 'dist':np.linalg.norm( centers[idx] - board_center)}
//#             for idx,sq in enumerate(square_cnts) ]
//# distsorted = sorted( sqdists, key = lambda x: x['dist'])
//
//#ut.show_contours( g_frame, square_cnts)
//sqdists = [ {'cnt':sq, 'dist':ut.contour_maxdist( sq, board_center)}
//           for sq in square_cnts ]
//distsorted = sorted( sqdists, key = lambda x: x['dist'])
//
//# Remove contours if there is a jump in distance
//lastidx = len(distsorted)
//for idx,c in enumerate(distsorted):
//if not idx: continue
//delta = c['dist'] - distsorted[idx-1]['dist']
//#print(c['dist'], delta)
//#ut.show_contours( g_frame, [c['cnt']])
//#print ('dist:%f delta: %f' % (c['dist'],delta))
//if delta > min(width,height) / 10.0:
//lastidx = idx
//break
//
//res = [x['cnt'] for x in distsorted[:lastidx]]
//return res
//

// Remove spurious contours outside the board
//-------------------------------------------------------------------------------
Contours filter_outside_contours( const Contours &conts,
                                 cv::Point board_center,
                                 int width, int height)
{
    typedef struct dist_idx {
        int idx;
        float dist;
    } dist_idx_t;
    
    //size_t sz = conts.size();
    std::vector<dist_idx_t> sqdists( conts.size());
    int i=0;
    std::generate( sqdists.begin(), sqdists.end(), [conts,board_center,&i] {
        Contour c = conts[i++];
        float dist = 0; int idx = -1;
        for (cv::Point p: c) {
            float d = sqrt( (p.x - board_center.x)*(p.x - board_center.x) +
                           (p.y - board_center.y)*(p.y - board_center.y));
            if (d > dist) { dist = d; idx = i-1; }
        }
        dist_idx_t res;
        res.dist = dist; res.idx = idx;
        return res;
    });
    
//    dist_idx_t di = sqdists[0];
//    di = sqdists[1];
//    di = sqdists[2];
//    di = sqdists[3];
//    di = sqdists[4];

    std::sort( sqdists.begin(), sqdists.end(), [](dist_idx_t a, dist_idx_t b) { return a.dist < b.dist; });

//    di = sqdists[0];
//    di = sqdists[1];
//    di = sqdists[2];
//    di = sqdists[3];
//    di = sqdists[4];

    size_t lastidx = sqdists.size();
    i=0;
    float lim = fmin( width, height) / 10.0;
    for (dist_idx_t di: sqdists) {
        if (i) {
            float delta = di.dist - sqdists[i-1].dist;
            //NSLog(@"%.2f",delta);
            assert(delta >= 0);
            if (delta > lim) {
                lastidx = i;
                break;
            }
        }
        i++;
    } // for
    Contours res( lastidx);
    for (i=0; i < lastidx; i++) {
        res[i] = conts[sqdists[i].idx];
    }
    //sz = res.size();
    return res;
} // filter_outside_contours()

// Reject board if opposing lines not parallel
// or adjacent lines not at right angles
//----------------------------------------------
bool board_valid( Points board)
{
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

#pragma mark - Processing Pipeline for debugging
//=================================================

//-----------------------------------------
- (UIImage *) f00_contours:(UIImage *)img
{
    // Convert UIImage to Mat
    //cv::Mat m;
    UIImageToMat( img, _m);
    // Resize
    resize( _m, _m, 1000);
    // Grayscale
    cv::cvtColor( _m, _m, cv::COLOR_BGR2GRAY);
    //cv::GaussianBlur( _m, _m, cv::Size( 7, 7), 0, 0 );
    // Contours
    cv::Mat drawing = cv::Mat::zeros( _m.size(), CV_8UC3 );
    _cont = get_contours(_m, _canny_low, _canny_hi);
    draw_contours( _cont, drawing);
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//-----------------------------------------
- (UIImage *) f01_filtered_contours
{
    if (!_cont.size()) { return MatToUIImage( _m);}
    cv::Mat drawing = cv::Mat::zeros( _m.size(), CV_8UC3 );
    int width = self.m.cols; int height = self.m.rows;
    // Straight large ones only
    Contours filtered = filter_contours( _cont, width, height);
    draw_contours( filtered, drawing);
    UIImage *res = MatToUIImage( drawing);
    _cont = filtered;
    return res;
}
//-----------------------------------------
- (UIImage *) f02_inside_contours
{
    if (!_cont.size()) { return MatToUIImage( _m);}
    cv::Mat drawing = cv::Mat::zeros( _m.size(), CV_8UC3 );
    int width = self.m.cols; int height = self.m.rows;
    // Only contours on the board
    cv::Point board_center = get_board_center( _cont);
    draw_point( board_center, drawing);
    Contours inside = filter_outside_contours( _cont, board_center, width, height);
    draw_contours( inside, drawing);

    // Convert back to UIImage
    UIImage *res = MatToUIImage( drawing);
    _cont = inside;
    return res;
}

//-----------------------------------
- (UIImage *) f03_find_board
{
    if (!_cont.size()) { return MatToUIImage( _m);}
    cv::Mat drawing = cv::Mat::zeros( _m.size(), CV_8UC3 );
    draw_contours( _cont, drawing);
    //int width = self.m.cols; int height = self.m.rows;
    Points board = approx_poly( flatten(_cont), 4);
    board = order_points( board);
    _cont = std::vector<Points>( 1, board);
    cv::drawContours( drawing, _cont, -1, cv::Scalar(255,0,0));
    // Convert back to UIImage
    UIImage *res = MatToUIImage( drawing);
    return res;
}

#pragma mark - Release Methods
//==============================

// f00 to f03_find_board in one go
//--------------------------------------------
- (UIImage *) findBoard:(UIImage *) img
{
    UIImageToMat( img, _m);
    cv::Mat small;
    resize( _m, small, 500);
    cv::cvtColor( small, _m, cv::COLOR_BGR2GRAY);
    _cont = get_contours(_m, _canny_low, _canny_hi );
    int width = self.m.cols; int height = self.m.rows;
    _cont = filter_contours( _cont, width, height);
    if (_cont.size()) {
        cv::Point board_center = get_board_center( _cont);
        _cont = filter_outside_contours( _cont, board_center, width, height);
        Points board = approx_poly( flatten(_cont), 4);
        board = order_points( board);
        if (board_valid( board)) {
            _board = board;
        }
        if (_board.size()) {
            _cont = std::vector<Points>( 1, _board);
            cv::drawContours( small, _cont, -1, cv::Scalar(255,0,0,255));
        }
    }
    UIImage *res = MatToUIImage( small);
    return res;
}

@end





























