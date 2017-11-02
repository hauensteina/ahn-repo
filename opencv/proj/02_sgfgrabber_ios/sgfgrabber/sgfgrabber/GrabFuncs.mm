//
//  GrabFuncs.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

//#include <math.h>
//#include <complex.h>
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "Common.h"
#import "GrabFuncs.h"

#define ILOOP(n) for (int i=0; i < (n); i++ )
#define JLOOP(n) for (int j=0; j < (n); j++ )
#define KLOOP(n) for (int k=0; k < (n); k++ )

typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
typedef std::vector<cv::Point2f> Points2f;
static cv::RNG rng(12345);
double PI = M_PI;
typedef std::complex<double> cplx;
cplx I(0.0, 1.0);

#define STRETCH_FACTOR 1.1

@interface GrabFuncs()
//=======================
@property cv::Mat gray;  // Garyscale version of img
@property cv::Mat m;     // Mat with image we are working on
@property cv::Mat mboard; // Mat with the exact board in grayscale
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

//---------------------------------------------------------------------------------------
void morph_closing( cv::Mat &m, int size, int iterations, int type = cv::MORPH_RECT )
{
    cv::Mat element = cv::getStructuringElement( type,
                                                cv::Size( 2*size + 1, 2*size+1 ),
                                                cv::Point( size, size ) );
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

// Make our 4-polygon a little larger
//-------------------------------------
Points2f enlarge_board( Points board)
{
    float factor = STRETCH_FACTOR;
    board = order_points( board);
    Points diag1_stretched = stretch_line( { board[0],board[2] }, factor);
    Points diag2_stretched = stretch_line( { board[1],board[3] }, factor);
    Points2f res = { diag1_stretched[0], diag2_stretched[0], diag1_stretched[1], diag2_stretched[1] };
    return res;
}


// Zoom into an image area where pts are the four corners.
// From pyimagesearch by Adrian Rosebrock
// TODO: It's a kludge. Do it right.
//--------------------------------------------------------
void four_point_transform( const cv::Mat &img, cv::Mat &warped, Points2f pts)
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
} // four_point_transform()

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

//#---------------------------------------
//def get_boardsize_by_fft(zoomed_img):
//CLIP = 5000
//width = zoomed_img.shape[1]
//# 1D fft per row, magnitude per row, average them all into a 1D array, clip
//magspec_clip = np.clip(np.average( np.abs( np.fft.fftshift( np.fft.fft( zoomed_img))), axis=0), 0, CLIP)
//# Smooth it
//smooth_magspec = np.convolve(magspec_clip, np.bartlett(7), 'same')
//if not len(smooth_magspec) % 2:
//smooth_magspec = np.append( smooth_magspec, 0.0)
//# The first frequency peak above 9 should be close to the board size.
//half = len(smooth_magspec) // 2
//plt.subplot(111)
//plt.plot(range( -half, half+1 ), smooth_magspec)
//plt.show()
//MINSZ = 9
//highf = smooth_magspec[width // 2 + MINSZ:]
//                       maxes = scipy.signal.argrelextrema( highf, np.greater)[0] + MINSZ
//                       res = maxes[0] if len(maxes) else 0
//                       print(res)
//                       if res > 19: res = 19
//#elif res > 13: res = 13
//                       else: res = 9
//                       return res

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
    double alpha = 0.3;
    ILOOP (width) { magsum[i] = (1-alpha)*magsum[i] + alpha*old; old = magsum[i]; }
    
    // Find max
    old = magsum[7];
    int argmax = 0;
    for (int i = 7; i < 30; i++ ) {
        double cur = magsum[i];
        double nnext = magsum[i+1];
        if (cur > old && cur > nnext) {
            argmax = i;
            break;
        }
    }
    if (argmax > 15) return 19;
    else return 9;
} // get_boardsize_by_fft


#pragma mark - Processing Pipeline for debugging
//=================================================

- (UIImage *) f00_adaptive_thresh:(UIImage *)img
{
    UIImageToMat( img, _m);
    resize( _m, _m, 350);
    cv::cvtColor( _m, _gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur( _m, _m, cv::Size( 7, 7), 0, 0 );
    adaptiveThreshold(_gray, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                      7, // neighborhood_size
                      4); // constant to add. 2 to 6 is the viable range
    UIImage *res = MatToUIImage( _m);
    return(res);
}

//-----------------------------------------
- (UIImage *) f01_closing
{
    int erosion_size = 1;
    int iterations = 3;
    morph_closing( _m, erosion_size, iterations);

    UIImage *res = MatToUIImage( _m);
    return res;
}

//-----------------------------------------
- (UIImage *) f02_flood
{
    flood_from_center( _m);

    UIImage *res = MatToUIImage( _m);
    return res;
}

//-----------------------------------
- (UIImage *) f03_find_board
{
    cv::findContours( _m, _cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (!_cont.size()) { return MatToUIImage( _m);}
    cv::Mat drawing = cv::Mat::zeros( _m.size(), CV_8UC3 );
    draw_contours( _cont, drawing);
    Points board = approx_poly( flatten(_cont), 4);
    board = order_points( board);
    _cont = std::vector<Points>( 1, board);
    cv::drawContours( drawing, _cont, -1, cv::Scalar(255,0,0));
    // Convert back to UIImage
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//-----------------------------------
- (UIImage *) f04_zoom_in
{
    // Just the board
    Points2f b =  Points2f( _cont[0].begin(), _cont[0].end());
    four_point_transform( _gray, _mboard, b);

    // Zoom out a little
    Points2f board_stretched = enlarge_board( _cont[0]);
    four_point_transform( _gray, _gray, board_stretched);

    UIImage *res = MatToUIImage( _gray);
    return res;
}

//-----------------------------------
- (int) f05_get_boardsize
{
    cv::resize( _mboard, _m, cv::Size(256,256), 0, 0, cv::INTER_AREA);
    //cv::GaussianBlur( _m, _m, cv::Size( 7, 7), 0, 0 );
    
    int sz = get_boardsize_by_fft( _m);
    return sz;
}


#pragma mark - Real time implementation
//========================================

// f00 to f03_find_board in one go
//--------------------------------------------
- (UIImage *) findBoard:(UIImage *) img
{
    UIImageToMat( img, _m);
    cv::Mat small;
    resize( _m, small, 350);
    cv::cvtColor( small, _m, cv::COLOR_BGR2GRAY);
    // Threshold
    adaptiveThreshold(_m, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                      7, // neighborhood_size
                      4); // constant to add. 2 to 6 is the viable range
    // Morph closing
    int erosion_size = 1;
    int iterations = 3;
    morph_closing( _m, erosion_size, iterations);
    // Flood
    flood_from_center( _m);
    // Find a 4-polygon enclosing all remaining pixels
    cv::findContours( _m, _cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (_cont.size()) {
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





























