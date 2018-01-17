//
//  CppInterface.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

// This class is the only place where Objective-C and C++ mix.
// All other files are either pure Obj-C or pure C++.

// Don't change the order of these two,
// and don't move them down
#import "Ocv.hpp"
#import <opencv2/imgcodecs/ios.h>

#import "Common.h"
#import "Globals.h"
#include "Helpers.hpp"

#import "AppDelegate.h"
#import "BlackWhiteEmpty.hpp"
#import "BlobFinder.hpp"
#import "Boardness.hpp"
#import "Clust1D.hpp"
#import "CppInterface.h"
#import "DrawBoard.hpp"


// Pyramid filter params
#define SPATIALRAD  5
#define COLORRAD    30
#define MAXPYRLEVEL 2

extern cv::Mat mat_dbg;

@interface CppInterface()
//=======================
@property cv::Mat small_pyr; // resized image, in color, pyramid filtered
@property Points pyr_board; // Initial guess at board location

@property cv::Mat orig_img;     // Mat with image we are working on
@property cv::Mat small_zoomed;  // small, zoomed into the board
@property cv::Mat gray;  // Grayscale version of small
@property cv::Mat gray_threshed;  // gray with inv_thresh and dilation
@property cv::Mat gray_zoomed;   // Grayscale version of small, zoomed into the board
@property cv::Mat pyr_zoomed;    // zoomed version of small_pyr
@property cv::Mat pyr_gray;      // zoomed version of small_pyr, in gray
@property cv::Mat pyr_masked;    // pyr_gray with black stones masked out

@property cv::Mat gz_threshed; // gray_zoomed with inv_thresh and dilation
@property Contours cont; // Current set of contours
@property int board_sz; // board size, 9 or 19
@property Points stone_or_empty; // places where we suspect stones or empty
@property std::vector<cv::Vec2f> horizontal_lines;
@property std::vector<cv::Vec2f> vertical_lines;
@property Points2f corners;
@property Points2f corners_zoomed;
@property Points2f intersections;
@property Points2f intersections_zoomed;
@property double dy;
@property double dx;
@property std::vector<Points2f> boards; // history of board corners
@property cv::Mat white_templ;
@property cv::Mat black_templ;
@property cv::Mat empty_templ;
@property std::vector<int> diagram; // The position we detected

@end

@implementation CppInterface
//=========================

//----------------------
- (instancetype)init
{
    self = [super init];
    if (self) {
        g_docroot = [getFullPath(@"") UTF8String];
        // Load template files
        cv::Mat tmat;
        NSString *fpath;
        
        fpath = findInBundle( @"white_templ", @"yml");
        cv::FileStorage fsw( [fpath UTF8String], cv::FileStorage::READ);
        fsw["white_template"] >> _white_templ;
        
        fpath = findInBundle( @"black_templ", @"yml");
        cv::FileStorage fsb( [fpath UTF8String], cv::FileStorage::READ);
        fsb["black_template"] >> _black_templ;
        
        fpath = findInBundle( @"empty_templ", @"yml");
        cv::FileStorage fse( [fpath UTF8String], cv::FileStorage::READ);
        fse["empty_template"] >> _empty_templ;
    }
    return self;
}

#pragma mark - Pipeline Helpers
//==================================

// Load image from file
//---------------------------------------------
void load_img( NSString *fname, cv::Mat &m)
{
    UIImage *img = [UIImage imageNamed:fname];
    UIImageToMat(img, m);
}

// Reject board if opposing lines not parallel
// or adjacent lines not at right angles
//-----------------------------------------------------
bool board_valid( Points2f board, const cv::Mat &img)
{
    double screenArea = img.rows * img.cols;
    if (board.size() != 4) return false;
    double area = cv::contourArea(board);
    if (area / screenArea > 0.95) return false;
    if (area / screenArea < 0.20) return false;
    
    double par_ang1   = (180.0 / M_PI) * angle_between_lines( board[0], board[1], board[3], board[2]);
    double par_ang2   = (180.0 / M_PI) * angle_between_lines( board[0], board[3], board[1], board[2]);
    double right_ang1 = (180.0 / M_PI) * angle_between_lines( board[0], board[1], board[1], board[2]);
    double right_ang2 = (180.0 / M_PI) * angle_between_lines( board[0], board[3], board[3], board[2]);
    //double horiz_ang   = (180.0 / M_PI) * angle_between_lines( board[0], board[1], cv::Point(0,0), cv::Point(1,0));
    //NSLog(@"%f.2, %f.2, %f.2, %f.2", par_ang1,  par_ang2,  right_ang1,  right_ang2 );
    //if (abs(horiz_ang) > 20) return false;
    if (abs(par_ang1) > 20) return false;
    if (abs(par_ang2) > 30) return false;
    if (abs(right_ang1 - 90) > 20) return false;
    if (abs(right_ang2 - 90) > 20) return false;
    return true;
}


#pragma mark - Processing Pipeline for debugging
//=================================================

//--------------------------
- (UIImage *) f00_blobs: (std::vector<cv::Mat>)imgQ
{
    _board_sz=19;
    g_app.mainVC.lbDbg.text = @"blobs";
    
    NSArray *fnames = @[
                        @"board_full.jpg",
                        @"board_full_1.jpg",
                        @"board01.jpg",
                        @"board02.jpg",
                        @"board03.jpg",
                        @"board04.jpg",
                        @"board05.jpg",
                        @"board06.jpg",
                        @"board07.jpg",
                        @"board08.jpg",
                        @"board09.jpg",
                        @"board10.jpg",
                        @"board11.jpg",
                        @"board12.jpg",
                        @"board13.jpg",
                        @"board14.jpg"
                        ];
    if (_sldDbg > 0 && _sldDbg <= fnames.count) {
        //if (1) {
        load_img( fnames[_sldDbg -1], _orig_img);
        //load_img( fnames[4], _m);
        cv::rotate(_orig_img, _orig_img, cv::ROTATE_90_CLOCKWISE);
        resize( _orig_img, _small_img, 350);
        cv::cvtColor( _small_img, _small_img, CV_RGBA2RGB); // Yes, RGBA not BGR
    }
    else { // Camera
        // Pick best frame from Q
        cv::Mat best;
        int maxBlobs = -1E9;
        int bestidx = -1;
        ILOOP (SZ(imgQ) - 1) { // ignore newest frame
            _small_img = imgQ[i];
            cv::cvtColor( _small_img, _gray, cv::COLOR_RGB2GRAY);
            thresh_dilate( _gray, _gray_threshed);
            _stone_or_empty.clear();
            BlobFinder::find_empty_places( _gray_threshed, _stone_or_empty); // has to be first
            BlobFinder::find_stones( _gray, _stone_or_empty);
            _stone_or_empty = BlobFinder::clean( _stone_or_empty);
            if (SZ(_stone_or_empty) > maxBlobs) {
                maxBlobs = SZ(_stone_or_empty);
                best = _small_img;
                bestidx = i;
            }
        }
        PLOG("best idx %d\n", bestidx);
        // Reprocess the best one
        _small_img = best;
    }
    cv::cvtColor( _small_img, _gray, cv::COLOR_RGB2GRAY);
    thresh_dilate( _gray, _gray_threshed);
    _stone_or_empty.clear();
    BlobFinder::find_empty_places( _gray_threshed, _stone_or_empty); // has to be first
    BlobFinder::find_stones( _gray, _stone_or_empty);
    _stone_or_empty = BlobFinder::clean( _stone_or_empty);
    
    cv::pyrMeanShiftFiltering( _small_img, _small_pyr, SPATIALRAD, COLORRAD, MAXPYRLEVEL );
    
    // Show results
    cv::Mat drawing = _small_img.clone();
    draw_points( _stone_or_empty, drawing, 2, cv::Scalar( 255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f00_blobs()



// Find vertical grid lines
//----------------------------------
- (UIImage *) f01_vert_lines
{
    static int state = 0;
    cv::Mat drawing;
    static std::vector<cv::Vec2f> all_vert_lines;
    
    switch (state) {
        case 0:
        {
            g_app.mainVC.lbDbg.text = @"verts";
            _vertical_lines = homegrown_vert_lines( _stone_or_empty);
            all_vert_lines = _vertical_lines;
            break;
        }
        case 1:
        {
            g_app.mainVC.lbDbg.text = @"dedup";
            dedup_verticals( _vertical_lines, _gray);
            break;
        }
        case 2:
        {
            g_app.mainVC.lbDbg.text = @"filter";
            filter_vert_lines( _vertical_lines);
            break;
        }
        case 3:
        {
            g_app.mainVC.lbDbg.text = @"fix";
            const double x_thresh = 4.0;
            fix_vertical_lines( _vertical_lines, all_vert_lines, _gray, x_thresh);
            break;
        }
        default:
            state = 0;
            return NULL;
    } // switch
    state++;
    
    // Show results
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color(true);
    ISLOOP( _vertical_lines) {
        draw_polar_line( _vertical_lines[i], drawing, get_color());
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f01_vert_lines()

// Replace close clusters of vert lines by their average.
//-----------------------------------------------------------------------------------
void dedup_verticals( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    if (SZ(lines) < 3) return;
    // Cluster by x in the middle
    //const double wwidth = 32.0;
    const double wwidth = 8.0;
    const double middle_y = img.rows / 2.0;
    const int min_clust_size = 0;
    auto Getter =  [middle_y](cv::Vec2f line) { return x_from_y( middle_y, line); };
    auto vert_line_cuts = Clust1D::cluster( lines, wwidth, Getter);
    std::vector<std::vector<cv::Vec2f> > clusters;
    Clust1D::classify( lines, vert_line_cuts, min_clust_size, Getter, clusters);
    
    // Average the clusters into single lines
    lines.clear();
    ISLOOP (clusters) {
        auto &clust = clusters[i];
        double theta = vec_avg( clust, [](cv::Vec2f line){ return line[1]; });
        double rho   = vec_avg( clust, [](cv::Vec2f line){ return line[0]; });
        cv::Vec2f line( rho, theta);
        lines.push_back( line);
    }
}

// Replace close clusters of horiz lines by their average.
//-----------------------------------------------------------------------------------
void dedup_horizontals( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    if (SZ(lines) < 3) return;
    // Cluster by y in the middle
    const double wwidth = 32.0;
    const double middle_x = img.cols / 2.0;
    const int min_clust_size = 0;
    auto Getter =  [middle_x](cv::Vec2f line) { return y_from_x( middle_x, line); };
    auto horiz_line_cuts = Clust1D::cluster( lines, wwidth, Getter);
    std::vector<std::vector<cv::Vec2f> > clusters;
    Clust1D::classify( lines, horiz_line_cuts, min_clust_size, Getter, clusters);
    
    // Average the clusters into single lines
    lines.clear();
    ISLOOP (clusters) {
        auto &clust = clusters[i];
        double theta = vec_avg( clust, [](cv::Vec2f line){ return line[1]; });
        double rho   = vec_avg( clust, [](cv::Vec2f line){ return line[0]; });
        cv::Vec2f line( rho, theta);
        lines.push_back( line);
    }
}

// Adjacent lines should have similar slope
//-----------------------------------------------------------------------------
void filter_vert_lines( std::vector<cv::Vec2f> &vlines)
{
    const double eps = 10.0;
    std::sort( vlines.begin(), vlines.end(), [](cv::Vec2f &a, cv::Vec2f &b) { return a[0] < b[0]; });
    int med_idx = good_center_line( vlines);
    if (med_idx < 0) return;
    const double med_theta = vlines[med_idx][1];
    // Going left and right, theta should not change abruptly
    std::vector<cv::Vec2f> good;
    good.push_back( vlines[med_idx]);
    const double EPS = eps * PI/180;
    double prev_theta;
    // right
    prev_theta = med_theta;
    for (int i = med_idx+1; i < SZ(vlines); i++ ) {
        double d = fabs( vlines[i][1] - prev_theta) + fabs( vlines[i][1] - med_theta);
        if (d < EPS) {
            good.push_back( vlines[i]);
            prev_theta = vlines[i][1];
        }
    }
    // left
    prev_theta = med_theta;
    for (int i = med_idx-1; i >= 0; i-- ) {
        double d = fabs( vlines[i][1] - prev_theta) + fabs( vlines[i][1] - med_theta);
        if (d < EPS) {
            good.push_back( vlines[i]);
            prev_theta = vlines[i][1];
        }
    }
    //std::sort( good.begin(), good.end(), [](cv::Vec2f a, cv::Vec2f b) { return a[0] < b[0]; });
    //good.clear();
    //good.push_back( _vertical_lines[med_idx]);
    vlines = good;
} // filter_vert_lines()

// Adjacent lines should have similar slope
//-----------------------------------------------------------------------------
void filter_horiz_lines( std::vector<cv::Vec2f> &vlines)
{
    const double eps = 1.1;
    std::sort( vlines.begin(), vlines.end(), [](cv::Vec2f &a, cv::Vec2f &b) { return a[0] < b[0]; });
    int med_idx = good_center_line( vlines);
    if (med_idx < 0) return;
    double theta = vlines[med_idx][1];
    // Going left and right, theta should not change abruptly
    std::vector<cv::Vec2f> good;
    good.push_back( vlines[med_idx]);
    const double EPS = eps * PI/180;
    double prev_theta;
    // right
    prev_theta = theta;
    for (int i = med_idx+1; i < SZ(vlines); i++ ) {
        if (fabs( vlines[i][1] - prev_theta) < EPS) {
            good.push_back( vlines[i]);
            prev_theta = vlines[i][1];
        }
    }
    // left
    prev_theta = theta;
    for (int i = med_idx-1; i >= 0; i-- ) {
        if (fabs( vlines[i][1] - prev_theta) < EPS) {
            good.push_back( vlines[i]);
            prev_theta = vlines[i][1];
        }
    }
    //std::sort( good.begin(), good.end(), [](cv::Vec2f a, cv::Vec2f b) { return a[0] < b[0]; });
    //good.clear();
    //good.push_back( _vertical_lines[med_idx]);
    vlines = good;
} // filter_horiz_lines()

// Find a line close to the middle with roughly median theta.
// The lines should be sorted by rho.
//--------------------------------------------------------------------------
int good_center_line( const std::vector<cv::Vec2f> &lines)
{
    const int r = 2;
    //const double EPS = 4 * PI/180;
    auto thetas = vec_extract( lines, [](cv::Vec2f line) { return line[1]; } );
    auto med_theta = vec_median( thetas);
    
    // Find a line close to the middle where theta is close to median theta
    int half = SZ(lines)/2;
    double mind = 1E9;
    int minidx = -1;
    ILOOP (r+1) {
        if (half - i >= 0) {
            double d = fabs( med_theta - thetas[half-i]);
            if (d < mind) {
                mind = d;
                minidx = half - i;
            }
        }
        if (half + i < SZ(lines)) {
            double d = fabs( med_theta - thetas[half+i]);
            if (d < mind) {
                mind = d;
                minidx = half + i;
            }
        }
    } // ILOOP
    return minidx;
} // good_center_line()

// Find line where top_x and bot_x match best.
//---------------------------------------------------------------------------------------------------------------
int closest_vert_line( const std::vector<cv::Vec2f> &lines, double top_x, double bot_x, double top_y, double bot_y,
                      double &min_dtop, double &min_dbot, double &min_err, double &min_top_rho, double &min_bot_rho) // out
{
    std::vector<double> top_rhos = vec_extract( lines,
                                               [top_y](cv::Vec2f a) { return x_from_y( top_y, a); });
    std::vector<double> bot_rhos = vec_extract( lines,
                                               [bot_y](cv::Vec2f a) { return x_from_y( bot_y, a); });
    int minidx = -1;
    min_err = 1E9;
    ISLOOP (top_rhos) {
        double dtop = fabs( top_rhos[i] - top_x );
        double dbot = fabs( bot_rhos[i] - bot_x );
        double err = dtop + dbot;
        if (err < min_err) {
            minidx = i;
            min_err = err;
            min_dtop = dtop;
            min_dbot = dbot;
            min_top_rho = top_rhos[i];
            min_bot_rho = bot_rhos[i];
        }
    }
    return minidx;
} // closest_vert_line()

// Find the x-change per line in in upper and lower screen area and synthesize
// the whole bunch starting at the middle. Replace synthesized lines with real
// ones if close enough.
//------------------------------------------------------------------------------------------------
void fix_vertical_lines( std::vector<cv::Vec2f> &lines, const std::vector<cv::Vec2f> &all_vert_lines,
                        const cv::Mat &img, double x_thresh = 4.0)
{
    const double width = img.cols;
    const int top_y = 0.2 * img.rows;
    const int bot_y = 0.8 * img.rows;
    //const int mid_y = 0.5 * img.rows;
    
    std::sort( lines.begin(), lines.end(),
              [bot_y](cv::Vec2f a, cv::Vec2f b) {
                  return x_from_y( bot_y, a) < x_from_y( bot_y, b);
              });
    std::vector<double> top_rhos = vec_extract( lines,
                                               [top_y](cv::Vec2f a) { return x_from_y( top_y, a); });
    std::vector<double> bot_rhos = vec_extract( lines,
                                               [bot_y](cv::Vec2f a) { return x_from_y( bot_y, a); });
    auto d_top_rhos = vec_delta( top_rhos);
    auto d_bot_rhos = vec_delta( bot_rhos);
    vec_filter( d_top_rhos, [](double d){ return d > 5 && d < 20;});
    vec_filter( d_bot_rhos, [](double d){ return d > 8 && d < 25;});
    double d_top_rho = vec_median( d_top_rhos);
    double d_bot_rho = vec_median( d_bot_rhos);
    
    // Find a good line close to the middle
    int good_idx = good_center_line( lines);
    if (good_idx < 0) {
        lines.clear();
        return;
    }
    cv::Vec2f med_line = lines[good_idx];
    
    // Interpolate the rest
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    double top_rho, bot_rho;
    // If there is a close line, use it. Else interpolate.
    const double X_THRESH = x_thresh; //6;
    // Lines to the right
    top_rho = x_from_y( top_y, med_line);
    bot_rho = x_from_y( bot_y, med_line);
    ILOOP(100) {
        top_rho += d_top_rho;
        bot_rho += d_bot_rho;
        //int close_idx = vec_closest( bot_rhos, bot_rho);
        //int close_idx = closest_vert_line( all_vert_lines, top_rho, bot_rho, top_y, bot_y);
        double dtop, dbot, err, top_x, bot_x;
        closest_vert_line( all_vert_lines, top_rho, bot_rho, top_y, bot_y, // in
                          dtop, dbot, err, top_x, bot_x); // out
        //double dbot = fabs( bot_rho - bot_rhos[close_idx]);
        //double dtop = fabs( top_rho - top_rhos[close_idx]);
        if (dbot < X_THRESH && dtop < X_THRESH) {
            top_rho   = top_x;
            bot_rho   = bot_x;
        }
        cv::Vec2f line = segment2polar( cv::Vec4f( top_rho, top_y, bot_rho, bot_y));
        if (top_rho > width) break;
        if (i > 19) break;
        //if (x_from_y( mid_y, line) > width) break;
        synth_lines.push_back( line);
    } // ILOOP
    // Lines to the left
    top_rho = x_from_y( top_y, med_line);
    bot_rho = x_from_y( bot_y, med_line);
    ILOOP(100) {
        top_rho -= d_top_rho;
        bot_rho -= d_bot_rho;
        //int close_idx = vec_closest( bot_rhos, bot_rho);
        //int close_idx = closest_vert_line( lines, top_rho, bot_rho, top_y, bot_y);
        double dtop, dbot, err, top_x, bot_x;
        closest_vert_line( all_vert_lines, top_rho, bot_rho, top_y, bot_y, // in
                          dtop, dbot, err, top_x, bot_x); // out
        if (dbot < X_THRESH && dtop < X_THRESH) {
            //PLOG("repl %d\n",i);
            top_rho   = top_x;
            bot_rho   = bot_x;
        }
        cv::Vec2f line = segment2polar( cv::Vec4f( top_rho, top_y, bot_rho, bot_y));
        if (top_rho < 0) break;
        if (i > 19) break;
        //if (x_from_y( mid_y, line) < 0) break;
        synth_lines.push_back( line);
    } // ILOOP
    std::sort( synth_lines.begin(), synth_lines.end(),
              [bot_y](cv::Vec2f a, cv::Vec2f b) {
                  return x_from_y( bot_y, a) < x_from_y( bot_y, b);
              });
    lines = synth_lines;
} // fix_vertical_lines()


// Find the median distance between vert lines for given horizontal.
// We use the result to find the next horizontal line.
// The idea is that on a grid, horizontal and vertical spacing are the same,
// and if we know one, we know the other.
//--------------------------------------------------------------------------------------
double hspace_at_line( const std::vector<cv::Vec2f> &vert_lines, cv::Vec2f hline)
{
    std::vector<double> dists;
    Point2f prev;
    ISLOOP (vert_lines) {
        //cv::Vec4f seg = polar2segment( vert_lines[i]);
        Point2f p = intersection( vert_lines[i], hline);
        if (i) {
            double d = cv::norm( p - prev);
            dists.push_back( d);
        }
        prev = p;
    }
    double res = vec_median( dists);
    //double res = dists[SZ(dists)/2];
    return res;
} // hspace_at_y()

// Similarity between two horizontal lines.
// y_distance**2 to the left plus y_distance**2 to the right.
//--------------------------------------------------------------------
double h_line_similarity( cv::Vec2f a, cv::Vec2f b, double middle_x)
{
    const int r = 50;
    double aleft  = y_from_x( middle_x - r, a);
    double bleft  = y_from_x( middle_x - r, b);
    double aright = y_from_x( middle_x + r, a);
    double bright = y_from_x( middle_x + r, b);
    double res = sqrt( SQR( aleft - bleft) + SQR( aright - bright));
    return res;
} // h_line_similarity()

// Find closest line in a bunch of horiz lines
//--------------------------------------------------------------------------------------------------
int closest_hline( cv::Vec2f line, const std::vector<cv::Vec2f> &hlines, double middle_x, double &d)
{
    int minidx = -1;
    d = 1E9;
    ISLOOP (hlines) {
        if (h_line_similarity( line, hlines[i], middle_x) < d) {
            d = h_line_similarity( line, hlines[i], middle_x);
            minidx = i;
        }
    }
    return minidx;
} // closest_hline()

// Similarity between two horizontal lines.
// x_distance**2 above plus x_distance**2 below.
//--------------------------------------------------------------------
double v_line_similarity( cv::Vec2f a, cv::Vec2f b, double middle_y)
{
    const int r = 50;
    double atop  = x_from_y( middle_y - r, a);
    double btop  = x_from_y( middle_y - r, b);
    double abot  = x_from_y( middle_y + r, a);
    double bbot  = x_from_y( middle_y + r, b);
    double res = sqrt( SQR( atop - btop) + SQR( abot - bbot));
    return res;
} // v_line_similarity

// Find the change per line in rho and theta and synthesize the whole bunch
// starting at the middle. Replace synthesized lines with real ones if close enough.
//---------------------------------------------------------------------------------------------
void fix_horiz_lines( std::vector<cv::Vec2f> &lines_, const std::vector<cv::Vec2f> &vert_lines,
                     const cv::Mat &img)
{
    const double middle_x = img.cols / 2.0;
    const double height = img.rows;
    
    // Convert hlines to chlines (center y + angle)
    std::vector<cv::Vec2f> lines;
    ISLOOP (lines_) {
        lines.push_back( polar2changle( lines_[i], middle_x));
    }
    std::sort( lines.begin(), lines.end(), [](cv::Vec2f a, cv::Vec2f b) { return a[0] < b[0]; } );
    lines_.clear();
    ISLOOP (lines) { lines_.push_back( changle2polar( lines[i], middle_x)); }
    
    auto rhos   = vec_extract( lines, [](cv::Vec2f line) { return line[0]; } );
    //auto thetas = vec_extract( lines, [](cv::Vec2f line) { return line[1]; } );
    auto d_rhos   = vec_delta( rhos);
    //vec_filter( d_rhos, [](double d){ return d > 10;});
    
    int good_idx = good_center_line( lines);
    if (good_idx < 0) {
        lines.clear();
        return;
    }
    cv::Vec2f med_line = lines[good_idx];
    
    
    // Interpolate the rest
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    
    double med_rho = med_line[0];
    double med_d_rho = vec_median( d_rhos);
    double alpha = RAT( hspace_at_line( vert_lines, cv::Vec2f( 0, PI/2)),
                       hspace_at_line( vert_lines, cv::Vec2f( med_rho, PI/2)));
    double dd_rho_per_y = RAT( med_d_rho * (1.0 - alpha), med_rho);
    
    double rho, theta, d_rho;
    cv::Vec2f line;
    
    // Lines below
    //d_rho = med_d_rho;
    d_rho = hspace_at_line( vert_lines, cv::Vec2f( med_rho, PI/2));
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        double old_rho = rho;
        rho += d_rho;
        double d;
        int close_idx = closest_hline( changle2polar( cv::Vec2f( rho, theta), middle_x), lines_, middle_x, d);
        if (d < d_rho * 0.6) {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
            d_rho = rho - old_rho;
        }
        else {
            d_rho += (rho - old_rho) * dd_rho_per_y;
            //PLOG("synth %d\n",i);
        }
        if (rho > height) break;
        cv::Vec2f line( rho,theta);
        synth_lines.push_back( line);
    } // ILOOP
    
    // Lines above
    //d_rho = med_d_rho;
    d_rho = 0.9 * hspace_at_line( vert_lines, cv::Vec2f( med_rho, PI/2));
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        double old_rho = rho;
        rho -= d_rho;
        double d;
        int close_idx = closest_hline( changle2polar( cv::Vec2f( rho, theta), middle_x), lines_, middle_x, d);
        if (d < d_rho * 0.6) {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
            d_rho = old_rho - rho;
        }
        else {
            d_rho += (rho - old_rho) * dd_rho_per_y;
            //PLOG("i %d d_rho %.2f\n", i, d_rho);
        }
        if (rho < 0) break;
        if (d_rho < 3) break;
        cv::Vec2f line( rho,theta);
        synth_lines.push_back( line);
    } // ILOOP
    // Sort top to bottom
    std::sort( synth_lines.begin(), synth_lines.end(),
              [](cv::Vec2f line1, cv::Vec2f line2) {
                  return line1[0] < line2[0];
              });
    lines_.clear();
    ISLOOP (synth_lines) { lines_.push_back( changle2polar( synth_lines[i], middle_x)); }
} // fix_horiz_lines()

// Convert horizontal (roughly) polar line to a pair
// y_at_middle, angle
//--------------------------------------------------------------
cv::Vec2f polar2changle( const cv::Vec2f pline, double middle_x)
{
    cv::Vec2f res;
    double y_at_middle = y_from_x( middle_x, pline);
    double angle;
    angle = -(pline[1] - PI/2);
    res = cv::Vec2f( y_at_middle, angle);
    return res;
}

// Convert a pair (y_at_middle, angle) to polar
//---------------------------------------------------------------
cv::Vec2f changle2polar( const cv::Vec2f cline, double middle_x)
{
    cv::Vec2f res;
    cv::Vec4f seg( middle_x, cline[0], middle_x + 1, cline[0] - tan(cline[1]));
    res = segment2polar( seg);
    return res;
}

// Convert vertical (roughly) polar line to a pair
// x_at_middle, angle
//--------------------------------------------------------------
cv::Vec2f polar2cvangle( const cv::Vec2f pline, double middle_y)
{
    cv::Vec2f res;
    double x_at_middle = x_from_y( middle_y, pline);
    double angle;
    angle = -pline[1];
    res = cv::Vec2f( x_at_middle, angle);
    return res;
}

// Convert a pair (x_at_middle, angle) to polar
//---------------------------------------------------------------
cv::Vec2f cvangle2polar( const cv::Vec2f cline, double middle_y)
{
    cv::Vec2f res;
    cv::Vec4f seg( cline[0], middle_y , cline[0] + tan(cline[1]), middle_y + 1);
    res = segment2polar( seg);
    return res;
}

//-----------------------------
- (UIImage *) f02_horiz_lines
{
    static int state = 0;
    cv::Mat drawing;
    
    switch (state) {
        case 0:
        {
            g_app.mainVC.lbDbg.text = @"horizontals";
            _horizontal_lines = homegrown_horiz_lines( _stone_or_empty);
            break;
        }
        case 1:
        {
            g_app.mainVC.lbDbg.text = @"dedup";
            dedup_horizontals( _horizontal_lines, _gray);
            break;
        }
        case 2:
        {
            g_app.mainVC.lbDbg.text = @"filter";
            filter_horiz_lines( _horizontal_lines);
            break;
        }
        case 3:
        {
            g_app.mainVC.lbDbg.text = @"fix";
            fix_horiz_lines( _horizontal_lines, _vertical_lines, _gray);
            break;
        }
        default:
            state = 0;
            return NULL;
    } // switch
    state++;
    
    // Show results
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color( true);
    ISLOOP (_horizontal_lines) {
        cv::Scalar col = get_color();
        draw_polar_line( _horizontal_lines[i], drawing, col);
    }
    //draw_polar_line( ratline, drawing, cv::Scalar( 255,128,64));
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//--------------------------------------------------------
int count_points_on_line( cv::Vec2f line, Points pts)
{
    int res = 0;
    for (auto p:pts) {
        double d = fabs(dist_point_line( p, line));
        if (d < 0.75) {
            res++;
        }
    }
    return res;
}

// Find a vertical line thru pt which hits a lot of other points
// PRECONDITION: allpoints must be sorted by y
//---------------------------------------------------------------------------------------
cv::Vec2f find_vert_line_thru_point( const Points &allpoints, cv::Point pt, int &maxhits)
{
    // Find next point below.
    //const double RHO_EPS = 10;
    const double THETA_EPS = /* 10 */ 20 * PI / 180;
    maxhits = -1;
    cv::Vec2f res;
    for (auto p: allpoints) {
        if (p.y <= pt.y) continue;
        Points pts = { pt, p };
        //cv::Vec2f newline = fit_pline( pts);
        cv::Vec2f newline = segment2polar( cv::Vec4f( pt.x, pt.y, p.x, p.y));
        if (fabs(newline[1]) < THETA_EPS ) {
            int nhits = count_points_on_line( newline, allpoints);
            if (nhits > maxhits) {
                maxhits = nhits;
                res = newline;
            }
        }
    }
    //PLOG( "maxhits:%d\n", maxhits);
    //int tt = count_points_on_line( res, allpoints);
    return res;
} // find_vert_line_thru_point()

// Find a horiz line thru pt which hits a lot of other points
// PRECONDITION: allpoints must be sorted by x
//------------------------------------------------------------------
cv::Vec2f find_horiz_line_thru_point( const Points &allpoints, cv::Point pt)
{
    // Find next point to the right.
    //const double RHO_EPS = 10;
    const double THETA_EPS = 5 * PI / 180;
    int maxhits = -1;
    cv::Vec2f res = {0,0};
    for (auto p: allpoints) {
        if (p.x <= pt.x) continue;
        Points pts = { pt, p };
        //cv::Vec2f newline = fit_pline( pts);
        cv::Vec2f newline = segment2polar( cv::Vec4f( pt.x, pt.y, p.x, p.y));
        if (fabs( fabs( newline[1]) - PI/2) < THETA_EPS ) {
            int nhits = count_points_on_line( newline, allpoints);
            if (nhits > maxhits) {
                maxhits = nhits;
                res = newline;
            }
        }
    }
    return res;
} // find_horiz_line_thru_point()

// Homegrown method to find vertical line candidates, as a replacement
// for thinning Hough lines.
//-----------------------------------------------------------------------------
std::vector<cv::Vec2f> homegrown_vert_lines( Points pts)
{
    std::vector<cv::Vec2f> res;
    // Find points in quartile with lowest y
    std::sort( pts.begin(), pts.end(), [](Point2f p1, Point2f p2) { return p1.y < p2.y; } );
    Points top_points( SZ(pts)/4);
    std::copy_n ( pts.begin(), SZ(pts)/4, top_points.begin() );
    // For each point, find a line that hits many other points
    for (auto tp: top_points) {
        int nhits;
        cv::Vec2f newline = find_vert_line_thru_point( pts, tp, nhits);
        //PLOG (">>>>>>>%d\n",nhits);
        if (/*nhits > 5 &&*/ newline[0] != 0) {
            res.push_back( newline);
        }
    }
    return res;
} // homegrown_vert_lines()

// Homegrown method to find horizontal line candidates
//-----------------------------------------------------------------------------
std::vector<cv::Vec2f> homegrown_horiz_lines( Points pts)
{
    std::vector<cv::Vec2f> res;
    // Find points in quartile with lowest x
    std::sort( pts.begin(), pts.end(), [](Point2f p1, Point2f p2) { return p1.x < p2.x; } );
    Points left_points( SZ(pts)/4);
    std::copy_n ( pts.begin(), SZ(pts)/4, left_points.begin() );
    // For each point, find a line that hits many other points
    for (auto tp: left_points) {
        cv::Vec2f newline = find_horiz_line_thru_point( pts, tp);
        if (newline[0] != 0) {
            res.push_back( newline);
        }
    }
    return res;
} // homegrown_horiz_lines()


// Among the largest two in m1, choose the one where m2 is larger
//------------------------------------------------------------------
cv::Point tiebreak( const cv::Mat &m1, const cv::Mat &m2)
{
    cv::Mat tmp = m1.clone();
    double m1min, m1max;
    cv::Point m1minloc, m1maxloc;
    cv::minMaxLoc( tmp, &m1min, &m1max, &m1minloc, &m1maxloc);
    cv::Point largest = m1maxloc;
    tmp.at<uint8_t>(largest) = 0;
    cv::minMaxLoc( tmp, &m1min, &m1max, &m1minloc, &m1maxloc);
    cv::Point second = m1maxloc;
    
    cv::Point res = largest;
    if (m2.at<uint8_t>(second) > m2.at<uint8_t>(largest)) {
        res = second;
    }
    return res;
} // tiebreak

// Use horizontal and vertical lines to find corners such that the board best matches the points we found
//-----------------------------------------------------------------------------------------------------------
Points2f find_corners( const Points blobs, std::vector<cv::Vec2f> &horiz_lines, std::vector<cv::Vec2f> &vert_lines,
                      const Points2f &intersections, const cv::Mat &img, const cv::Mat &threshed, int board_sz = 19)
{
    if (SZ(horiz_lines) < 3 || SZ(vert_lines) < 3) return Points2f();
    
    Boardness bness( intersections, blobs, img, board_sz, horiz_lines, vert_lines);
    cv::Mat &edgeness  = bness.edgeness();
    cv::Mat &blobness  = bness.blobness();
    
    cv::Point max_loc = tiebreak( blobness, edgeness);
    //cv::Point min_loc, max_loc; double mmin, mmax;
    //cv::minMaxLoc(blobness, &mmin, &mmax, &min_loc, &max_loc);
    
    cv::Point tl = max_loc;
    cv::Point tr( tl.x + board_sz-1, tl.y);
    cv::Point br( tl.x + board_sz-1, tl.y + board_sz-1);
    cv::Point bl( tl.x, tl.y + board_sz-1);
    
    // Return the board lines only
    horiz_lines = vec_slice( horiz_lines, max_loc.y, board_sz);
    vert_lines  = vec_slice( vert_lines, max_loc.x, board_sz);
    
    // Mark corners for visualization
    mat_dbg = bness.m_pyrpix_edgeness.clone();
    mat_dbg.at<cv::Vec3b>( pf2p(tl)) = cv::Vec3b( 255,0,0);
    mat_dbg.at<cv::Vec3b>( pf2p(tr)) = cv::Vec3b( 255,0,0);
    mat_dbg.at<cv::Vec3b>( pf2p(bl)) = cv::Vec3b( 255,0,0);
    mat_dbg.at<cv::Vec3b>( pf2p(br)) = cv::Vec3b( 255,0,0);
    cv::resize( mat_dbg, mat_dbg, img.size(), 0,0, CV_INTER_NN);
    
    auto isec2pf = [&blobness, &intersections](cv::Point p) { return p2pf( intersections[p.y*blobness.cols + p.x]); };
    Points2f corners = { isec2pf(tl), isec2pf(tr), isec2pf(br), isec2pf(bl) };
    return corners;
} // find_corners()

// Get intersections of two sets of lines
//-------------------------------------------------------------------
Points2f get_intersections( const std::vector<cv::Vec2f> &hlines,
                           const std::vector<cv::Vec2f> &vlines)
{
    Points2f res;
    RSLOOP( hlines) {
        cv::Vec2f hl = hlines[r];
        CSLOOP( vlines) {
            cv::Vec2f vl = vlines[c];
            Point2f pf = intersection( hl, vl);
            res.push_back( pf);
        }
    }
    return res;
}

// Find the corners
//----------------------------
- (UIImage *) f03_corners
{
    g_app.mainVC.lbDbg.text = @"find corners";
    
    _intersections = get_intersections( _horizontal_lines, _vertical_lines);
    //auto crosses = find_crosses( _gray_threshed, intersections);
    _corners.clear();
    do {
        if (SZ( _horizontal_lines) > 55) break;
        if (SZ( _horizontal_lines) < 5) break;
        if (SZ( _vertical_lines) > 55) break;
        if (SZ( _vertical_lines) < 5) break;
        _corners = find_corners( _stone_or_empty, _horizontal_lines, _vertical_lines,
                                _intersections, _small_pyr, _gray_threshed );
        // Intersections for only the board lines
        _intersections = get_intersections( _horizontal_lines, _vertical_lines);
    } while(0);
    
    // Show results
    //cv::Mat drawing = _small_pyr.clone();
    //cv::Mat drawing; cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    //mat_dbg.convertTo( mat_dbg, CV_8UC1);
    //cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    //double alpha = 0.5;
    //cv::addWeighted( _small, alpha, drawing, 1-alpha, 0, drawing);
    //draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( mat_dbg);
    return res;
} // f03_corners()

// Unwarp the square defined by corners
//------------------------------------------------------------------------
void zoom_in( const cv::Mat &img, const Points2f &corners, cv::Mat &dst, cv::Mat &M)
{
    int marg = img.cols / 20;
    // Target square for transform
    Points2f square = {
        cv::Point( marg, marg),
        cv::Point( img.cols - marg, marg),
        cv::Point( img.cols - marg, img.cols - marg),
        cv::Point( marg, img.cols - marg) };
    M = cv::getPerspectiveTransform(corners, square);
    cv::warpPerspective(img, dst, M, cv::Size( img.cols, img.rows));
}

// Fill image outside of board with average. Helps with adaptive thresholds.
//----------------------------------------------------------------------------
void fill_outside_with_average_gray( cv::Mat &img, const Points2f &corners)
{
    uint8_t mean = cv::mean( img)[0];
    img.forEach<uint8_t>( [&mean, &corners](uint8_t &v, const int *p)
                         {
                             int x = p[1]; int y = p[0];
                             if (x < corners[0].x - 10) v = mean;
                             else if (x > corners[1].x + 10) v = mean;
                             if (y < corners[0].y - 10) v = mean;
                             else if (y > corners[3].y + 10) v = mean;
                         });
} // fill_outside_with_average_gray()

//----------------------------------------------------------------------------
void fill_outside_with_average_rgb( cv::Mat &img, const Points2f &corners)
{
    cv::Scalar smean = cv::mean( img);
    Pixel mean( smean[0], smean[1], smean[2]);
    img.forEach<Pixel>( [&mean, &corners](Pixel &v, const int *p)
                       {
                           int x = p[1]; int y = p[0];
                           if (x < corners[0].x - 10) v = mean;
                           else if (x > corners[1].x + 10) v = mean;
                           if (y < corners[0].y - 10) v = mean;
                           else if (y > corners[3].y + 10) v = mean;
                       });
} // fill_outside_with_average_rgb()

// Zoom in
//----------------------------
- (UIImage *) f04_zoom_in
{
    g_app.mainVC.lbDbg.text = @"zoom";
    cv::Mat threshed;
    cv::Mat dst;
    if (SZ(_corners) == 4) {
        cv::Mat M;
        zoom_in( _gray,  _corners, _gray_zoomed, M);
        zoom_in( _small_img, _corners, _small_zoomed, M);
        zoom_in( _small_pyr, _corners, _pyr_zoomed, M);
        cv::cvtColor( _pyr_zoomed, _pyr_gray, cv::COLOR_RGB2GRAY);
        cv::perspectiveTransform( _corners, _corners_zoomed, M);
        cv::perspectiveTransform( _intersections, _intersections_zoomed, M);
        fill_outside_with_average_gray( _gray_zoomed, _corners_zoomed);
        fill_outside_with_average_rgb( _small_zoomed, _corners_zoomed);
        fill_outside_with_average_rgb( _pyr_zoomed, _corners_zoomed);
        
        thresh_dilate( _gray_zoomed, _gz_threshed, 4);
        
        // Try stuff
        cv::Mat tt;
        //cv::cvtColor( _gray_zoomed, tt, cv::COLOR_RGB2GRAY);
        cv::GaussianBlur( _gray_zoomed, tt, cv::Size(5,5),0,0);
        //thresh_dilate( dst, dst, 3);
        //cv::threshold( tt, dst, 50, 255, cv::THRESH_BINARY); // Black is black
        //cv::threshold( tt, dst, 200, 255, cv::THRESH_BINARY);
        //cv::adaptiveThreshold( _gray_zoomed, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -50); // good
        //cv::cvtColor( _pyr_zoomed, dst, cv::COLOR_RGB2GRAY);
        //cv::adaptiveThreshold( _gray_zoomed, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -50); // good
        //cv::GaussianBlur( dst, tt, cv::Size(9,9),0,0);
        cv::adaptiveThreshold( tt, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 50);
        //cv::adaptiveThreshold( tt, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, -30);
        //int nhood_sz =  25;
        //double thresh = -32;
        //cv::adaptiveThreshold( tt, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
        //                      nhood_sz, thresh);
        //        thresh_dilate( tt, _gz_threshed, 3);
    }
    //    cv::Mat tt;
    //    cv::adaptiveThreshold( _gray_zoomed, tt, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
    //                          5 /* 11 */ ,  // neighborhood_size
    //                          3);  // threshold
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gz_threshed, drawing, cv::COLOR_GRAY2RGB);
    //cv::cvtColor( dst, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP (_intersections_zoomed) {
        Point2f p = _intersections_zoomed[i];
        draw_square( p, 3, drawing, cv::Scalar(255,0,0));
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f04_zoom_in()

// Dark places to find B stones
//-----------------------------------------------------------
- (UIImage *) f05_dark_places
{
    g_app.mainVC.lbDbg.text = @"adaptive dark";
    //_corners = _corners_zoomed;
    
    cv::Mat dark_places;
    //cv::GaussianBlur( _gray_zoomed, dark_places, cv::Size(9,9),0,0);
    //cv::adaptiveThreshold( dark_places, dark_places, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 50);
    cv::adaptiveThreshold( _pyr_gray, dark_places, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 50);
    //@@@
    // Show results
    cv::Mat drawing;
    cv::cvtColor( dark_places, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP (_intersections_zoomed) {
        Point2f p = _intersections_zoomed[i];
        draw_square( p, 3, drawing, cv::Scalar(255,0,0));
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f05_dark_places()

// Replace dark places with average to make white dynamic threshold work
//-----------------------------------------------------------------------
- (UIImage *) f06_mask_dark
{
    g_app.mainVC.lbDbg.text = @"hide dark";
    
    uint8_t mean = cv::mean( _pyr_gray)[0];
    cv::Mat black_places;
    cv::adaptiveThreshold( _pyr_gray, black_places, mean, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 50);
    _pyr_masked = _pyr_gray.clone();
    // Copy over if not zero
    _pyr_masked.forEach<uint8_t>( [&black_places](uint8_t &v, const int *p)
                                 {
                                     int row = p[0]; int col = p[1];
                                     if (auto p = black_places.at<uint8_t>( row,col)) {
                                         v = p;
                                     }
                                 });
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _pyr_masked, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP (_intersections_zoomed) {
        Point2f p = _intersections_zoomed[i];
        draw_square( p, 3, drawing, cv::Scalar(255,0,0));
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f06_mask_dark()


// Find White places
//----------------------------------------
- (UIImage *) f07_white_holes
{
    g_app.mainVC.lbDbg.text = @"adaptive bright";
    
    // The White stones become black holes, all else is white
    int nhood_sz =  25;
    double thresh = -32;
    cv::Mat white_holes;
    cv::adaptiveThreshold( _pyr_masked, white_holes, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          nhood_sz, thresh);
    //cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(2,2));
    //cv::dilate( white_holes, white_holes, element );
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( white_holes, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP (_intersections_zoomed) {
        Point2f p = _intersections_zoomed[i];
        draw_square( p, 3, drawing, cv::Scalar(255,0,0));
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f07_white_holes()

// Visualize features, one per intersection.
//------------------------------------------------------------------------------------------------------
void viz_feature( const cv::Mat &img, const Points2f &intersections, const std::vector<double> features,
                 cv::Mat &dst, const double multiplier = 255)
{
    dst = cv::Mat::zeros( img.size(), img.type());
    ISLOOP (intersections) {
        auto pf = intersections[i];
        double feat = features[i];
        auto hood = make_hood( pf, 5, 5);
        if (check_rect( hood, img.rows, img.cols)) {
            dst( hood) = fmin( 255, feat * multiplier);
        }
    }
} // viz_feature()

// Visualize some features
//---------------------------
- (UIImage *) f08_features
{
    g_app.mainVC.lbDbg.text = @"brightness";
    static int state = 0;
    std::vector<double> feats;
    cv::Mat drawing;
    
    switch (state) {
        case 0:
        {
            // Gray mean
            const int r = 4;
            const int yshift = 0;
            const bool dontscale = false;
            BlackWhiteEmpty::get_feature( _pyr_gray, _intersections_zoomed, r,
                                         [](const cv::Mat &hood) { return cv::mean(hood)[0]; },
                                         feats, yshift, dontscale);
            viz_feature( _pyr_gray, _intersections_zoomed, feats, drawing, 1);
            break;
        }
        default:
            state = 0;
            return NULL;
    } // switch
    state++;
    
    // Show results
    cv::cvtColor( drawing, drawing, cv::COLOR_GRAY2RGB);
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f08_features()

// Translate a bunch of points
//----------------------------------------------------------------
void translate_points( const Points2f &pts, int dx, int dy, Points2f &dst)
{
    dst = Points2f(SZ(pts));
    ISLOOP (pts) {
        dst[i] = Point2f( pts[i].x + dx, pts[i].y + dy);
    }
}

//------------------------------------------------------------------------------------------------------
std::vector<int> classify( const Points2f &intersections, const cv::Mat &img, const cv::Mat &gray,
                          int TIMEBUFSZ = 1)
{
    double match_quality;
    std::vector<int> diagram = BlackWhiteEmpty::classify( img, gray,
                                                         intersections, match_quality);
    // Vote across time
    static std::vector<std::vector<int> > timevotes(19*19);
    assert( SZ(diagram) <= 19*19);
    ISLOOP (diagram) {
        ringpush( timevotes[i], diagram[i], TIMEBUFSZ);
    }
    ISLOOP (timevotes) {
        std::vector<int> counts( DDONTKNOW, 0); // index is bwe
        for (int bwe: timevotes[i]) { ++counts[bwe]; }
        int winner = argmax( counts);
        diagram[i] = winner;
    }
    return diagram;
} // classify()

// Set any points to empty if outside of the image
//----------------------------------------------------------------------------------------------
void fix_diagram( std::vector<int> &diagram, const Points2f intersections, const cv::Mat &img)
{
    double marg = 10;
    ISLOOP (diagram) {
        Point2f p = intersections[i];
        if (p.x < marg || p.y < marg || p.x > img.cols - marg || p.y > img.rows - marg) {
            diagram[i] = EEMPTY;
        }
    }
} // fix_diagram()

// Classify intersections into black, white, empty
//-----------------------------------------------------------
- (UIImage *) f09_classify
{
    g_app.mainVC.lbDbg.text = @"classify";
    if (SZ(_corners_zoomed) != 4) { return MatToUIImage( _gray); }
    
    //std::vector<int> diagram;
    if (_small_zoomed.rows > 0) {
        //cv::Mat gray_blurred;
        //cv::GaussianBlur( _gray_zoomed, gray_blurred, cv::Size(5, 5), 2, 2 );
        const int TIME_BUF_SZ = 1;
        _diagram = classify( _intersections_zoomed, _pyr_zoomed, _gray_zoomed, TIME_BUF_SZ);
    }
    fix_diagram( _diagram, _intersections, _small_img);
    
    // Show results
    cv::Mat drawing;
    //DrawBoard drb( _gray_zoomed, _corners_zoomed[0].y, _corners_zoomed[0].x, _board_sz);
    //drb.draw( _diagram);
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    
    Points2f dummy;
    get_intersections_from_corners( _corners_zoomed, _board_sz, dummy, _dx, _dy);
    int dx = ROUND( _dx/4.0);
    int dy = ROUND( _dy/4.0);
    ISLOOP (_diagram) {
        cv::Point p(ROUND(_intersections_zoomed[i].x), ROUND(_intersections_zoomed[i].y));
        cv::Rect rect( p.x - dx,
                      p.y - dy,
                      2*dx + 1,
                      2*dy + 1);
        cv::rectangle( drawing, rect, cv::Scalar(0,0,255,255));
        if (_diagram[i] == BBLACK) {
            draw_point( p, drawing, 2, cv::Scalar(0,255,0,255));
        }
        else if (_diagram[i] == WWHITE) {
            draw_point( p, drawing, 5, cv::Scalar(255,0,0,255));
        }
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f09_classify()

// Save small crops around intersections for inspection
//-------------------------------------------------------------------------------
void save_intersections( const cv::Mat img,
                        const Points &intersections, int delta_v, int delta_h)
{
    ILOOP( intersections.size())
    {
        int x = intersections[i].x;
        int y = intersections[i].y;
        int dx = round(delta_h/2.0); int dy = round(delta_v/2.0);
        cv::Rect rect( x - dx, y - dy, 2*dx+1, 2*dy+1 );
        if (0 <= rect.x &&
            0 <= rect.width &&
            rect.x + rect.width <= img.cols &&
            0 <= rect.y &&
            0 <= rect.height &&
            rect.y + rect.height <= img.rows)
        {
            const cv::Mat &hood( img(rect));
            NSString *fname = nsprintf(@"hood_%03d.jpg",i);
            fname = getFullPath( fname);
            cv::imwrite( [fname UTF8String], hood);
        }
    } // ILOOP
} // save_intersections()

// Find all intersections from corners and boardsize
//--------------------------------------------------------------------------------
template <typename Points_>
void get_intersections_from_corners( const Points_ &corners, int boardsz, // in
                                    Points_ &result, double &delta_h, double &delta_v) // out
{
    if (corners.size() != 4) return;
    
    cv::Point2f tl = corners[0];
    cv::Point2f tr = corners[1];
    cv::Point2f br = corners[2];
    cv::Point2f bl = corners[3];
    
    std::vector<double> left_x;
    std::vector<double> left_y;
    std::vector<double> right_x;
    std::vector<double> right_y;
    ILOOP (boardsz) {
        left_x.push_back(  tl.x + i * (bl.x - tl.x) / (double)(boardsz-1));
        left_y.push_back(  tl.y + i * (bl.y - tl.y) / (double)(boardsz-1));
        right_x.push_back( tr.x + i * (br.x - tr.x) / (double)(boardsz-1));
        right_y.push_back( tr.y + i * (br.y - tr.y) / (double)(boardsz-1));
    }
    std::vector<double> top_x;
    std::vector<double> top_y;
    std::vector<double> bot_x;
    std::vector<double> bot_y;
    ILOOP (boardsz) {
        top_x.push_back( tl.x + i * (tr.x - tl.x) / (double)(boardsz-1));
        top_y.push_back( tl.y + i * (tr.y - tl.y) / (double)(boardsz-1));
        bot_x.push_back( bl.x + i * (br.x - bl.x) / (double)(boardsz-1));
        bot_y.push_back( bl.y + i * (br.y - bl.y) / (double)(boardsz-1));
    }
    delta_v = (bot_y[0] - top_y[0]) / (boardsz -1);
    delta_h = (right_x[0] - left_x[0]) / (boardsz -1);
    
    result = Points_();
    RLOOP (boardsz) {
        CLOOP (boardsz) {
            cv::Point2f p = intersection( cv::Point2f( left_x[r], left_y[r]), cv::Point2f( right_x[r], right_y[r]),
                                         cv::Point2f( top_x[c], top_y[c]), cv::Point2f( bot_x[c], bot_y[c]));
            result.push_back(p);
        }
    }
} // get_intersections_from_corners()

#pragma mark - Real time implementation
//========================================

// f00_*, f01_*, ... all in one go
//--------------------------------------------
- (UIImage *) real_time_flow:(UIImage *) img
{
    _board_sz = 19;
    cv::Mat drawing;
    
    bool success = false;
    do {
        static std::vector<Points> boards; // Some history for averaging
        UIImageToMat( img, _orig_img, false);
        resize( _orig_img, _small_img, 350);
        cv::cvtColor( _small_img, _small_img, CV_RGBA2RGB);
        cv::cvtColor( _small_img, _gray, cv::COLOR_RGB2GRAY);
        thresh_dilate( _gray, _gray_threshed);
        
        // Find stones and intersections
        _stone_or_empty.clear();
        BlobFinder::find_empty_places( _gray_threshed, _stone_or_empty); // has to be first
        BlobFinder::find_stones( _gray, _stone_or_empty);
        _stone_or_empty = BlobFinder::clean( _stone_or_empty);
        if (SZ(_stone_or_empty) < 0.8 * SQR(_board_sz)) break;
        
        // Break if not straight
        double theta = direction( _gray, _stone_or_empty) - PI/2;
        if (fabs(theta) > 4 * PI/180) break;
        
        // Find vertical lines
        _vertical_lines = homegrown_vert_lines( _stone_or_empty);
        std::vector<cv::Vec2f> all_vert_lines = _vertical_lines;
        dedup_verticals( _vertical_lines, _gray);
        filter_vert_lines( _vertical_lines);
        const int x_thresh = 4.0;
        fix_vertical_lines( _vertical_lines, all_vert_lines, _gray, x_thresh);
        if (SZ( _vertical_lines) > 55) break;
        if (SZ( _vertical_lines) < 5) break;
        
        // Find horiz lines
        _horizontal_lines = homegrown_horiz_lines( _stone_or_empty);
        dedup_horizontals( _horizontal_lines, _gray);
        filter_horiz_lines( _horizontal_lines);
        fix_horiz_lines( _horizontal_lines, _vertical_lines, _gray);
        if (SZ( _horizontal_lines) > 55) break;
        if (SZ( _horizontal_lines) < 5) break;
        
        // Find corners
        _intersections = get_intersections( _horizontal_lines, _vertical_lines);
        cv::pyrMeanShiftFiltering( _small_img, _small_pyr, SPATIALRAD, COLORRAD, MAXPYRLEVEL );
        _corners.clear();
        if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
            _corners = find_corners( _stone_or_empty, _horizontal_lines, _vertical_lines,
                                    _intersections, _small_pyr, _gray_threshed);
        }
        // Intersections for only the board lines
        _intersections = get_intersections( _horizontal_lines, _vertical_lines);
        if (!board_valid( _corners, _gray)) {
            break;
        }
        
        // Zoom in
        cv::Mat M;
        zoom_in( _gray,  _corners, _gray_zoomed, M);
        zoom_in( _small_pyr, _corners, _pyr_zoomed, M);
        cv::perspectiveTransform( _corners, _corners_zoomed, M);
        cv::perspectiveTransform( _intersections, _intersections_zoomed, M);
        fill_outside_with_average_gray( _gray_zoomed, _corners_zoomed);
        fill_outside_with_average_rgb( _pyr_zoomed, _corners_zoomed);
        
        // Classify
        const int TIME_BUF_SZ = 10;
        _diagram = classify( _intersections_zoomed, _pyr_zoomed, _gray_zoomed, TIME_BUF_SZ);
        fix_diagram( _diagram, _intersections, _small_img);
        success = true;
    } while(0);
    
    // Draw real time results on screen
    //------------------------------------
    cv::Mat *canvas;
    canvas = &_small_img;
    
    static std::vector<cv::Vec2f> old_hlines, old_vlines;
    static Points2f old_corners, old_intersections;
    if (!success) {
        _horizontal_lines = old_hlines;
        _vertical_lines = old_vlines;
        _corners = old_corners;
        _intersections = old_intersections;
    }
    else {
        old_hlines = _horizontal_lines;
        old_vlines = _vertical_lines;
        old_corners = _corners;
        old_intersections = _intersections;
    }
    
    if (SZ(_corners) == 4) {
        draw_line( cv::Vec4f( _corners[0].x, _corners[0].y, _corners[1].x, _corners[1].y),
                  *canvas, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[1].x, _corners[1].y, _corners[2].x, _corners[2].y),
                  *canvas, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[2].x, _corners[2].y, _corners[3].x, _corners[3].y),
                  *canvas, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[3].x, _corners[3].y, _corners[0].x, _corners[0].y),
                  *canvas, cv::Scalar( 255,0,0,255));
        
        // One horiz and vert line
        draw_polar_line( _horizontal_lines[SZ(_horizontal_lines)/2], *canvas, cv::Scalar( 255,255,0,255));
        draw_polar_line( _vertical_lines[SZ(_vertical_lines)/2], *canvas, cv::Scalar( 255,255,0,255));
        
        // Show classification result
        ISLOOP (_diagram) {
            cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
            if (_diagram[i] == BBLACK) {
                draw_point( p, *canvas, 5, cv::Scalar(255,0,0,255));
            }
            else if (_diagram[i] == WWHITE) {
                draw_point( p, *canvas, 5, cv::Scalar(0,255,0,255));
            }
        }
        ISLOOP (_intersections) {
            draw_point( _intersections[i], *canvas, 2, cv::Scalar(0,0,255,255));
        }
    } // if (SZ(corners) == 4)
    
    UIImage *res = MatToUIImage( *canvas);
    //UIImage *res = MatToUIImage( drawing);
    return res;
} // real_time_flow()


@end





























