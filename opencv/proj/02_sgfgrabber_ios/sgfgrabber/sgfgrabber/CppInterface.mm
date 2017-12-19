//
//  GrabFuncs.mm
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
#import "AppDelegate.h"
#import "Globals.h"
#import "CppInterface.h"
//#import "LineFinder.hpp"
//#import "LineFixer.hpp"
#import "BlackWhiteEmpty.hpp"
#import "BlobFinder.hpp"
#import "Clust1D.hpp"
#import "DrawBoard.hpp"
#import "Boardness.hpp"

// Pyramid filter params
#define SPATIALRAD  5
#define COLORRAD    30
//#define COLORRAD    15
#define MAXPYRLEVEL 2
//#define MAXPYRLEVEL 1

extern cv::Mat mat_dbg;

@interface CppInterface()
//=======================
@property cv::Mat small; // resized image, in color, RGB
@property cv::Mat small_pyr; // resized image, in color, pyramid filtered
@property Points pyr_board; // Initial guess at board location

@property cv::Mat small_zoomed;  // small, zoomed into the board
@property cv::Mat gray;  // Grayscale version of small
@property cv::Mat gray_threshed;  // gray with inv_thresh and dilation
@property cv::Mat gray_zoomed;   // Grayscale version of small, zoomed into the board

@property cv::Mat gz_threshed; // gray_zoomed with inv_thresh and dilation
@property cv::Mat m;     // Mat with image we are working on
@property Contours cont; // Current set of contours
@property int board_sz; // board size, 9 or 19
@property Points stone_or_empty; // places where we suspect stones or empty
@property std::vector<cv::Vec2f> horizontal_lines;
@property std::vector<cv::Vec2f> vertical_lines;
@property Points2f corners;
@property Points2f corners_zoomed;
@property Points2f intersections;
@property Points2f intersections_zoomed;
@property float dy;
@property float dx;
//@property LineFinder finder;
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
    float screenArea = img.rows * img.cols;
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

// Apply inverse thresh and dilate grayscale image.
//---------------------------------------------------------
void thresh_dilate( const cv::Mat &img, cv::Mat &dst, int thresh = 8)
{
    cv::adaptiveThreshold( img, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          5 /* 11 */ ,  // neighborhood_size
                          thresh);  // threshold
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate( dst, dst, element );
}

#pragma mark - Processing Pipeline for debugging
//=================================================

//-----------------------------------------
- (UIImage *) f00_blobs:(UIImage *)img
{
    _board_sz=19;
    g_app.mainVC.lbDbg.text = @"00";
    
    // Live Camera
    UIImageToMat( img, _m);
    
    // From file
    //load_img( @"board06.jpg", _m);
    //cv::rotate(_m, _m, cv::ROTATE_90_CLOCKWISE);

    resize( _m, _small, 350);
    cv::cvtColor( _small, _small, CV_RGBA2RGB); // Yes, RGB not BGR
    cv::cvtColor( _small, _gray, cv::COLOR_RGB2GRAY);
    thresh_dilate( _gray, _gray_threshed);
    cv::pyrMeanShiftFiltering( _small, _small_pyr, SPATIALRAD, COLORRAD, MAXPYRLEVEL );

    _stone_or_empty.clear();
    BlobFinder::find_empty_places( _gray_threshed, _stone_or_empty); // has to be first
    BlobFinder::find_stones( _gray, _stone_or_empty);
    _stone_or_empty = BlobFinder::clean( _stone_or_empty);
    
    // Show results
    cv::Mat drawing = _small_pyr.clone();
    draw_points( _stone_or_empty, drawing, 2, cv::Scalar( 255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Find closest line among a bunch of roughly horizontal lines,
// using distance at x == width/2
//---------------------------------------------------------------------------------------------
cv::Vec2f closest_hline( const std::vector<cv::Vec2f> &lines, const cv::Vec2f line, int width,
                        float &dy) // out
{
    float middle_x = width / 2.0;
    float y_0 = y_from_x( middle_x, line);

    dy = 1E9;
    cv::Vec2f res;
    ISLOOP( lines) {
        float y_line = y_from_x( middle_x, lines[i]);
        float d = abs( y_0 - y_line);
        if (d < dy) {
            dy = d;
            res = lines[i];
        }
    }
    return res;
}

//------------------------------------------------------------------------
void find_horiz_lines( cv::Vec2f ratline, float dy, float dy_rat,
                      const std::vector<cv::Vec2f> &h_lines, int boardsz, int width,
                      std::vector<cv::Vec2f> &fixed_lines) // out
{
    //fixed_lines = h_lines; return;
    
    float THRESH_RAT = 1/4.0;
    float THRESH_THETA = PI/180;
    cv::Vec2f cur_line;
    float cur_dy, d;
    
    // above ratline
    //PLOG(">>>>>above\n");
    cur_line = ratline;
    cur_dy = dy;
    ILOOP (boardsz) {
        cur_line[0] -= cur_dy;
        cv::Vec2f nbor = closest_hline( h_lines, cur_line, width, d);
        float dtheta = fabs( nbor[1] - cur_line[1]);
        //PLOG( "d:%.4f\n",d);
        if (d < THRESH_RAT * cur_dy && dtheta < THRESH_THETA ) {
            //PLOG( "fixed\n");
            cur_line = nbor;
        }
        cur_dy /= dy_rat;
        fixed_lines.push_back( cur_line);
    }

    // ratline and below
    //PLOG(">>>>>below\n");
    cur_line = ratline;
    cur_dy = dy;
    ILOOP (boardsz) {
        cv::Vec2f nbor = closest_hline( h_lines, cur_line, width, d);
        float dtheta = fabs( nbor[1] - cur_line[1]);
        //PLOG( "cur_dy: %.4f d:%.4f dtheta:%.4f\n", cur_dy, d, dtheta);
        if (d < THRESH_RAT * cur_dy && dtheta < THRESH_THETA  ) {
            //PLOG( "fixed\n");
            cur_line = nbor;
        }
        fixed_lines.push_back( cur_line);
        cur_dy *= dy_rat;
        cur_line[0] += cur_dy;
    }
    // Sort by rho (aka y)
    std::sort( fixed_lines.begin(), fixed_lines.end(),
              [](cv::Vec2f line1, cv::Vec2f line2) { return line1[0] < line2[0]; });
} // find_horiz_lines()

//-----------------------------
- (UIImage *) f02_horiz_lines
{
    g_app.mainVC.lbDbg.text = @"02";
    
    _horizontal_lines = homegrown_horiz_lines( _stone_or_empty);
    dedup_horiz_lines( _horizontal_lines, _gray);
    fix_horiz_lines( _horizontal_lines, _gray);
    
    // Show results
    cv::Mat drawing;
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

// Thin down vertical Hough lines
//----------------------------------
- (UIImage *) f03_vert_lines
{
    g_app.mainVC.lbDbg.text = @"03";
    _vertical_lines = homegrown_vert_lines( _stone_or_empty);
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color(true);
    ISLOOP( _vertical_lines) {
        //if (i<400) continue;
        draw_polar_line( _vertical_lines[i], drawing, cv::Scalar(255,0,0));
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f03_vert_lines()

// Replace close clusters of vert lines by their average.
//-----------------------------------------------------------------------------------
void dedup_vertical_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    // Cluster by x in the middle
    const float wwidth = 32.0;
    const float middle_y = img.rows / 2.0;
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
void dedup_horiz_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    // Cluster by y in the middle
    const float wwidth = 32.0;
    const float middle_x = img.cols / 2.0;
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

// Cluster vertical Hough lines to remove close duplicates.
//------------------------------------------------------------
- (UIImage *) f04_vert_lines_2
{
    g_app.mainVC.lbDbg.text = @"04";
    dedup_vertical_lines( _vertical_lines, _gray);
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color(true);
    ISLOOP( _vertical_lines) {
        draw_polar_line( _vertical_lines[i], drawing, get_color());
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Find the change per line in rho and theta and synthesize the whole bunch
// starting at the middle. Replace synthesized lines with real ones if close enough.
//----------------------------------------------------------------------------------
void fix_vertical_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    const float middle_y = img.rows / 2.0;
    const float width = img.cols;
    
    auto rhos   = vec_extract( lines, [](cv::Vec2f line) { return line[0]; } );
    auto thetas = vec_extract( lines, [](cv::Vec2f line) { return line[1]; } );
    auto xes    = vec_extract( lines, [middle_y](cv::Vec2f line) { return x_from_y( middle_y, line); });
    auto d_rhos   = vec_delta( rhos);
    auto d_thetas = vec_delta( thetas);
    auto d_rho   = vec_median( d_rhos);
    auto d_theta = vec_median( d_thetas);
    
    cv::Vec2f med_line = lines[SZ(lines)/2];
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    float rho, theta;
    // If there is a close line, use it. Else interpolate.
    const float X_THRESH = 3; //6;
    const float THETA_THRESH = PI / 180;
    // Lines to the right
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        if (!i) continue;
        rho += d_rho;
        theta += d_theta;
        float x = x_from_y( middle_y, cv::Vec2f( rho, theta));
        int close_idx = vec_closest( xes, x);
        if (fabs( x - xes[close_idx]) < X_THRESH &&
            fabs( theta - thetas[close_idx]) < THETA_THRESH)
        {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
        }
        if (rho == 0) break;
        cv::Vec2f line( rho,theta);
        if (x_from_y( middle_y, line) > width) break;
        synth_lines.push_back( line);
    }
    // Lines to the left
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        if (!i) continue;
        rho -= d_rho;
        theta -= d_theta;
        float x = x_from_y( middle_y, cv::Vec2f( rho, theta));
        int close_idx = vec_closest( xes, x);
        if (fabs( x - xes[close_idx]) < X_THRESH &&
            fabs( theta - thetas[close_idx]) < THETA_THRESH)
        {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
        }
        if (rho == 0) break;
        cv::Vec2f line( rho,theta);
        if (x_from_y( middle_y, line) < 0) break;
        synth_lines.push_back( line);
    }
    std::sort( synth_lines.begin(), synth_lines.end(),
              [middle_y](cv::Vec2f line1, cv::Vec2f line2) {
                  return x_from_y( middle_y, line1) < x_from_y( middle_y, line2);
              });
    
    lines = synth_lines;
} // fix_vertical_lines()

// Find the change per line in rho and theta and synthesize the whole bunch
// starting at the middle. Replace synthesized lines with real ones if close enough.
//----------------------------------------------------------------------------------
void fix_horiz_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
{
    const float middle_x = img.rows / 2.0;
    const float height = img.rows;
    
    auto rhos   = vec_extract( lines, [](cv::Vec2f line) { return line[0]; } );
    auto thetas = vec_extract( lines, [](cv::Vec2f line) { return line[1]; } );
    auto ys     = vec_extract( lines, [middle_x](cv::Vec2f line) { return y_from_x( middle_x, line); });
    auto d_rhos   = vec_delta( rhos);
    auto d_thetas = vec_delta( thetas);
    auto d_rho   = vec_median( d_rhos);
    auto d_theta = vec_median( d_thetas);
    
    cv::Vec2f med_line = lines[SZ(lines)/2];
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    float rho, theta;
    // If there is a close line, use it. Else interpolate.
    const float Y_THRESH = 3; // 6;
    const float THETA_THRESH = PI / 180;
    // Lines below
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        if (!i) continue;
        rho += d_rho;
        theta += d_theta;
        float y = y_from_x( middle_x, cv::Vec2f( rho, theta));
        int close_idx = vec_closest( ys, y);
        if (fabs( y - ys[close_idx]) < Y_THRESH &&
            fabs( theta - thetas[close_idx]) < THETA_THRESH)
        {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
        }
        if (rho == 0) break;
        cv::Vec2f line( rho,theta);
        if (y_from_x( middle_x, line) > height) break;
        synth_lines.push_back( line);
    }
    // Lines above
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        if (!i) continue;
        rho -= d_rho;
        theta -= d_theta;
        float y = y_from_x( middle_x, cv::Vec2f( rho, theta));
        int close_idx = vec_closest( ys, y);
        if (fabs( y - ys[close_idx]) < Y_THRESH &&
            fabs( theta - thetas[close_idx]) < THETA_THRESH)
        {
            rho   = lines[close_idx][0];
            theta = lines[close_idx][1];
        }
        if (rho == 0) break;
        cv::Vec2f line( rho,theta);
        if (y_from_x( middle_x, line) < 0) break;
        synth_lines.push_back( line);
    }
    std::sort( synth_lines.begin(), synth_lines.end(),
              [middle_x](cv::Vec2f line1, cv::Vec2f line2) {
                  return y_from_x( middle_x, line1) < y_from_x( middle_x, line2);
              });
    lines = synth_lines;
} // fix_horiz_lines()


// Find vertical line parameters
//---------------------------------
- (UIImage *) f05_vert_params
{
    g_app.mainVC.lbDbg.text = @"05";
    fix_vertical_lines( _vertical_lines, _gray);
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color(true);
    ISLOOP( _vertical_lines) {
        draw_polar_line( _vertical_lines[i], drawing, get_color());
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f05_vert_params()

// Use y at the center to sum feat at points between two horizontal lines.
//---------------------------------------------------------------------------------
float sum_points_between_horiz_lines( cv::Vec2f top_line, cv::Vec2f bot_line,
                                   const std::vector<PFeat> &pfts, int middle_x)
{
    float res = 0;
    const float EPS = 5;
    for (auto pft: pfts) {
        float tdist = dist_point_line( pft.p, top_line);
        float bdist = dist_point_line( pft.p, bot_line);
        if ((tdist > 0 || fabs( tdist) < EPS) && (bdist < 0 || fabs( bdist) < EPS) ) {
            res += pft.feat;
        }
    }
    return res;
} // sum_points_between_horiz_lines()

// Use y at the center to sum feat at points between two vertical lines.
//---------------------------------------------------------------------------------
float sum_points_between_vert_lines( cv::Vec2f left_line, cv::Vec2f right_line,
                                    const std::vector<PFeat> &pfts, int middle_y)
{
    float res = 0;
    const float EPS = 5;
    for (auto pft: pfts) {
        float ldist = dist_point_line( pft.p, left_line);
        float rdist = dist_point_line( pft.p, right_line);
        if ((ldist > 0 || fabs( ldist) < EPS) && (rdist < 0 || fabs( rdist) < EPS) ) {
            res += pft.feat;
        }
    }
    return res;
} // sum_points_between_vert_lines()

// Sum feat at points enclosed by four lines
//---------------------------------------------------------------------------------
float sum_points_between_four_lines( cv::Vec2f top_line, cv::Vec2f bot_line,
                                    cv::Vec2f left_line, cv::Vec2f right_line,
                                    const std::vector<PFeat> &pfts,
                                    int midle_x, int middle_y)
{
    float res = 0;
    const float EPS = 5;
    for (auto pft: pfts) {
        float ldist = dist_point_line( pft.p, left_line);
        float rdist = dist_point_line( pft.p, right_line);
        float tdist = dist_point_line( pft.p, top_line);
        float bdist = dist_point_line( pft.p, bot_line);
        if ((ldist > 0 || fabs( ldist) < EPS) && (rdist < 0 || fabs( rdist) < EPS) ) {
            if ((tdist > 0 || fabs( tdist) < EPS) && (bdist < 0 || fabs( bdist) < EPS) ) {
                res += pft.feat;
            }
        }
    }
    return res;
} // sum_points_between_four_lines()



//--------------------------------------------------------
int count_points_on_line( cv::Vec2f line, Points pts)
{
    int res = 0;
    for (auto p:pts) {
        float d = fabs(dist_point_line( p, line));
        if (d < 0.75) {
            res++;
        }
    }
    return res;
}

//------------------------------------------------------------------------------------
int pixels_on_segment( const cv::Mat &threshed, cv::Point p1, cv::Point p2, int width=1)
{
    cv::Mat mask( threshed.rows, threshed.cols, CV_8UC1);
    cv::line( mask, p1, p2, cv::Scalar(1), width);
    cv::bitwise_and( threshed, mask, mask);
    int res = cv::sum(mask)[0];
    return res;
} // pixels_on_segment()

// Find a vertical line thru pt which hits a lot of other points
// PRECONDITION: allpoints must be sorted by y
//------------------------------------------------------------------
cv::Vec2f find_vert_line_thru_point( const Points &allpoints, cv::Point pt)
{
    // Find next point below.
    //const float RHO_EPS = 10;
    const float THETA_EPS = 10 * PI / 180;
    int maxhits = -1;
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
    //const float RHO_EPS = 10;
    const float THETA_EPS = 5 * PI / 180;
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
        cv::Vec2f newline = find_vert_line_thru_point( pts, tp);
        if (newline[0] != 0) {
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

// Color after pyrMeanShiftFiltering
//-------------------------------------------------------------------------------------
void floodFillPostprocess( cv::Mat& img, const cv::Scalar& colorDiff=cv::Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    cv::RNG rng = cv::theRNG();
    cv::Mat mask( img.rows+2, img.cols+2, CV_8UC1, cv::Scalar::all(0) );
    get_color(true);
    for( int y = 0; y < img.rows; y++ ) {
        for( int x = 0; x < img.cols; x++ ) {
            if( mask.at<uchar>(y+1, x+1) == 0 ) {
                cv::Scalar color( rng(256), rng(256), rng(256) );
                //auto color = get_color();
                floodFill( img, mask, cv::Point(x,y), color, 0, colorDiff, colorDiff );
            }
        }
    }
} // floodFillPostprocess()

// For each pixel, sum diff to neighbors, unless they are 0 or you are 0.
// Scale result into 0-255 range.
//------------------------------------------------------------------------
void neighbor_diffs( const cv::Mat &img, cv::Mat &dst)
{
    dst = cv::Mat( img.rows, img.cols, CV_8UC1);
    float mmax = -1E9;
    RLOOP (img.rows) {
        int topr = r - 1;
        int botr = r + 1;
        CLOOP (img.cols) {
            int leftc = c - 1;
            int rightc = c + 1;
            int v = img.at<uint8_t>(r,c);
            int pn = 0; float psum = 0;
            if (v) {
                int left=0, right=0, top=0, bot=0;
                if (leftc >= 0)        { left  = img.at<uint8_t>(r, leftc); }
                if (rightc < img.cols) { right = img.at<uint8_t>(r, rightc); }
                if (topr >= 0)         { top   = img.at<uint8_t>(topr, c); }
                if (botr < img.rows)   { bot   = img.at<uint8_t>(botr, c); }
                
                if (left)  { psum += fabs( left - v);  pn++; }
                if (right) { psum += fabs( right - v);  pn++; }
                if (top)   { psum += fabs( top - v);  pn++; }
                if (bot)   { psum += fabs( bot - v);  pn++; }
                psum = RAT( psum, pn);
            }
            dst.at<uint8_t>(r,c) = psum;
            if (psum > mmax) mmax = psum;
        } // CLOOP
    } // RLOOP
    // Scale to 255
    dst.convertTo( dst, CV_8UC1, 255.0 / mmax , 0);
    // Eliminate suspiciously large ones
    //dst.forEach<uint8_t>( [](uint8_t &v, const int *p) { if (v > 200) v = 0; } );
    // Scale again
    //dst.convertTo( dst, CV_8UC1, 255.0 / mmax , 0);
}

// Color difference between first line and just outside
//------------------------------------------------------------------------
float outer_color_diff( const cv::Mat &m, int r_, int c_, int board_sz)
{
    float ssum = 0;
    if (!p_on_img(cv::Point( c_ - 1,        r_ ), m)) return -1;
    if (!p_on_img(cv::Point( c_ + board_sz, r_ ), m)) return -1;
    if (!p_on_img(cv::Point( c_,            r_ - 1), m)) return -1;
    if (!p_on_img(cv::Point( c_,            r_ + board_sz), m)) return -1;
    
    // Left and right edge
    for (int r = r_; r < r_ + board_sz; r++) {
        auto b_l = m.at<cv::Vec3b>(r, c_); // on the board
        auto b_r = m.at<cv::Vec3b>(r, c_ + board_sz -1);
        auto o_l = m.at<cv::Vec3b>(r, c_ - 1); // just outside the board
        auto o_r = m.at<cv::Vec3b>(r, c_ + board_sz);
        ssum += cv::norm( b_l - o_l);
        ssum += cv::norm( b_r - o_r);
    }
    // Top and bottom edge
    for (int c = c_; c < c_ + board_sz; c++) {
        auto b_t = m.at<cv::Vec3b>(r_, c); // on the board
        auto b_b = m.at<cv::Vec3b>(r_ + board_sz - 1, c);
        auto o_t = m.at<cv::Vec3b>(r_ - 1, c); // just outside the board
        auto o_b = m.at<cv::Vec3b>(r_ + board_sz, c);
        ssum += cv::norm( b_t - o_t);
        ssum += cv::norm( b_b - o_b);
    }
    return ssum;
} // outer_diff()

// Replace anything inside the board with the median.
//-----------------------------------------------------------------
void medianize_board( cv::Mat &m, int r_, int c_, int board_sz)
{
    std::vector<int> rv, gv, bv;
    
    for (int r = r_ ; r < r_ + board_sz; r++) {
        for (int c = c_; c < c_ + board_sz; c++) {
            auto v = m.at<cv::Vec3b>(r, c);
            rv.push_back( v[0]);
            gv.push_back( v[1]);
            bv.push_back( v[2]);
        } // for c
    } // for r
    int rmed = vec_median( rv);
    int gmed = vec_median( gv);
    int bmed = vec_median( bv);
    cv::Vec3b med(rmed,gmed,bmed);
    for (int r = r_ ; r < r_ + board_sz; r++) {
        for (int c = c_ ; c < c_ + board_sz; c++) {
            m.at<cv::Vec3b>(r, c) = med;
        } // for c
    } // for r
} // medianize_board()


// A good board has only B,W,E
//------------------------------------------------
float boardness( const cv::Mat &m)
{
    std::vector<int> rv, gv, bv, grayv;
    std::vector<cv::Vec3b> pixels;
    
    RLOOP (m.rows) {
        CLOOP (m.cols) {
            auto v = m.at<cv::Vec3b>(r, c);
            rv.push_back( v[0]);
            gv.push_back( v[1]);
            bv.push_back( v[2]);
            grayv.push_back( sqrt( SQR(v[0])+SQR(v[1])+SQR(v[2])));
            pixels.push_back( v);
        } // CLOOP
    } // RLOOP
    
    float maxr = vec_max( rv);
    float maxg = vec_max( gv);
    float maxb = vec_max( bv);
    cv::Vec3b mmax( maxr, maxg, maxb);
    
    float minr = vec_min( rv);
    float ming = vec_min( gv);
    float minb = vec_min( bv);
    cv::Vec3b mmin( minr, ming, minb);

    float avgr = vec_avg( rv);
    float avgg = vec_avg( gv);
    float avgb = vec_avg( bv);
    cv::Vec3b aavg( avgr, avgg, avgb);
    
    auto white_pixels = pixels;
    vec_filter( white_pixels, [mmin,mmax,aavg](cv::Vec3b p) {
        return cv::norm( p,mmax) < cv::norm( p,mmin)
        && cv::norm( p,mmax) < cv::norm( p,aavg);
    });
    
    auto black_pixels = pixels;
    vec_filter( black_pixels, [mmin,mmax,aavg](cv::Vec3b p) {
        //float d3 = cv::norm( p,mmin);
        //float d4 = cv::norm( p,mmax);
        return cv::norm( p,mmin) < cv::norm( p,mmax)
        && cv::norm( p,mmin) < cv::norm( p,aavg);
    });
    
    auto empty_pixels = pixels;
    vec_filter( empty_pixels, [mmin,mmax,aavg](cv::Vec3b p) {
        return cv::norm( p,aavg) < cv::norm( p,mmin)
        && cv::norm( p,aavg) < cv::norm( p,mmax);
    });
    
    float res = 0;
    for (auto x:white_pixels) { res += cv::norm( x,mmax); }
    for (auto x:black_pixels) { res += cv::norm( x,mmin); }
    for (auto x:empty_pixels) { res += cv::norm( x,aavg); }

    // Debug
    mat_dbg = cv::Mat( m.rows, m.cols, CV_8UC3);
    RLOOP (m.rows) {
        CLOOP (m.cols) {
            auto p = m.at<cv::Vec3b>(r,c);
            if (cv::norm( p,mmax) < cv::norm( p,mmin)
                && cv::norm( p,mmax) < cv::norm( p,aavg))
            {
                mat_dbg.at<cv::Vec3b>(r,c) = cv::Vec3b( 255,0,0);
            }
            else if (cv::norm( p,mmin) < cv::norm( p,mmax)
                && cv::norm( p,mmin) < cv::norm( p,aavg))
            {
                mat_dbg.at<cv::Vec3b>(r,c) = cv::Vec3b( 0,255,0);
            }
            else if (cv::norm( p,aavg) < cv::norm( p,mmin)
                && cv::norm( p,aavg) < cv::norm( p,mmax))
            {
                mat_dbg.at<cv::Vec3b>(r,c) = cv::Vec3b( 0,0,255);
            }
        } // CLOOP
    } // RLOOP
    return res;
} // boardness()

// Use horizontal and vertical lines to find corners such that the board best matches the points we found
//-----------------------------------------------------------------------------------------------------------
Points2f find_corners( const Points blobs, std::vector<cv::Vec2f> &horiz_lines, std::vector<cv::Vec2f> &vert_lines, 
                     const Points2f &intersections, const cv::Mat &img, const cv::Mat &threshed, int board_sz = 19) //@@@
{
    if (SZ(horiz_lines) < 3 || SZ(vert_lines) < 3) return Points2f();
    
    Boardness bness( intersections, blobs, img, board_sz, horiz_lines, vert_lines);
    cv::Mat &edgeness = bness.edgeness();
    cv::Mat &blobness = bness.blobness();

    // Mark corners for visualization
    //aux.at<cv::Vec3b>(cv::Point( cmax, rmax)) = cv::Vec3b( 255,0,0);
    //aux.at<cv::Vec3b>(cv::Point( cmax+board_sz-1, rmax+board_sz-1)) = cv::Vec3b( 255,0,0);
    cv::resize(blobness, mat_dbg, img.size(), 0,0, CV_INTER_NN);
    //cv::resize(aux, aux, img.size(), 0,0, CV_INTER_NN);
    //mat_dbg = aux.clone();
    //cv::cvtColor( auxgray, mat_dbg, cv::COLOR_GRAY2RGB);
//    for (auto &pft: crosses) {
//        draw_point( pft.p, mat_dbg, 4, cm_penny_lane(pft.feat) );
//    }
    
//    int tt = 42;
//
//
//
//    int minr = -1, minc = -1;
//    double mindist = 1E9;
//    std::vector<int> vals;
//    RLOOP (auxgray.rows) {
//        if (r + board_sz > auxgray.rows) break;
//        CLOOP (auxgray.cols) {
//            if (c + board_sz > auxgray.cols) break;
//            cv::Rect rect( c, r, board_sz, board_sz);
//            cv::Mat tmp = auxgray( rect).clone();
//            vals.assign( tmp.begin<uint8_t>(), tmp.end<uint8_t>());
//            double compactness;
//            int tries=3, iter=10, eps=1.0;
//            auto clusters = cluster(vals, 3, [](int v) { return float(v); }, compactness, tries, iter, eps);
//            //PLOG( "r c compactness %5d %5d %.0f\n", r, c, compactness);
//            if (compactness < mindist) {
//                mindist = compactness;
//                minc = c;
//                minr = r;
//            }
//        } // CLOOP
//    } // RLOOP
//    if (minr < 0 || minc < 0) return Points2f();
//    // Return the board lines only
//    horiz_lines = vec_slice( horiz_lines, minr, board_sz);
//    vert_lines  = vec_slice( vert_lines, minc, board_sz);
//
//    // Mark corners for visualization
//    //aux.at<cv::Vec3b>(cv::Point( minc, minr)) = cv::Vec3b( 255,0,0);
//    //aux.at<cv::Vec3b>(cv::Point( minc+board_sz-1, minr+board_sz-1)) = cv::Vec3b( 255,0,0);
//
//
//    auto rc2p = [&intersections, &auxgray](int r, int c) { return intersections[r*auxgray.cols + c]; };
//    auto tl = rc2p( minr, minc);
//    auto tr = rc2p( minr, minc+board_sz-1);
//    auto br = rc2p( minr+board_sz-1, minc+board_sz-1);
//    auto bl = rc2p( minr+board_sz-1, minc);
//    Points2f corners = { tl, tr, br, bl };
//
//    //cv::resize(auxgray, auxgray, img.size(), 0,0, CV_INTER_NN);
//    //cv::resize(aux, aux, img.size(), 0,0, CV_INTER_NN);
    
    //aux.copyTo(mat_dbg);
    Points2f corners;
    return corners;
} // find_corners()

//// Use horizontal and vertical lines to find corners such that the board best matches the points we found
////-----------------------------------------------------------------------------------------------------------
//Points2f get_corners( const std::vector<cv::Vec2f> &horiz_lines, const std::vector<cv::Vec2f> &vert_lines,
//                     const cv::Mat &gray,  const cv::Mat &threshed, int board_sz = 19)
//{
//    int max_idx = -1;
//    float max_score= -1E9;
//    cv::Vec2f top_line, bot_line, left_line, right_line;
//
//    for (int r=0; r < SZ(horiz_lines) - board_sz + 1; r++) {
//        std::vector<cv::Vec2f> hlines = vec_slice( horiz_lines, r, board_sz);
//        for (int c=0; c < SZ(vert_lines) - board_sz + 1; c++) {
//            std::vector<cv::Vec2f> vlines = vec_slice( vert_lines, c, board_sz);
//            auto intersections = get_intersections( hlines, vlines);
//            float score;
//            BlackWhiteEmpty::classify( gray, threshed, intersections, score);
//            PLOG( "Score: %.0f\n", score);
//            if (score > max_score) {
//                max_score = score;
//                top_line = hlines.front(); bot_line = hlines.back();
//                left_line = vlines.front(); right_line = vlines.back();
//            }
//        }
//    }
//    int tt = 42;
//
//    Point2f tl = intersection( left_line,  top_line);
//    Point2f tr = intersection( right_line, top_line);
//    Point2f br = intersection( right_line, bot_line);
//    Point2f bl = intersection( left_line,  bot_line);
//    Points2f corners = {tl, tr, br, bl};
//    return corners;
//} // get_corners()

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

// Look at each intersection and give a guess whether you think it
// is an empty intersection. This helps us find the location of the board.
// The img is binary, the result of threshold and dilate.
// ------------------------------------------------------------------------
std::vector<PFeat> find_crosses( const cv::Mat &threshed,
                                const Points2f &intersections)
{
    std::vector<float> features, inner;
    std::vector<PFeat> res;
    int r=10;
    BlackWhiteEmpty::get_feature( threshed, intersections, r, BlackWhiteEmpty::cross_feature,
                                 features, 0, true);
    
    ISLOOP (features) {
        res.push_back( { intersections[i], features[i] } );
    }
    return res;
} // find_crosses()

// Find the corners
//----------------------------
- (UIImage *) f06_corners
{
    g_app.mainVC.lbDbg.text = @"06";

    auto intersections = get_intersections( _horizontal_lines, _vertical_lines);
    //auto crosses = find_crosses( _gray_threshed, intersections);
    _corners.clear();
    do {
        if (SZ( _horizontal_lines) > 45) break;
        if (SZ( _horizontal_lines) < 5) break;
        if (SZ( _vertical_lines) > 35) break;
        if (SZ( _vertical_lines) < 5) break;
        _corners = find_corners( _stone_or_empty, _horizontal_lines, _vertical_lines,
                                intersections, _small_pyr, _gray_threshed );
    } while(0);
    
    // Show results
    //cv::Mat drawing = _small_pyr.clone();
    //cv::Mat drawing; cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    //mat_dbg.convertTo( mat_dbg, CV_8UC1);
    //cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    //float alpha = 0.5;
    //cv::addWeighted( _small, alpha, drawing, 1-alpha, 0, drawing);
    //draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( mat_dbg);
    return res;
} // f06_corners()


//// Find the corners
////----------------------------
//- (UIImage *) f06_corners
//{
//    g_app.mainVC.lbDbg.text = @"06";
//
//    auto intersections = get_intersections( _horizontal_lines, _vertical_lines);
//    auto crosses = find_crosses( _gray_threshed, intersections);
//    _corners.clear();
//    if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
//        _corners = get_corners( _horizontal_lines, _vertical_lines, _gray, _gray_threshed);
//    }
//    
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray_threshed, drawing, cv::COLOR_GRAY2RGB);
//    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//} // f06_corners()

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

// Zoom in
//----------------------------
- (UIImage *) f07_zoom_in
{
    g_app.mainVC.lbDbg.text = @"07";
    cv::Mat threshed;
    cv::Mat dst;
    if (SZ(_corners) == 4) {
        cv::Mat M;
        zoom_in( _gray,  _corners, _gray_zoomed, M);
        zoom_in( _small, _corners, _small_zoomed, M);
        cv::perspectiveTransform( _corners, _corners_zoomed, M);
        thresh_dilate( _gray_zoomed, _gz_threshed, 4);
    }
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Repeat whole process 01 to 06 on the zoomed in version
//-----------------------------------------------------------
- (UIImage *) f08_show_threshed
{
    g_app.mainVC.lbDbg.text = @"08";
    _corners = _corners_zoomed;
    
    // Show results
    cv::Mat drawing;
//    int s = 2*BlackWhiteEmpty::RING_R+1;
//    cv::Rect re( 100, 100, s, s);
//    BlackWhiteEmpty::ringmask().copyTo( _gz_threshed( re));
    cv::cvtColor( _gz_threshed, drawing, cv::COLOR_GRAY2RGB);
    //cv::cvtColor( _hue_zoomed, drawing, cv::COLOR_GRAY2RGB);
    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f08_repeat_on_zoomed()

// Intersections on zoomed from corners
//----------------------------------------
- (UIImage *) f09_intersections
{
    g_app.mainVC.lbDbg.text = @"09";
    
    if (SZ(_corners_zoomed) == 4) {
        get_intersections_from_corners( _corners_zoomed, _board_sz, _intersections_zoomed, _dx, _dy);
    }
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    draw_points( _intersections_zoomed, drawing, 1, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f09_intersections()

// Visualize features, one per intersection.
//------------------------------------------------------------------------------------------------------
void viz_feature( const cv::Mat &img, const Points2f &intersections, const std::vector<float> features,
                 cv::Mat &dst, const float multiplier = 255)
{
    dst = cv::Mat::zeros( img.size(), img.type());
    ISLOOP (intersections) {
        auto pf = intersections[i];
        float feat = features[i];
        auto hood = make_hood( pf, 5, 5);
        if (check_rect( hood, img.rows, img.cols)) {
            dst( hood) = fmin( 255, feat * multiplier);
        }
    }
} // viz_feature()

// Visualize some features
//---------------------------
- (UIImage *) f10_features
{
    g_app.mainVC.lbDbg.text = @"10";
    static int state = 0;
    std::vector<float> feats;
    cv::Mat drawing;
    int r;

    switch (state) {
        case 0:
        {
            r=11;
            BlackWhiteEmpty::get_feature( _gz_threshed, _intersections_zoomed, r,
                                         BlackWhiteEmpty::sum_feature, feats);
            viz_feature( _gz_threshed, _intersections_zoomed, feats, drawing, 1);
            break;
        }
        default:
            state = 0;
            return NULL;
    } // switch
    state++;
    
    // Show results
    cv::cvtColor( drawing, drawing, cv::COLOR_GRAY2RGB);
    //cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f10_features()

// Translate a bunch of points
//----------------------------------------------------------------
Points2f translate_points( const Points2f &pts, int dx, int dy)
{
    Points2f res(SZ(pts));
    ISLOOP (pts) {
        res[i] = Point2f( pts[i].x + dx, pts[i].y + dy);
    }
    return res;
}

//------------------------------------------------------------------------------------------------------
std::vector<int> classify( const Points2f &intersections_, const cv::Mat &img, const cv::Mat &threshed,
                          float dx, float dy,
                          int TIMEBUFSZ = 1)
{
    Points2f intersections;
    std::vector<std::vector<int> > diagrams;
    // Wiggle the regions a little.
    intersections = translate_points( intersections_, 0, 0);
    float match_quality;
    diagrams.push_back( BlackWhiteEmpty::classify( img, threshed,
                                                  intersections, match_quality));
    //    intersections = translate_points( intersections_, -1, 0);
    //    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
    //                                                  intersections, match_quality));
    
    // Vote across wiggle
    std::vector<int> diagram; // vote result
    ISLOOP( diagrams[0]) {
        std::vector<int> votes(4,0);
        for (auto d:diagrams) {
            int idx = d[i];
            votes[idx]++;
        }
        int winner = argmax( votes);
        diagram.push_back( winner);
    }
    // Vote across time
    static std::vector<std::vector<int> > timevotes(19*19);
    ISLOOP (diagram) {
        ringpush( timevotes[i], diagram[i], TIMEBUFSZ);
    }
    ISLOOP (timevotes) {
        std::vector<int> counts( BlackWhiteEmpty::DONTKNOW, 0); // index is bwe
        for (int bwe: timevotes[i]) { ++counts[bwe]; }
        int winner = argmax( counts);
        diagram[i] = winner;
    }
    
    return diagram;
} // classify()

// Classify intersections into black, white, empty
//-----------------------------------------------------------
- (UIImage *) f11_classify
{
    g_app.mainVC.lbDbg.text = @"11";
    if (SZ(_corners_zoomed) != 4) { return MatToUIImage( _gray); }
    
    std::vector<int> diagram;
    if (_small_zoomed.rows > 0) {
        //cv::Mat gray_blurred;
        //cv::GaussianBlur( _gray_zoomed, gray_blurred, cv::Size(5, 5), 2, 2 );
        diagram = classify( _intersections_zoomed, _gray_zoomed, _gz_threshed, _dx, _dy, 1);
    }
    
    // Show results
    cv::Mat drawing;
    DrawBoard drb( _gray_zoomed, _corners_zoomed[0].y, _corners_zoomed[0].x, _board_sz);
    drb.draw( diagram);
    //cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);

    int dx = ROUND( _dx/4.0);
    int dy = ROUND( _dy/4.0);
    ISLOOP (diagram) {
        cv::Point p(ROUND(_intersections_zoomed[i].x), ROUND(_intersections_zoomed[i].y));
        cv::Rect rect( p.x - dx,
                      p.y - dy,
                      2*dx + 1,
                      2*dy + 1);
        cv::rectangle( drawing, rect, cv::Scalar(0,0,255,255));
        if (diagram[i] == BlackWhiteEmpty::BBLACK) {
            draw_point( p, drawing, 2, cv::Scalar(0,255,0,255));
        }
        else if (diagram[i] == BlackWhiteEmpty::WWHITE) {
            draw_point( p, drawing, 5, cv::Scalar(255,0,0,255));
        }
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f11_classify()

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
                                    Points_ &result, float &delta_h, float &delta_v) // out
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
    bool pyr_filtered = false;

    do {
        static std::vector<Points> boards; // Some history for averaging
        UIImageToMat( img, _m, false);
        resize( _m, _small, 350);
        cv::cvtColor( _small, _small, CV_RGBA2RGB);
        cv::cvtColor( _small, _gray, cv::COLOR_RGB2GRAY);
        thresh_dilate( _gray, _gray_threshed);
        
        // Find stones and intersections
        _stone_or_empty.clear();
        BlobFinder::find_empty_places( _gray_threshed, _stone_or_empty); // has to be first
        BlobFinder::find_stones( _gray, _stone_or_empty);
        _stone_or_empty = BlobFinder::clean( _stone_or_empty);
        if (SZ(_stone_or_empty) < 0.8 * SQR(_board_sz)) break;

        // Break if not straight
        float theta = direction( _gray, _stone_or_empty) - PI/2;
        if (fabs(theta) > 4 * PI/180) break;
        
        // Find horiz lines
        _horizontal_lines = homegrown_horiz_lines( _stone_or_empty);
        dedup_horiz_lines( _horizontal_lines, _gray);
        fix_horiz_lines( _horizontal_lines, _gray);
        //PLOG( "HLINES:%d\n", SZ(_horizontal_lines));
        if (SZ( _horizontal_lines) > 40) break;
        if (SZ( _horizontal_lines) < 5) break;

        // Find vertical lines
        _vertical_lines = homegrown_vert_lines( _stone_or_empty);
        dedup_vertical_lines( _vertical_lines, _gray);
        fix_vertical_lines( _vertical_lines, _gray);
        if (SZ( _vertical_lines) > 40) break;
        if (SZ( _vertical_lines) < 5) break;

        // Find corners
        _intersections = get_intersections( _horizontal_lines, _vertical_lines);
        cv::pyrMeanShiftFiltering( _small, _small_pyr, SPATIALRAD, COLORRAD, MAXPYRLEVEL );
        pyr_filtered = true;
        _corners.clear();
        if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
            _corners = find_corners( _stone_or_empty, _horizontal_lines, _vertical_lines,
                                    _intersections, _small_pyr, _gray_threshed);
        }
        if (!board_valid( _corners, _gray)) break;
        // Use median border coordinates to prevent flicker
        //const int BORDBUFLEN = 1;
        //ringpush( _boards, _corners, BORDBUFLEN);
        //Points2f med_board = med_quad( _boards);
        //_corners = med_board;

        _intersections = get_intersections( _horizontal_lines, _vertical_lines);
        
        // Zoom in
        cv::Mat M;
        zoom_in( _gray,  _corners, _gray_zoomed, M);
        zoom_in( _small, _corners, _small_zoomed, M);
        cv::perspectiveTransform( _corners, _corners_zoomed, M);
        cv::perspectiveTransform( _intersections, _intersections_zoomed, M);
        thresh_dilate( _gray_zoomed, _gz_threshed);

        // Classify
        const int TIME_BUF_SZ = 10;
        _diagram = classify( _intersections_zoomed, _gray_zoomed, _gz_threshed, _dx, _dy, TIME_BUF_SZ);
    } while(0);
    
    cv::Mat *canvas;
    if (pyr_filtered) {
        canvas = &_small_pyr;
    }
    else {
        canvas = &_small;
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

#define SHOW_CLASS
#ifdef SHOW_CLASS
        // Show classification result
        ISLOOP (_diagram) {
            cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
            if (_diagram[i] == BlackWhiteEmpty::BBLACK) {
                draw_point( p, *canvas, 5, cv::Scalar(255,0,0,255));
            }
            else if (_diagram[i] == BlackWhiteEmpty::WWHITE) {
                draw_point( p, *canvas, 5, cv::Scalar(0,255,0,255));
            }
        }
        ISLOOP (_intersections) {
            draw_point( _intersections[i], *canvas, 2, cv::Scalar(0,0,255,255));
        }
#else
        // Show one feature for debugging
        ISLOOP (diagram) {
            cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
            //int feat = BWE_sigma[i];
            //int feat = BWE_sum[i];
            //int feat = BWE_crossness_new[i];
            //int feat = BWE_brightness[i];
            int feat = BWE_white_templ_feat[i];
            draw_point( p, img, 5, cm_penny_lane( feat));
        }
#endif
    }
    UIImage *res = MatToUIImage( *canvas);
    //UIImage *res = MatToUIImage( drawing);
    return res;
} // findBoard()


@end





























