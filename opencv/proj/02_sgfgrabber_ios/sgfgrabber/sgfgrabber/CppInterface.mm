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
#import "LineFinder.hpp"
#import "LineFixer.hpp"
#import "BlackWhiteEmpty.hpp"
#import "BlobFinder.hpp"
#import "Clust1D.hpp"

extern cv::Mat mat_dbg;

#define STRETCH_FACTOR 1.1

@interface CppInterface()
//=======================
@property cv::Mat small; // resized image, in color
@property cv::Mat small_zoomed;  // small, zoomed into the board
@property cv::Mat gray;  // Grayscale version of small
@property cv::Mat gray_zoomed;  // Grayscale version of small, zoomed into the board
@property cv::Mat m;     // Mat with image we are working on
@property Contours cont; // Current set of contours
@property int board_sz; // board size, 9 or 19
@property Points stone_or_empty; // places where we suspect stones or empty
@property std::vector<cv::Vec2f> horizontal_lines;
@property std::vector<cv::Vec2f> vertical_lines;
@property Points2f corners;
@property Points2f corners_zoomed;
@property Points2f intersections;
@property float dy;
@property float dx;
@property LineFinder finder;
@property std::vector<Points2f> boards; // history of board corners

@end

@implementation CppInterface
//=========================

//----------------------
- (instancetype)init
{
    self = [super init];
    if (self) {
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
    //load_img( @"board03.jpg", _m);
    //cv::rotate(_m, _m, cv::ROTATE_90_CLOCKWISE);

    resize( _m, _small, 350);
    cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
    //normalize_plane_local(_gray, _gray, 15);

    _stone_or_empty.clear();
    BlobFinder::find_empty_places( _gray, _stone_or_empty); // has to be first
    BlobFinder::find_stones( _gray, _stone_or_empty);
    auto cleaned = BlobFinder::clean( _stone_or_empty);
    _stone_or_empty = cleaned;
    
    // Show results
    cv::Mat drawing;
    //cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP ( _stone_or_empty) {
        //draw_point( _stone_or_empty[i], drawing, 2);
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//--------------------------
- (UIImage *) f01_straight
{
    g_app.mainVC.lbDbg.text = @"01";
    // Get direction of grid. Should be around pi/2 for horizontals
    float theta = direction( _gray, _stone_or_empty) - PI/2;
    // Rotate to exactly pi/2
    rot_img( _gray, theta, _gray);
    rot_img( _small, theta, _small);
    // Rerun the blob detection. We could just rotate the blobs for efficiency.
    _stone_or_empty.clear();
    BlobFinder::find_empty_places( _gray, _stone_or_empty); // has to be first
    BlobFinder::find_stones( _gray, _stone_or_empty);
    auto cleaned = BlobFinder::clean( _stone_or_empty);
    _stone_or_empty = cleaned;
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    draw_points(_stone_or_empty, drawing, 2, cv::Scalar(255,0,0));
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
//    _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
//    // This also removes dups from the points in _finder.horizontal_clusters
//    _finder.cluster();
//
//    cv::Vec2f ratline;
//    do {
//        if (SZ(_finder.m_horizontal_clusters) < 3) break;
//        float dy; int rat_idx;
//        float dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
//
//        _horizontal_lines.clear();
//        find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray.cols,
//                         _horizontal_lines);
//        _vertical_lines.clear();
//    } while(0);
    
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
    //auto d_rho   = (rhos.back() - rhos.front()) / (SZ(rhos)-1);
    auto d_theta = vec_median( d_thetas);
    //auto d_theta  = (thetas.back() - thetas.front()) / (SZ(thetas)-1);
    
    cv::Vec2f med_line = lines[SZ(lines)/2];
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    float rho, theta;
    // If there is a close line, use it. Else interpolate.
    const float X_THRESH = 6;
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
    //auto d_rho   = (rhos.back() - rhos.front()) / (SZ(rhos)-1);
    auto d_theta = vec_median( d_thetas);
    //auto d_theta  = (thetas.back() - thetas.front()) / (SZ(thetas)-1);
    
    cv::Vec2f med_line = lines[SZ(lines)/2];
    std::vector<cv::Vec2f> synth_lines;
    synth_lines.push_back(med_line);
    float rho, theta;
    // If there is a close line, use it. Else interpolate.
    const float Y_THRESH = 6;
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

// Find a vertical line thru pt which hits a lot of other points
// WARNING: allpoints must be sorted by y
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
// WARNING: allpoints must be sorted by x
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
    //PLOG( "maxhits:%d\n", maxhits);
    //int tt = count_points_on_line( res, allpoints);
    if (res[1] < 0.1) {
        int tt = 42;
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

// Use horizontal and vertical lines to find corners such that the board best matches the points we found
//-----------------------------------------------------------------------------------------------------------
Points2f get_corners( const std::vector<cv::Vec2f> &horiz_lines, const std::vector<cv::Vec2f> &vert_lines,
                     const std::vector<PFeat> &pfts, const cv::Mat &img, int board_sz = 19)
{
    int height = img.rows;
    int width  = img.cols;
    int max_idx;
    float max_score;
    
    // Find bounding horiz lines
    max_score = -1E9; max_idx = -1;
    for (int i=0; i < SZ(horiz_lines) - board_sz + 1; i++) {
        cv::Vec2f top_line = horiz_lines[i];
        cv::Vec2f bot_line = horiz_lines[i + board_sz - 1];
        float score = sum_points_between_horiz_lines( top_line, bot_line, pfts, width / 2);
        if (score > max_score) {
            max_score = score;
            max_idx = i;
        }
    }
    //PLOG("MAX:%f\n", max_score);
    //if (max_score < 150) { return Points2f(); }
    //int sz = SZ(pts);
    cv::Vec2f top_line = horiz_lines[max_idx];
    cv::Vec2f bot_line = horiz_lines[max_idx + board_sz - 1];
    
    // Find bounding vert lines
    max_score = -1E9; max_idx = -1;
    for (int i=0; i < SZ(vert_lines) - board_sz + 1; i++) {
        cv::Vec2f left_line = vert_lines[i];
        cv::Vec2f right_line = vert_lines[i + board_sz - 1];
        float score = sum_points_between_vert_lines( left_line, right_line, pfts, height / 2);
        if (score > max_score) {
            max_score = score;
            max_idx = i;
        }
    }
    cv::Vec2f left_line = vert_lines[max_idx];
    cv::Vec2f right_line = vert_lines[max_idx + board_sz - 1];
    
    Point2f tl = intersection( left_line,  top_line);
    Point2f tr = intersection( right_line, top_line);
    Point2f br = intersection( right_line, bot_line);
    Point2f bl = intersection( left_line,  bot_line);
    Points2f corners = {tl, tr, br, bl};
    return corners;
} // get_corners()

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
// ------------------------------------------------------------------------
std::vector<PFeat> find_crosses( const cv::Mat &img,
                                const Points2f &intersections)
{
    int dx=10, dy=10;
    cv::Mat mtmp;
    cv::adaptiveThreshold( img, mtmp, 1, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          11,  // neighborhood_size
                          8); // 8 or ten, need to try both. 8 better for 19x19
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate( mtmp, mtmp, element );
    std::vector<float> features;
    std::vector<PFeat> res;
    // For each intersection of two lines
    for (auto pf:intersections) {
        float crossness = BlackWhiteEmpty::cross_feature_new( mtmp, pf, dx, dy);
        res.push_back( { pf, 1 - crossness } );
        features.push_back( crossness );
    }
//    ISLOOP (intersections) {
//        if (features[i] < 0.01) {
//            res.push_back( intersections[i]);
//        }
//    }
    viz_feature( img, intersections, features, mat_dbg, 255);
    return res;
} // find_crosses()

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
            dst( hood) = feat * multiplier;
        }
    }
} // viz_feature()

// Find the corners
//----------------------------
- (UIImage *) f06_corners
{
    g_app.mainVC.lbDbg.text = @"06";
    auto intersections = get_intersections( _horizontal_lines, _vertical_lines);
    auto crosses = find_crosses( _gray, intersections);
    _corners.clear();
    if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
        _corners = get_corners( _horizontal_lines, _vertical_lines, crosses, _gray);
    }
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f06_corners()

//// Find the corners
////----------------------------
//- (UIImage *) f06_corners
//{
//    g_app.mainVC.lbDbg.text = @"06";
//    _corners.clear();
//    if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
//        _corners = get_corners( _horizontal_lines, _vertical_lines, _stone_or_empty, _gray);
//    }
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    get_color(true);
//    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//}

// Unwarp the square defined by corners
//------------------------------------------------------------------------
void zoom_in( const cv::Mat &img, const Points2f &corners, cv::Mat &dst, cv::Mat &M)
{
    int marg = img.cols / 20;
    // Target square for transform
    Points2f square = {
        cv::Point( marg, marg),
        cv::Point( img.cols - marg, marg),
        cv::Point( img.cols - marg, img.cols - 2*marg),
        cv::Point( marg, img.cols - 2*marg) };
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
    }
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Repeat whole process 01 to 06 on the zoomed in version
//-----------------------------------------------------------
- (UIImage *) f08_repeat_on_zoomed
{
    g_app.mainVC.lbDbg.text = @"08";
    _corners = _corners_zoomed;
    //_gray_zoomed = _gray.clone();

//    do {
//        if (_gray_zoomed.rows == 0) break;
//        // Blobs
//        _stone_or_empty.clear();
//        BlobFinder::find_empty_places( _gray_zoomed, _stone_or_empty); // has to be first
//        //BlobFinder::find_stones( _gray_zoomed, _stone_or_empty);
//
//        // Horizontals
//        _finder = LineFinder( _stone_or_empty, _board_sz, _gray_zoomed.size() );
//        // This also removes dups from the points in _finder.horizontal_clusters
//        _finder.cluster();
//        if (SZ(_finder.m_horizontal_clusters) < 3) {
//            int tt=42;
//            break;
//        }
//        cv::Vec2f ratline;
//        float dy; int rat_idx;
//        float dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
//        _horizontal_lines.clear();
//        find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray_zoomed.cols,
//                         _horizontal_lines);
//
//        // Verticals
//        _vertical_lines = homegrown_vert_lines( _stone_or_empty);
//        dedup_vertical_lines( _vertical_lines, _gray_zoomed);
//        fix_vertical_lines( _vertical_lines, _gray_zoomed);
//
//        // Corners
//        _corners = get_corners( _horizontal_lines, _vertical_lines, _stone_or_empty, _gray_zoomed);
//    } while(0);
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    //cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);
    //ISLOOP ( _stone_or_empty) {
    //    draw_point( _stone_or_empty[i], drawing, 2);
    //}
    //draw_polar_lines( _horizontal_lines, drawing);
    //draw_polar_lines( _vertical_lines, drawing);
    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f08_repeat_on_zoomed()

// Intersections on zoomed from corners
//-----------------------------------------------------------
- (UIImage *) f09_intersections
{
    g_app.mainVC.lbDbg.text = @"09";
    
    if (SZ(_corners) == 4) {
        get_intersections_from_corners( _corners, _board_sz, _intersections, _dx, _dy);
    }
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    draw_points( _intersections, drawing, 1, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f09_intersections()

// Visualize some features
//-------------------------------
- (UIImage *) f10_features
{
    return NULL;
    g_app.mainVC.lbDbg.text = @"10";
    
    if (SZ(_corners) == 4) {
        get_intersections_from_corners( _corners, _board_sz, _intersections, _dx, _dy);
    }
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    draw_points( _intersections, drawing, 1, cv::Scalar(255,0,0));
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

//--------------------------------------------------------------------------------------------------
std::vector<int> classify( const Points2f &intersections_, const cv::Mat &gray_normed, float dx, float dy)
{
    Points2f intersections;
    std::vector<std::vector<int> > diagrams;
    // Wiggle the regions a little.
    // Perspective tends to see stones with y too small (higher up).
    // Therefore, look two pixels up, but never down.
    intersections = translate_points( intersections_, 0, 0);
    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
                                                  intersections,
                                                  dx, dy));
//    intersections = translate_points( intersections_, -1, 0);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, 1, 0);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, 0, 1);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, -1, 1);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, 1, 1);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, 0, 2);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, -1, 2);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
//    intersections = translate_points( intersections_, 1, 2);
//    diagrams.push_back( BlackWhiteEmpty::classify( gray_normed,
//                                                  intersections,
//                                                  dx, dy));
    
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
    const int BUFSZ = 20; // frames of history for voting
    static std::vector<std::vector<int> > timevotes(19*19);
    ISLOOP (diagram) {
        ringpush( timevotes[i], diagram[i], BUFSZ);
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
    
    std::vector<int> diagram;
    if (_small_zoomed.rows > 0) {
        //cv::Mat gray_blurred;
        //cv::GaussianBlur( _gray_zoomed, gray_blurred, cv::Size(5, 5), 2, 2 );
        diagram = classify( _intersections, _gray_zoomed, _dx, _dy);
    }
    
    // Show results
    cv::Mat drawing;
    //cv::cvtColor( _gray_zoomed, drawing, cv::COLOR_GRAY2RGB);
    cv::cvtColor( mat_dbg, drawing, cv::COLOR_GRAY2RGB);

    int dx = ROUND( _dx/4.0);
    int dy = ROUND( _dy/4.0);
    ISLOOP (diagram) {
        cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
        cv::Rect rect( p.x - dx,
                      p.y - dy,
                      2*dx + 1,
                      2*dy + 1);
        cv::rectangle( drawing, rect, cv::Scalar(255,0,0,255));
        if (diagram[i] == BlackWhiteEmpty::BBLACK) {
            draw_point( p, drawing, 1, cv::Scalar(255,255,255,255));
        }
        else if (diagram[i] == BlackWhiteEmpty::WWHITE) {
            draw_point( p, drawing, 2, cv::Scalar(0,0,255,255));
        }
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f10_classify()

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
            cv::Mat hood = cv::Mat( img, rect);
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
} // get_intersections()

#pragma mark - Real time implementation
//========================================

// f00_*, f01_*, ... all in one go
//--------------------------------------------
- (UIImage *) findBoard:(UIImage *) img
{
    //static int counter = 0;
    _board_sz = 19;
    do {
        //const int N_BOARDS = 8;
        static std::vector<Points> boards; // Some history for averaging
        UIImageToMat( img, _m, false);
        resize( _m, _small, 350);
        cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
        
        // Find stones and intersections
        _stone_or_empty.clear();
        BlobFinder::find_empty_places( _gray, _stone_or_empty); // has to be first
        BlobFinder::find_stones( _gray, _stone_or_empty);
        if (SZ(_stone_or_empty) < 3) break;

        // Find dominant direction
        float theta = direction( _gray, _stone_or_empty) - PI/2;
        if (fabs(theta) > 4 * PI/180) break;
        
        // Find horiz lines
        _horizontal_lines = homegrown_horiz_lines( _stone_or_empty);
        dedup_horiz_lines( _horizontal_lines, _gray);
        fix_horiz_lines( _horizontal_lines, _gray);
//
//        _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
//        _finder.cluster();
//        if (SZ(_finder.m_horizontal_clusters) < 3) break;
//        cv::Vec2f ratline;
//        float dy; int rat_idx;
//        float dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
//        _horizontal_lines.clear();
//        find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray.cols,
//                         _horizontal_lines);
        
        // Find vertical lines
        _vertical_lines = homegrown_vert_lines( _stone_or_empty);
        dedup_vertical_lines( _vertical_lines, _gray);
        fix_vertical_lines( _vertical_lines, _gray);
        
        //PLOG("$$$$$$$$$$$ %d\n", counter++ % 1000);
        
        // Find corners
        auto intersections = get_intersections( _horizontal_lines, _vertical_lines);
        auto crosses = find_crosses( _gray, intersections);
        _corners.clear();
        if (SZ(_horizontal_lines) && SZ(_vertical_lines)) {
            _corners = get_corners( _horizontal_lines, _vertical_lines, crosses, _gray);
        }
        if (!board_valid( _corners, _gray)) break;
        //@@@ prevent board flicker
        const int BORDBUFLEN = 5;
        ringpush( _boards, _corners, BORDBUFLEN);
        Points2f med_board = med_quad( _boards);
        //float diff = diff_quads( _corners, med_board);
        //PLOG( "######### %.2f\n", diff);
        //if (diff > 0.1) { _corners = med_board; }
        _corners = med_board;

        get_intersections_from_corners( _corners, _board_sz, _intersections, _dx, _dy);
        if (_dx < 2 || _dy < 2) break;
        //auto diagram = classify( _intersections, _small, _dx, _dy);

        //draw_points( _stone_or_empty, _small, 1, cv::Scalar(255,0,0,255));
        
        // Zoom in
        cv::Mat M;
        zoom_in( _gray,  _corners, _gray_zoomed, M);
        zoom_in( _small, _corners, _small_zoomed, M);
        cv::perspectiveTransform( _corners, _corners_zoomed, M);

        draw_line( cv::Vec4f( _corners[0].x, _corners[0].y, _corners[1].x, _corners[1].y),
                  _small, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[1].x, _corners[1].y, _corners[2].x, _corners[2].y),
                  _small, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[2].x, _corners[2].y, _corners[3].x, _corners[3].y),
                  _small, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[3].x, _corners[3].y, _corners[0].x, _corners[0].y),
                  _small, cv::Scalar( 255,0,0,255));
//
//
//        // Repeat process on zoomed image
//        _stone_or_empty.clear();
//        BlobFinder::find_empty_places( _gray_zoomed, _stone_or_empty); // has to be first
//        BlobFinder::find_stones( _gray_zoomed, _stone_or_empty);
//        if (SZ(_stone_or_empty) < 3) break;
//
//        // Horizontals
//        _finder = LineFinder( _stone_or_empty, _board_sz, _gray_zoomed.size() );
//        // This also removes dups from the points in _finder.horizontal_clusters
//        _finder.cluster();
//        if (SZ(_finder.m_horizontal_clusters) < 3) break;
//        //cv::Vec2f ratline;
//        //float dy; int rat_idx;
//        dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
//        _horizontal_lines.clear();
//        find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray_zoomed.cols,
//                         _horizontal_lines);
//
//        // Verticals
//        _vertical_lines = homegrown_vert_lines( _stone_or_empty);
//        dedup_vertical_lines( _vertical_lines, _gray_zoomed);
//        fix_vertical_lines( _vertical_lines, _gray_zoomed);
//
//        // Corners
//        _corners = get_corners( _horizontal_lines, _vertical_lines, _stone_or_empty, _gray_zoomed);
//        if (SZ(_corners) != 4) break;

        // Classify
        Points2f intersections_zoomed;
        get_intersections_from_corners( _corners_zoomed, _board_sz, intersections_zoomed, _dx, _dy);
        if (_dx < 2 || _dy < 2) break;
        //cv::Mat gray_normed;
        //normalize_plane( _gray_zoomed, gray_normed);
        ///cv::Mat gray_blurred;
        //cv::GaussianBlur( _gray_zoomed, gray_blurred, cv::Size(5, 5), 2, 2 );
        //cv::GaussianBlur( _gray, gray_blurred, cv::Size(5, 5), 2, 2 );
        auto diagram = classify( intersections_zoomed, _gray_zoomed, _dx, _dy);
//        auto diagram = classify( _intersections, _gray, _dx, _dy);

        ISLOOP (diagram) {
//            cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
//            cv::Rect rect( p.x - dx,
//                          p.y - dy,
//                          2*dx + 1,
//                          2*dy + 1);
//            cv::rectangle( drawing, rect, cv::Scalar(255,0,0,255));
            if (diagram[i] == BlackWhiteEmpty::BBLACK) {
                cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
                draw_point( p, _small, 2, cv::Scalar(255,255,255,255));
            }
            else if (diagram[i] == BlackWhiteEmpty::WWHITE) {
                cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
                draw_point( p, _small, 2, cv::Scalar(255,0,0,255));
            }
//            else if (diagram[i] == BlackWhiteEmpty::EEMPTY) {
//                cv::Point p(ROUND(_intersections[i].x), ROUND(_intersections[i].y));
//                draw_point( p, _small, 2, cv::Scalar(0,255,0,255));
//            }
        }
        
    } while(0);
        
    UIImage *res = MatToUIImage( _small);
    return res;
} // findBoard()


@end





























