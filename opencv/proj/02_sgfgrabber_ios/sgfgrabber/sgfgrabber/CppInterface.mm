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
#include "Clust1D.hpp"

const cv::Size TMPL_SZ(16,16);

#define STRETCH_FACTOR 1.1

@interface CppInterface()
//=======================
@property cv::Mat small; // resized image, in color
@property cv::Mat gray;  // Grayscale version of small
@property cv::Mat m;     // Mat with image we are working on
@property Contours cont; // Current set of contours
@property Points board;  // Current hypothesis on where the board is
@property Points board_zoomed; // board corners after zooming in
@property int board_sz; // board size, 9 or 19
@property Points stone_or_empty; // places where we suspect stones or empty
@property std::vector<cv::Vec2f> horizontal_lines;
@property std::vector<cv::Vec2f> vertical_lines;
@property Points2f corners;
@property cv::Vec2f horiz_line;
@property cv::Vec2f vert_line;
@property LineFinder finder;

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

// Make our 4-polygon a little larger
//-------------------------------------
Points2f enlarge_board( Points board)
{
    return stretch_quad( board, STRETCH_FACTOR);
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
    if (!conts.size()) return whole_img_quad( binImg);
    boardImg = cv::Mat::zeros( binImg.size(), CV_8UC1 );
    cv::drawContours( boardImg, conts, 0, cv::Scalar(255), 3);
    // Find lines
    std::vector<cv::Vec4f> lines;
    HoughLinesP(boardImg, lines, 1, PI/180, 150, 0, 0 );
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
    if (!top_bottom_lines.size()) return whole_img_quad( binImg);
    // Separate left from right
    std::vector<std::vector<cv::Vec4f> > left_right_lines;
    left_right_lines = cluster( vert_lines, 2,
                               [](cv::Vec4f &line) {
                                   return (line[0] + line[2]) / 2.0;
                               });
    if (!left_right_lines.size()) return whole_img_quad( binImg);

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
    if (board.size() != 4) return whole_img_quad( binImg);
    return board;
} // find_board()


#pragma mark - Processing Pipeline for debugging
//=================================================

/*
 For each point on ratline
   for each line thru point with angle between -10 and +10 degrees
     find the number of points on that line
     keep the line with the most points
 Sort the lines by x for y==middle
 Interpolate any gaps (dx > 1.5 * median dx
 Find the best grid position => corners.
 Warp.
 
 */

//-----------------------------------------
- (UIImage *) f00_blobs:(UIImage *)img
{
    _board_sz=19;
    g_app.mainVC.lbDbg.text = @"00";
    
    // Live Camera
    //UIImageToMat( img, _m);
    
    // From file
    load_img( @"board03.jpg", _m);
    cv::rotate(_m, _m, cv::ROTATE_90_CLOCKWISE);

    resize( _m, _small, 350);
    cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
    
    _stone_or_empty.clear();
    BlobFinder::find_empty_places( _gray, _stone_or_empty); // has to be first
    BlobFinder::find_stones( _gray, _stone_or_empty);
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP ( _stone_or_empty) {
        draw_point( _stone_or_empty[i], drawing, 2);
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
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Given the dy ratio, and the baseline vertical distance dy_0,
// compute pixel y coord for gridpoint n horizontal lines up.
//-----------------------------------------------------------------
int pixel_y( float dy_rat, int n, float dy_0 = 1.0)
{
    float f = exp( n * log( dy_rat));
    float res = (1.0 / log( dy_rat)) * (1.0 - f);
    return ROUND( -res);
}

//// Convert diatance in x direction to what it would have been
//// if viewed straight from the top. cur_d_rho is the delta x on
//// the current horizontal line, treu_d_rho is the delta x on
//// the horizontal we want to remain unchanged.
////---------------------------------------------------------------
//float true_x( float true_d_rho, float cur_d_rho, float cur_x)
//{
//    return cur_x * (true_d_rho / cur_d_rho);
//}
//
//// Convert duistance in y direction to what it would have been
//// if viewed strsight from the top. true_d_rho is the distance
//// between the unchanged horizontal line and the next horizontal
//// line above. loss_ratio is the ratio between adjacent horizontal
//// line distances. The true distance is a geometric sum.
////-----------------------------------------------------------------
//float true_y( float true_d_rho, float loss_ratio, int cur_h_line)
//{
//    true_d_rho * exp( 1 - loss_ratio, 1 + cur_y /
//}

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

//// Find the most vertical line
////------------------------------------------------------------
//cv::Vec2f best_vline( const std::vector<cv::Vec2f> &vlines)
//{
//    cv::Vec2f res = { 0, 1E9 };
//    float mindiff = 1E9;
//    ISLOOP (vlines) {
//        float theta = vlines[i][1];
//        float d = fabs( theta );
//        PLOG( "theta, d: %.2f %.2f\n", theta, d);
//        if (d < mindiff) {
//            mindiff = d;
//            res = vlines[i];
//        }
//    }
//    return res;
//}

// Walk the distance d along the line, starting at p.
//--------------------------------------------------------------
cv::Point walk_the_line( cv::Vec2f line, cv::Point p, float d)
{
    cv::Point res;
    float dx = d * sin( line[1]);
    float dy = -d * cos( line[1]);
    res = { int(round(p.x + dx)), int(round(p.y + dy)) };
    return res;
}

// Find line n lines up from line. lines must be sorted by rho.
// Use negative n to look the other way.
//-----------------------------------------------------------------------------------
cv::Vec2f n_lines_up( const std::vector<cv::Vec2f> &lines, cv::Vec2f line, int n )
{
    int idx = -1;
    ISLOOP (lines) {
        if (lines[i][0] > line[0]) {
            idx = i;
            break;
        }
    }
    idx = idx - 1 - n;
    if (idx < 0) { idx = 0; }
    return lines[idx];
}

// Get median dx of a cluster
//------------------------------
float median_dx( Points pts)
{
    cv::Point res;
    std::sort( pts.begin(), pts.end(), [](cv::Point p1, cv::Point p2) { return p1.x < p2.x; });
    Points deltas = vec_delta( pts);
    res = vec_median( deltas, [](cv::Point p) { return p.x; });
    return res.x;
}

// Various loss functions to remove one line at a time
//=========================================================

//// Get the variance in dx in the middle for a bunch of lines.
//// Maybe skip one of them.
////--------------------------------------------------------------------------------
//float x_line_var( const std::vector<cv::Vec2f> &lines, int height, int skip = -1)
//{
//    std::vector<float> vals;
//    float middle = height / 2.0;
//    ISLOOP (lines) {
//        if (i == skip) continue;
//        auto line = lines[i];
//        float x = x_from_y( middle, line);
//        vals.push_back( x);
//    }
//    std::sort( vals.begin(), vals.end(), [](float v1, float v2) { return v1 < v2; });
//    std::vector<float> deltas = vec_delta( vals);
//    float res = vec_var( deltas);
//    //float res = vec_var_med( deltas);
//    return res;
//}

//// Get the variance in x at the top for a bunch of lines.
//// Maybe skip one of them.
////---------------------------------------------------------------------------
//float top_x_line_var( const std::vector<cv::Vec2f> &lines, int skip = -1)
//{
//    std::vector<float> vals;
//    ISLOOP (lines) {
//        if (i == skip) continue;
//        auto line = lines[i];
//        float x = x_from_y( 1, line);
//        vals.push_back( x);
//    }
//    float res = vec_var( vals);
//    return res;
//}

// Sort lines by x in the screen middle, then get variance of dx nearer the top.
// Maybe skip one line to see if removing it helps.
// The idea is that the order at the top should be the same as in the middle,
// and the lines should be equidistant for a given y.
//------------------------------------------------------------------------------------
float top_x_var_by_middle( std::vector<cv::Vec2f> lines, int height, int skip = -1)
{
    assert( skip < 0 || skip < SZ(lines));
    typedef struct { int idx; float val; } IdxVal;
    int res_len = SZ( lines);
    if (skip >= 0) res_len--;
    std::vector<IdxVal> topvals( res_len);
    std::vector<IdxVal> midvals( res_len);
    const float middle = height / 2.0;
    int k = 0;
    ISLOOP (lines) {
        if (i == skip) continue;
        auto line = lines[i];
        //IdxVal top = { k, x_from_y( 20, line) };
        IdxVal top = { k, x_from_y( middle/12, line) };
        //IdxVal top = { k, x_from_y( 1, line) };
        IdxVal mid = { k, x_from_y( middle, line) };
        topvals[k] = top;
        midvals[k] = mid;
        k++;
    }
    std::sort( topvals.begin(), topvals.end(),
              [midvals](IdxVal iv1, IdxVal iv2) {
                  return midvals[iv1.idx].val < midvals[iv2.idx].val;
              });
    auto vals = vec_extract( topvals, [](IdxVal iv){return iv.val;});
    auto deltas = vec_delta( vals);
    //PLOG( "deltas:\n");
    //ISLOOP( deltas) { PLOG( "%.2f\n", deltas[i]); }
    float res = vec_var( deltas);
    //float res = vec_var_med( deltas);
    //PLOG( "AVG:%.2f\n", vec_avg( deltas));
    return res;
}

//-----------------------------
- (UIImage *) f02_horiz_lines
{
    g_app.mainVC.lbDbg.text = @"02";
    _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
    // This also removes dups from the points in _finder.horizontal_clusters
    _finder.cluster();
    cv::Vec2f ratline;
    float dy; int rat_idx;
    float dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
    
    _horizontal_lines.clear();
    find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray.cols,
                     _horizontal_lines);
    _vertical_lines.clear();
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color( true);
    ISLOOP (_horizontal_lines) {
        cv::Scalar col = get_color();
        draw_polar_line( _horizontal_lines[i], drawing, col);
    }
    draw_polar_line( ratline, drawing, cv::Scalar( 255,128,64));
    UIImage *res = MatToUIImage( drawing);
    return res;
}

// Thin down vertical Hough lines
//----------------------------------
- (UIImage *) f03_vert_lines
{
    g_app.mainVC.lbDbg.text = @"03";
    if (!SZ(_vertical_lines)) {
        get_vertical_hough_lines( _stone_or_empty, _gray, _vertical_lines);
    }
    thin_vertical_hough_lines( _vertical_lines, _gray);
    
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

// Get an initial set of vertical candidate lines
//---------------------------------------------------------------------
void get_vertical_hough_lines( const Points &pts, const cv::Mat &img,
                              std::vector<cv::Vec2f> &out)
{
    // Find candidate Hough lines
    std::vector<cv::Vec2f> lines;
    cv::Mat canvas = cv::Mat::zeros( img.rows, img.cols, CV_8UC1);
    ISLOOP (pts) {
        draw_point( pts[i], canvas,2, cv::Scalar(255));
    }
    std::vector<std::vector<cv::Vec2f> > horiz_vert_other_lines;
    HoughLines(canvas, lines, 1, PI/180, 10, 0, 0 );
    lines.resize(500); // Only keep the best ones
    for (auto& line: lines) {
        if (line[0] < 0) { line[0] *= -1; line[1] -= PI; }
    }
    
    // Get the vertical lines
    horiz_vert_other_lines = partition( lines, 3,
                                       [](cv::Vec2f &line) {
                                           const float thresh = 25.0;
                                           float theta = line[1] * (180.0 / PI);
                                           if (fabs(theta - 180) < thresh) return 1;
                                           else if (fabs(theta) < thresh) return 1;
                                           else if (fabs(theta-90) < thresh) return 0;
                                           else return 2;
                                       });
    out = horiz_vert_other_lines[1];
}

// Thin a bunch of vertical lines down to reduce crossings and variance.
//---------------------------------------------------------------------------------
void thin_vertical_hough_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img,
                               int board_sz = 19)
{
    // Thin the vertical lines down
    float base_cost;
    do {
        base_cost = top_x_var_by_middle( lines, img.rows);
        //float base_cost = x_line_var( _vertical_lines, _gray.rows);
        float min_loss = 1E9;
        int min_idx = 0;
        ISLOOP( lines) {
            float cost = top_x_var_by_middle( lines, img.rows, i);
            //float cost = x_line_var( _vertical_lines, _gray.rows, i);
            float loss = cost - base_cost;
            //PLOG("%5d base_cost: %.2f cost: %.2f loss: %.2f \n", i, base_cost, cost, loss);
            if (loss < min_loss) {
                min_loss = loss;
                min_idx = i;
            }
        }
        //PLOG("base_cost %.0f with %d lines\n", base_cost, SZ(_vertical_lines));
        lines.erase( lines.begin() + min_idx);
    } while (SZ(lines) > board_sz );
}

// Replace close clusters of vert lines by their average.
//-----------------------------------------------------------------------------------
void dedup_vertical_hough_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
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

// Cluster vertical Hough lines to remove close duplicates.
//------------------------------------------------------------
- (UIImage *) f04_vert_lines_2
{
    g_app.mainVC.lbDbg.text = @"04";
    dedup_vertical_hough_lines( _vertical_lines, _gray);
    
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
void fix_vertical_hough_lines( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
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
    float thresh = 6;
    // Lines to the right
    rho = med_line[0];
    theta = med_line[1];
    ILOOP(100) {
        if (!i) continue;
        rho += d_rho;
        theta += d_theta;
        float x = x_from_y( middle_y, cv::Vec2f( rho, theta));
        int close_idx = vec_closest( xes, x);
        if (fabs( x - xes[close_idx]) < thresh) {
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
        if (fabs( x - xes[close_idx]) < thresh) {
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
} // fix_vertical_hough_lines()

// Find vertical line parameters
//---------------------------------
- (UIImage *) f05_vert_params
{
    g_app.mainVC.lbDbg.text = @"05";
    fix_vertical_hough_lines( _vertical_lines, _gray);
    
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

// Use y at the center to count points between two horizontal lines.
//--------------------------------------------------------------------------------------------------------------
int count_points_between_horiz_lines( cv::Vec2f top_line, cv::Vec2f bot_line, Points points, int middle_x)
{
    int res = 0;
    const float EPS = 5;
    for (auto p: points) {
        float tdist = dist_point_line( p, top_line);
        float bdist = dist_point_line( p, bot_line);
        if ((tdist > 0 || fabs( tdist) < EPS) && (bdist < 0 || fabs( bdist) < EPS) ) {
            res++;
        }
    }
    return res;
}

// Use x at the center to count points between two vertical lines.
//--------------------------------------------------------------------------------------------------------------
int count_points_between_vert_lines( cv::Vec2f left_line, cv::Vec2f right_line, Points points, int middle_y)
{
    int res = 0;
    const float EPS = 5;
    for (auto p: points) {
        float ldist = dist_point_line( p, left_line);
        float rdist = dist_point_line( p, right_line);
        if ((ldist > 0 || fabs( ldist) < EPS) && (rdist < 0 || fabs( rdist) < EPS) ) {
            res++;
        }
    }
    return res;
}

// Use horizontal and vertical lines to find corners such that the board best matches the points we found
//-----------------------------------------------------------------------------------------------------------
Points2f get_corners( const std::vector<cv::Vec2f> &horiz_lines, const std::vector<cv::Vec2f> &vert_lines,
                     const Points &pts, const cv::Mat &img, int board_sz = 19)
{
    int height = img.rows;
    int width  = img.cols;
    int max_n, max_idx;
    
    // Find bounding horiz lines
    max_n = -1E9; max_idx = -1;
    for (int i=0; i < SZ(horiz_lines) - board_sz + 1; i++) {
        cv::Vec2f top_line = horiz_lines[i];
        cv::Vec2f bot_line = horiz_lines[i + board_sz - 1];
        int n = count_points_between_horiz_lines( top_line, bot_line, pts, width / 2);
        if (n > max_n) {
            max_n = n;
            max_idx = i;
        }
    }
    cv::Vec2f top_line = horiz_lines[max_idx];
    cv::Vec2f bot_line = horiz_lines[max_idx + board_sz - 1];
    
    // Find bounding vert lines
    max_n = -1E9; max_idx = -1;
    for (int i=0; i < SZ(vert_lines) - board_sz + 1; i++) {
        cv::Vec2f left_line = vert_lines[i];
        cv::Vec2f right_line = vert_lines[i + board_sz - 1];
        int n = count_points_between_vert_lines( left_line, right_line, pts, height / 2);
        if (n > max_n) {
            max_n = n;
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
}

// Find the corners
//----------------------------
- (UIImage *) f06_corners //@@@
{
    g_app.mainVC.lbDbg.text = @"06";
    _corners = get_corners( _horizontal_lines, _vertical_lines, _stone_or_empty, _gray);
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    get_color(true);
    draw_points( _corners, drawing, 3, cv::Scalar(255,0,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}

//// Improve line candidates by interpolation
////--------------------------------------------
//- (UIImage *) f04_clean_horiz_lines
//{
//    g_app.mainVC.lbDbg.text = @"09";
//    LineFixer fixer;
//    fixer.fix( _horizontal_lines, _finder.m_horizontal_clusters, _horizontal_lines );
//
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    draw_lines( _horizontal_lines, drawing, cv::Scalar(0,0,255));
//    draw_points( _stone_or_empty, drawing, 2, cv::Scalar(0,255,0));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//}

//----------------------
- (UIImage *) f02_flood
{
    g_app.mainVC.lbDbg.text = @"02";
    flood_from_center( _m);
    UIImage *res = MatToUIImage( _m);
    return res;
}

//-----------------------------
- (UIImage *) f03_find_board
{
    g_app.mainVC.lbDbg.text = @"03";
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
    g_app.mainVC.lbDbg.text = @"04";
    if (!_board.size()) { return MatToUIImage( _m); }
    // Zoom out a little
    Points2f board_stretched = enlarge_board( _board);
    cv::Mat transform = zoom_quad( _gray, _gray, board_stretched);
    cv::warpPerspective( _small, _small, transform, cv::Size(_small.cols, _small.rows));
    Points2f b,tt;
    points2float( _board, b);
    cv::perspectiveTransform( b, tt, transform);
    points2int( tt, _board_zoomed);
    
    UIImage *res = MatToUIImage( _gray);
    return res;
}

// Save small crops around intersections for use as template
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

// Find rough guess at stones and empty intersections
// Also determines board size (9,13,19)
//------------------------------------------------------
- (UIImage *) f05_find_intersections
{
    g_app.mainVC.lbDbg.text = @"05";
    Points pts;
    BlobFinder::find_empty_places( _gray, pts); // has to be first
    BlobFinder::find_stones( _gray, pts);
    // Use only inner ones
    Points2f innerboard = stretch_quad( _board_zoomed, 1.01);
    _stone_or_empty = Points();
    ISLOOP (pts) {
        cv::Point2f p( pts[i]);
        if (cv::pointPolygonTest( innerboard, p, false) > 0) {
            _stone_or_empty.push_back( p);
        }
    }
    _board_sz = 19;
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

//// Find line candidates from stones and intersections
////------------------------------------------------------------------------
//- (UIImage *) f06_find_lines
//{
//    g_app.mainVC.lbDbg.text = @"06";
//    NSString *func = @"f06_find_h_lines()";
//    if (_stone_or_empty.size() < _board_sz) {
//        NSLog( @"%@: not enough points", func);
//        return MatToUIImage( _gray);
//    }
//    _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
//    //cv::Vec4f hslope, vslope;
//    _finder.get_lines( _horizontal_lines, _vertical_lines);
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    ISLOOP( _finder.m_vertical_clusters) {
//        Points cl = _finder.m_vertical_clusters[i];
//        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
//        draw_points( cl, drawing, 2, color);
//    }
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//} // f06_find_lines()

//// Show the lines we found
////------------------------------------------------------------------------
//- (UIImage *) f07_show_horiz_lines
//{
//    g_app.mainVC.lbDbg.text = @"07";
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    draw_lines( _horizontal_lines, drawing, cv::Scalar(0,0,255));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//} // f07_show_horiz_lines()
//
//// Show the lines we found
////------------------------------------------------------------------------
//- (UIImage *) f08_show_vert_lines
//{
//    g_app.mainVC.lbDbg.text = @"08";
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    draw_lines( _vertical_lines, drawing, cv::Scalar(0,0,255));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//} // f08_show_vert_lines()
//
//
////--------------------------------------------
//- (UIImage *) f10_clean_vert_lines
//{
//    g_app.mainVC.lbDbg.text = @"10";
//    LineFixer fixer;
//    fixer.fix( _vertical_lines, _finder.m_vertical_clusters, _vertical_lines );
//
//    // Show results
//    cv::Mat drawing;
//    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
//    draw_lines( _vertical_lines, drawing, cv::Scalar(0,0,255));
//    draw_points( _stone_or_empty, drawing, 2, cv::Scalar(0,255,0));
//    UIImage *res = MatToUIImage( drawing);
//    return res;
//}



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
    int dx = round(wavelen_h/4.0); int dy = round(wavelen_v/4.0);
    cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
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
        float brightness = ssum[0] / area;
        //float brightness_g = ssum[1] / area;
        //float brightness_b = ssum[2] / area;
        //float v = sqrt (brightness_r*brightness_r + brightness_g*brightness_g + brightness_b*brightness_b);
        f.features.push_back( brightness);
        //f.features.push_back( brightness_g);
        //f.features.push_back( brightness_b);
        //std::cout << v << std::endl;
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

//// Classify intersections into b,w,empty
////----------------------------------------
//- (UIImage *) f11_classify
//{
//    g_app.mainVC.lbDbg.text = @"11";
//    Points intersections;
//    std::vector<int> diagram =
//    BlackWhiteEmpty::get_diagram( _small,
//                                 _horizontal_lines, _finder.m_wavelen_h,
//                                 _vertical_lines, _finder.m_wavelen_v,
//                                 _board_sz,
//                                 _board_zoomed,
//                                 intersections);
//
//    cv::Mat drawing;
//    drawing = _small.clone();
//    int dx = round(_finder.m_wavelen_h/4.0);
//    int dy = round(_finder.m_wavelen_h/4.0);
//    // Black stones
//    ISLOOP (diagram) {
//        cv::Point p = intersections[i];
//        cv::Rect rect( p.x - dx,
//                      p.y - dy,
//                      2*dx+1,
//                      2*dy+1);
//        cv::rectangle( drawing, rect, cv::Scalar(255,0,0,255));
//        if (diagram[i] == BlackWhiteEmpty::BBLACK) {
//            draw_point( p, drawing, 1, cv::Scalar(255,255,255,255));
//        }
//        else if (diagram[i] != BlackWhiteEmpty::EEMPTY) {
//            draw_point( p, drawing, 2, cv::Scalar(0,0,0,255));
//        }
//    }
//
//    UIImage *res = MatToUIImage( drawing);
//    //UIImage *res = MatToUIImage( zoomed_edges);
//    return res;
//
//} // f07_classify()

#pragma mark - Real time implementation
//========================================

// f00_*, f01_*, ... all in one go
//--------------------------------------------
- (UIImage *) findBoard:(UIImage *) img
{
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
        if (fabs(theta) > 0.02) break;
        
        // Find horiz lines
        _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
        _finder.cluster();
        if (SZ(_finder.m_horizontal_clusters) < 3) break;
        cv::Vec2f ratline;
        float dy; int rat_idx;
        float dy_rat = _finder.dy_rat( ratline, dy, rat_idx);
        _horizontal_lines.clear();
        find_horiz_lines( ratline, dy, dy_rat, _finder.m_horizontal_lines, _board_sz, _gray.cols,
                         _horizontal_lines);
        
        // Find vertical lines
        get_vertical_hough_lines( _stone_or_empty, _gray, _vertical_lines);
        thin_vertical_hough_lines( _vertical_lines, _gray);
        dedup_vertical_hough_lines( _vertical_lines, _gray);
        fix_vertical_hough_lines( _vertical_lines, _gray);
        
        // Find corners
        _corners = get_corners( _horizontal_lines, _vertical_lines, _stone_or_empty, _gray);
        //draw_line( cv::Vec4f( _corners[0].x, _corners[0].y, _corners[1].x, _corners[1].y),
        //          _small, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[1].x, _corners[1].y, _corners[2].x, _corners[2].y),
                  _small, cv::Scalar( 255,0,0,255));
        //draw_line( cv::Vec4f( _corners[2].x, _corners[2].y, _corners[3].x, _corners[3].y),
        //          _small, cv::Scalar( 255,0,0,255));
        draw_line( cv::Vec4f( _corners[3].x, _corners[3].y, _corners[0].x, _corners[0].y),
                  _small, cv::Scalar( 255,0,0,255));
        draw_points( _stone_or_empty, _small, 1, cv::Scalar(255,0,0,255));

    } while(0);
        
    //cv::cvtColor( small, small, cv::COLOR_RGB2BGR);
    UIImage *res = MatToUIImage( _small);
    return res;
} // findBoard()

//// f00_*, f01_*, ... all in one go
////--------------------------------------------
//- (UIImage *) findBoard:(UIImage *) img
//{
//    const int N_BOARDS = 8;
//    static std::vector<Points> boards; // Some history for averaging
//    UIImageToMat( img, _m, false);
//    resize( _m, _small, 350);
//    //cv::cvtColor( small, small, cv::COLOR_BGR2RGB);
//    cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
//    cv::adaptiveThreshold(_gray, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
//                          3, // neighborhood_size
//                          4); // constant to add. 2 to 6 is the viable range
//    // Make sure the board in the center is in one piece
//    int iterations = 1;
//    morph_closing( _m, cv::Size(3,1), iterations);
//    morph_closing( _m, cv::Size(1,3), iterations);
//    // Find the biggest connected piece in the center
//    flood_from_center( _m);
//    // Find Hough lines and construct the board from them
//    cv::Mat boardImg;
//    _board = find_board( _m, boardImg);
//    if ( board_valid( _board, cv::contourArea(whole_img_quad(_small)))) {
//        boards.push_back( _board);
//        if (boards.size() > N_BOARDS) { boards.erase( boards.begin()); }
//        //_board = smallest_board( boards);
//        _board = avg_quad( boards);
//        draw_contour( _small, _board, cv::Scalar(255,0,0,255));
//        //        _cont = std::vector<Points>( 1, _board);
//        //        cv::drawContours( small, _cont, -1, cv::Scalar(255,0,0,255));
//    }
//    //cv::cvtColor( small, small, cv::COLOR_RGB2BGR);
//    UIImage *res = MatToUIImage( _small);
//    return res;
//} // findBoard()

@end





























