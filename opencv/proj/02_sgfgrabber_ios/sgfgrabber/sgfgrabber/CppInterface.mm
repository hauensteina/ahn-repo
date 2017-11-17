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
//@property Points intersections; // locations of line intersections (81,361)
//@property int delta_v; // approx vertical line dist
//@property int delta_h; // approx horiz line dist
@property Points stone_or_empty; // places where we suspect stones or empty
//@property std::vector<Points> horizontal_clusters; // Result of Hough lines and clustering
//@property std::vector<Points> vertical_clusters;
@property std::vector<cv::Vec4f> horizontal_lines;
//@property float wavelen_h;
//@property float delta_wavelen_h;
//@property float slope_h;
//@property float delta_slope_h;
//@property float median_rho_h;
@property std::vector<cv::Vec4f> vertical_lines;
//@property float wavelen_v;
//@property float delta_wavelen_v;
//@property float slope_v;
//@property float delta_slope_v;
//@property float median_rho_v;
@property LineFinder finder;

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

- (UIImage *) f00_adaptive_thresh:(UIImage *)img
{
    g_app.mainVC.lbDbg.text = @"00";
    UIImageToMat( img, _m);
    resize( _m, _small, 350);
    cv::cvtColor( _small, _gray, cv::COLOR_BGR2GRAY);
    //cv::cvtColor( _small, _small, cv::COLOR_BGR2RGB);
    adaptiveThreshold( _gray, _m, 100, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                      3, // neighborhood_size
                      4); // constant to add. 2 to 6 is the viable range
    UIImage *res = MatToUIImage( _m);
    return(res);
}

//--------------------------
- (UIImage *) f01_closing
{
    g_app.mainVC.lbDbg.text = @"01";
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
    g_app.mainVC.lbDbg.text = @"05";
    Points pts, crosses;
    find_empty_places( _gray, pts); // , crosses); // has to be first
    find_stones( _gray, pts);
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

// Find line candidates from stones and intersections
//------------------------------------------------------------------------
- (UIImage *) f06_find_lines
{
    g_app.mainVC.lbDbg.text = @"06";
    NSString *func = @"f06_find_h_lines()";
    if (_stone_or_empty.size() < _board_sz) {
        NSLog( @"%@: not enough points", func);
        return MatToUIImage( _gray);
    }
    _finder = LineFinder( _stone_or_empty, _board_sz, _gray.size() );
    //cv::Vec4f hslope, vslope;
    _finder.get_lines( _horizontal_lines, _vertical_lines);
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    ISLOOP( _finder.m_vertical_clusters) {
        Points cl = _finder.m_vertical_clusters[i];
        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
        draw_points( cl, drawing, 2, color);
    }
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f06_find_lines()

// Show the lines we found
//------------------------------------------------------------------------
- (UIImage *) f07_show_horiz_lines
{
    g_app.mainVC.lbDbg.text = @"07";
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    draw_lines( _horizontal_lines, drawing, cv::Scalar(0,0,255));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f07_show_horiz_lines()

// Show the lines we found
//------------------------------------------------------------------------
- (UIImage *) f08_show_vert_lines
{
    g_app.mainVC.lbDbg.text = @"08";
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    draw_lines( _vertical_lines, drawing, cv::Scalar(0,0,255));
    UIImage *res = MatToUIImage( drawing);
    return res;
} // f08_show_vert_lines()

// Improve line candidates by interpolation
//--------------------------------------------
- (UIImage *) f09_clean_horiz_lines
{
    g_app.mainVC.lbDbg.text = @"09";
    LineFixer fixer;
    fixer.fix( _horizontal_lines, _finder.m_horizontal_clusters, _horizontal_lines );
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    draw_lines( _horizontal_lines, drawing, cv::Scalar(0,0,255));
    draw_points( _stone_or_empty, drawing, 2, cv::Scalar(0,255,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
}
//--------------------------------------------
- (UIImage *) f10_clean_vert_lines
{
    g_app.mainVC.lbDbg.text = @"10";
    LineFixer fixer;
    fixer.fix( _vertical_lines, _finder.m_vertical_clusters, _vertical_lines );
    
    // Show results
    cv::Mat drawing;
    cv::cvtColor( _gray, drawing, cv::COLOR_GRAY2RGB);
    draw_lines( _vertical_lines, drawing, cv::Scalar(0,0,255));
    draw_points( _stone_or_empty, drawing, 2, cv::Scalar(0,255,0));
    UIImage *res = MatToUIImage( drawing);
    return res;
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


// Classify intersections into b,w,empty
//----------------------------------------
- (UIImage *) f11_classify
{
    g_app.mainVC.lbDbg.text = @"11";
    Points intersections;
    std::vector<int> diagram =
    BlackWhiteEmpty::get_diagram( _small,
                                 _horizontal_lines, _finder.m_wavelen_h,
                                 _vertical_lines, _finder.m_wavelen_v,
                                 _board_sz,
                                 _board_zoomed,
                                 intersections);

    cv::Mat drawing;
    drawing = _small.clone();
    int dx = round(_finder.m_wavelen_h/4.0);
    int dy = round(_finder.m_wavelen_h/4.0);
    // Black stones //@@@
    ISLOOP (diagram) {
        cv::Point p = intersections[i];
        cv::Rect rect( p.x - dx,
                      p.y - dy,
                      2*dx+1,
                      2*dy+1);
        cv::rectangle( drawing, rect, cv::Scalar(255,0,0,255));
        if (diagram[i] == BlackWhiteEmpty::BBLACK) {
            draw_point( p, drawing, 1, cv::Scalar(255,255,255,255));
        }
        else if (diagram[i] != BlackWhiteEmpty::EEMPTY) {
            draw_point( p, drawing, 2, cv::Scalar(0,0,0,255));
        }
    }

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
    if ( board_valid( _board, cv::contourArea(whole_img_quad(_small)))) {
        boards.push_back( _board);
        if (boards.size() > N_BOARDS) { boards.erase( boards.begin()); }
        //_board = smallest_board( boards);
        _board = avg_quad( boards);
        draw_contour( _small, _board, cv::Scalar(255,0,0,255));
//        _cont = std::vector<Points>( 1, _board);
//        cv::drawContours( small, _cont, -1, cv::Scalar(255,0,0,255));
    }
    //cv::cvtColor( small, small, cv::COLOR_RGB2BGR);
    UIImage *res = MatToUIImage( _small);
    return res;
}

@end





























