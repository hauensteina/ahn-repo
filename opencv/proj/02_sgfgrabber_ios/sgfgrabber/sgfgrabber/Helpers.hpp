//
//  Helpers.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-09.
//  Copyright © 2018 AHN. All rights reserved.
//

// Collection of app specific standalone C++ functions

#ifndef Helpers_hpp
#define Helpers_hpp
#ifdef __cplusplus

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <regex>

#include "Common.hpp"
#include "Clust1D.hpp"

// Apply inverse thresh and dilate grayscale image.
//-------------------------------------------------------------------------------------------
inline void thresh_dilate( const cv::Mat &img, cv::Mat &dst, int thresh = 8)
{
    cv::adaptiveThreshold( img, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          5 /* 11 */ ,  // neighborhood_size
                          thresh);  // threshold
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate( dst, dst, element );
}

// Convert vector of int diagram to sgf.
// sgf coordinates range a,b,c,..,i,j,...,s
// column first, row second
// Example:
/*
 (;GM[1]
 GN[<game_name>]
 FF[4]
 CA[UTF-8]
 AP[sgfgrabber]
 RU[Chinese]
 SZ[19]
 KM[7.5]
 GC[Super great game]
 PW[w]
 PB[ahnb]
 AB[aa][ba][ja][sa][aj][jj][sj][as][js][ss])
*/
//----------------------------------------------------------------------------------
inline std::string generate_sgf( const std::string &title, const std::vector<int> diagram,
                         double komi=7.5)
{
    const int BUFSZ = 10000;
    char buf[BUFSZ+1];
    if (!SZ(diagram)) return "";
    int boardsz = ROUND( sqrt( SZ(diagram)));
    snprintf( buf, BUFSZ,
             "(;GM[1]"
             " GN[%s]"
             " FF[4]"
             " CA[UTF-8]"
             " AP[sgfgrabber]"
             " RU[Chinese]"
             " SZ[%d]"
             " KM[%f]",
             title.c_str(), boardsz, komi);

    std::string moves="";
    ISLOOP (diagram) {
        int row = i / boardsz;
        int col = i % boardsz;
        char ccol = 'a' + col;
        char crow = 'a' + row;
        std::string tag;
        if (diagram[i] == WWHITE) { tag = "AW"; }
        else if (diagram[i] == BBLACK) { tag = "AB"; }
        else continue;
        moves += tag + "[" + ccol + crow + "]";
    }
    return buf + moves + ")\n";
} // generate_sgf()

// e.g for board size, call get_sgf_tag( sgf, "SZ")
//----------------------------------------------------------------------------------------
inline std::string get_sgf_tag( const std::string &sgf, const std::string &tag)
{
    std::string res;
    std::regex re_tag( tag + "\\[[^\\]]*\\]");
    std::smatch m;
    std::regex_search( sgf, m, re_tag);
    if (!SZ(m)) {
        return "";
    }
    std::string mstr = m[0];
    std::vector<std::string> parts, parts1;
    str_split( mstr, parts, '[');
    str_split( parts[1], parts1, ']');
    res = parts1[0];
    return res;
} // get_sgf_tag()

// Look for AB[ab][cd] or AW[ab]... and transform into a linear vector
// of ints
//-----------------------------------------------------------
inline std::vector<int> sgf2vec( const std::string &sgf_)
{
    //PLOG("==========\n");
    const int NONE = 0;
    const int AB = 1;
    const int AW = 2;
    std::string sgf = std::regex_replace( sgf_, std::regex("\\s+"), "" ); // no whitespace
    int boardsz = std::stoi( get_sgf_tag( sgf, "SZ"));
    std::vector<int> res( boardsz * boardsz, EEMPTY);
    if (SZ(sgf) < 3) return res;
    char window[4];
    window[0] = sgf[0];
    window[1] = sgf[1];
    window[2] = sgf[2];
    window[3] = '\0';
    int i;
    auto shiftwin = [&i,&window,&sgf](){window[0] = window[1]; window[1] = window[2]; window[2] = sgf[i++];};
    int mode = NONE;
    for (i=3; i < SZ(sgf); ) {
        std::string tstr(window);
        if (window[2] != '[') {
            mode = NONE;
            shiftwin();
            continue;
        }
        else if (tstr == "AB[" || mode == AB) {
            mode = AB;
            if (i+2 > SZ(sgf)) break;
            int col = sgf[i] - 'a';
            shiftwin();
            int row = sgf[i] - 'a';
            shiftwin();
            int idx = col + row * boardsz;
            res[idx] = BBLACK;
            //PLOG("B at %c%c\n",col+'a',row+'a');
            shiftwin(); shiftwin();
        }
        else if (tstr == "AW[" || mode == AW) {
            mode = AW;
            if (i+2 > SZ(sgf)) break;
            int col = sgf[i] - 'a';
            shiftwin();
            int row = sgf[i] - 'a';
            shiftwin();
            int idx = col + row * boardsz;
            res[idx] = WWHITE;
            //PLOG("W at %c%c\n",col+'a',row+'a');
            shiftwin(); shiftwin();
        }
        else {
            mode = NONE;
            shiftwin();
        }
    } // for
    return res;
} // sgf2vec

// Draw sgf on a square single channel Mat
//----------------------------------------------------------------------
inline void draw_sgf( const std::string &sgf_, cv::Mat &dst, int width)
{
    std::string sgf = std::regex_replace( sgf_, std::regex("\\s+"), "" ); // no whitespace
    int height = width;
    dst = cv::Mat( height, width, CV_8UC1);
    dst = 180;
    int boardsz = 19;
    std::vector<int> diagram( boardsz*boardsz,EEMPTY);
    int marg = width * 0.05;
    int innerwidth = width - 2*marg;
    if (SZ(sgf) > 3) {
        boardsz = std::stoi( get_sgf_tag( sgf, "SZ"));
        diagram = sgf2vec( sgf);
    }
    auto rc2p = [boardsz, innerwidth, marg](int row, int col) {
        cv::Point res;
        float d = innerwidth / (boardsz-1.0) ;
        res.x = ROUND( marg + d*col);
        res.y = ROUND( marg + d*row);
        return res;
    };
    // Draw the lines
    ILOOP (boardsz) {
        cv::Point p1 = rc2p( i, 0);
        cv::Point p2 = rc2p( i, boardsz-1);
        cv::Point q1 = rc2p( 0, i);
        cv::Point q2 = rc2p( boardsz-1, i);
        cv::line( dst, p1, p2, cv::Scalar(0,0,0), 1, CV_AA);
        cv::line( dst, q1, q2, cv::Scalar(0,0,0), 1, CV_AA);
    }
    // Draw the hoshis
    int r = ROUND( 0.25 * innerwidth / (boardsz-1.0));
    cv::Point p;
    p = rc2p( 3, 3);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 15, 15);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 3, 15);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 15, 3);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 9, 9);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 9, 3);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 3, 9);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 9, 15);
    cv::circle( dst, p, r, 0, -1);
    p = rc2p( 15, 9);
    cv::circle( dst, p, r, 0, -1);

    // Draw the stones
    int rad = ROUND( 0.5 * innerwidth / (boardsz-1.0));
    ISLOOP (diagram) {
        int r = i / boardsz;
        int c = i % boardsz;
        cv::Point p = rc2p( r,c);
        if (diagram[i] == WWHITE) {
            cv::circle( dst, p, rad, 255, -1);
            cv::circle( dst, p, rad, 0, 2);
        }
        else if (diagram[i] == BBLACK) {
            cv::circle( dst, p, rad, 0, -1);
        }
    } // ISLOOP
} // draw_sgf()

// Reject board if opposing lines not parallel
// or adjacent lines not at right angles
//-----------------------------------------------------------
inline bool board_valid( Points2f board, const cv::Mat &img)
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
} // board_valid()

// Replace close clusters of vert lines by their average.
//-----------------------------------------------------------------------------------
inline void dedup_verticals( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
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
} // dedup_verticals()

// Replace close clusters of horiz lines by their average.
//-----------------------------------------------------------------------------------
inline void dedup_horizontals( std::vector<cv::Vec2f> &lines, const cv::Mat &img)
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
} // dedup_horizontals()

// Find a line close to the middle with roughly median theta.
// The lines should be sorted by rho.
//--------------------------------------------------------------------------
inline int good_center_line( const std::vector<cv::Vec2f> &lines)
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

// Adjacent lines should have similar slope
//-----------------------------------------------------------------------------
inline void filter_vert_lines( std::vector<cv::Vec2f> &vlines)
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
    vlines = good;
} // filter_vert_lines()

// Adjacent lines should have similar slope
//-----------------------------------------------------------------
inline void filter_horiz_lines( std::vector<cv::Vec2f> &hlines)
{
    const double eps = 1.1;
    std::sort( hlines.begin(), hlines.end(), [](cv::Vec2f &a, cv::Vec2f &b) { return a[0] < b[0]; });
    int med_idx = good_center_line( hlines);
    if (med_idx < 0) return;
    double theta = hlines[med_idx][1];
    // Going up and down, theta should not change abruptly
    std::vector<cv::Vec2f> good;
    good.push_back( hlines[med_idx]);
    const double EPS = eps * PI/180;
    double prev_theta;
    // down
    prev_theta = theta;
    for (int i = med_idx+1; i < SZ(hlines); i++ ) {
        if (fabs( hlines[i][1] - prev_theta) < EPS) {
            good.push_back( hlines[i]);
            prev_theta = hlines[i][1];
        }
    }
    // up
    prev_theta = theta;
    for (int i = med_idx-1; i >= 0; i-- ) {
        if (fabs( hlines[i][1] - prev_theta) < EPS) {
            good.push_back( hlines[i]);
            prev_theta = hlines[i][1];
        }
    }
    hlines = good;
} // filter_horiz_lines()

#endif /* __clusplus */
#endif /* Helpers_hpp */
