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

//-----------------------------------------------------------------------------
inline std::string get_sgf_token( const std::string &sgf, const std::string &tag)
{
    std::string res;
    //std::regex tag("(\\+|-)?[[:alnum:]]+");
    //std::regex tag("(" + tag + "\\[.*\\])");
    //std::regex re_tag( tag + "\\[.*\\]"); // ECMA
    std::regex re_tag( tag);
    std::smatch m;
    std::regex_search( sgf, m, re_tag);
    if (!SZ(m)) {
        return "";
    }
    std::string mstr = m[0];
    int tt = 42;
    return res;
}

// Draw sgf on a square one channel Mat
//----------------------------------------------------------------------
inline void draw_sgf( const std::string &sgf, cv::Mat &dst, int width)
{
    int height = width;
    dst = cv::Mat( height, width, CV_8UC1);
    int marg = width * 0.05;
    int innerwidth = width - 2*marg;
    int boardsz = std::stoi( get_sgf_token( sgf, "SZ"));
    
//    Points2f dummy;
//    get_intersections_from_corners( _corners_zoomed, _board_sz, dummy, _dx, _dy);
//    int dx = ROUND( _dx/4.0);
//    int dy = ROUND( _dy/4.0);
//    ISLOOP (_diagram) {
//        cv::Point p(ROUND(_intersections_zoomed[i].x), ROUND(_intersections_zoomed[i].y));
//        cv::Rect rect( p.x - dx,
//                      p.y - dy,
//                      2*dx + 1,
//                      2*dy + 1);
//        cv::rectangle( drawing, rect, cv::Scalar(0,0,255,255));
//        if (_diagram[i] == BBLACK) {
//            draw_point( p, drawing, 2, cv::Scalar(0,255,0,255));
//        }
//        else if (_diagram[i] == WWHITE) {
//            draw_point( p, drawing, 5, cv::Scalar(255,0,0,255));
//        }
//    }
} // draw_sgf()

#endif /* __clusplus */
#endif /* Helpers_hpp */
