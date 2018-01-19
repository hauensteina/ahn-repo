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
    PLOG("==========\n");
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
            //window[0] = window[1]; window[1] = window[2]; window[2] = sgf[i++];
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
            PLOG("B at %c%c\n",col+'a',row+'a');
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
            PLOG("W at %c%c\n",col+'a',row+'a');
            shiftwin(); shiftwin();
        }
        else {
            mode = NONE;
            shiftwin();
        }
    } // for
    return res;
} // sgf2vec

// Draw sgf on a square one channel Mat
//----------------------------------------------------------------------
inline void draw_sgf( const std::string &sgf_, cv::Mat &dst, int width)
{
    std::string sgf = std::regex_replace( sgf_, std::regex("\\s+"), "" ); // no whitespace
    int height = width;
    dst = cv::Mat( height, width, CV_8UC1);
    std::vector<int> diagram(19*19,EEMPTY);
    int marg = width * 0.05;
    int innerwidth = width - 2*marg;
    if (SZ(sgf) > 3) {
        int boardsz = std::stoi( get_sgf_tag( sgf, "SZ"));
        diagram = sgf2vec( sgf);
    }
    
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
