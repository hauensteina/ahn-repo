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

#endif /* __clusplus */
#endif /* Helpers_hpp */
