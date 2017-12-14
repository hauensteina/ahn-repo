//
//  DrawBoard.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-12-14.
//  Copyright © 2017 AHN. All rights reserved.
//

#include "DrawBoard.hpp"


// Draw a board from a list of B,W,E
//-------------------------------------------------------------------------------
void DrawBoard::draw( const cv::Mat &img, cv::Mat &dst, std::vector<int> diagram,
                     int topmarg, int leftmarg, int board_sz)
{
    int x,y;
    RLOOP (board_sz) {
        b2xy( img, r, 0, topmarg, leftmarg, board_sz, x, y);
        cv::Point p1( x,y);
        b2xy( img, r, 18, topmarg, leftmarg, board_sz, x, y);
        cv::Point p2( x,y);
        int color = 0; int width = 1;
        cv::line( dst, p1, p2, color, width, CV_AA);
    }
    CLOOP (board_sz) {
        b2xy( img, 0, c, topmarg, leftmarg, board_sz, x, y);
        cv::Point p1( x,y);
        b2xy( img, 18, c, topmarg, leftmarg, board_sz, x, y);
        cv::Point p2( x,y);
        int color = 0; int width = 1;
        cv::line( dst, p1, p2, color, width, CV_AA);
    }
}

// Board row and col (0-18) to pixel coord
//--------------------------------------------------------------
void DrawBoard::b2xy( const cv::Mat &img, int boardrow, int boardcol,
                     int topmarg, int leftmarg, int board_sz,
                     int &x, int &y) // out
{
    const int rightmarg = leftmarg;
    const float boardwidth  = img.cols - leftmarg - rightmarg;
    const float boardheight = boardwidth;
    x = ROUND( leftmarg + boardcol * boardwidth / (board_sz-1));
    y = ROUND( topmarg + boardrow * boardheight / (board_sz-1));
}
