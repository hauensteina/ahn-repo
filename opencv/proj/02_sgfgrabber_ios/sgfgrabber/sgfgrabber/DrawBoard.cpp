//
//  DrawBoard.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-12-14.
//  Copyright © 2017 AHN. All rights reserved.
//

#include "DrawBoard.hpp"


// Draw a board from a list of B,W,E
//---------------------------------------------------------------
void DrawBoard::draw( std::vector<int> diagram)
{
    int x,y;
    RLOOP (board_sz) {
        b2xy( r, 0, x, y);
        cv::Point p1( x,y);
        b2xy( r, 18, x, y);
        cv::Point p2( x,y);
        int color = 0; int width = 1;
        cv::line( dst, p1, p2, color, width, CV_AA);
    }
    CLOOP (board_sz) {
        b2xy( 0, c, x, y);
        cv::Point p1( x,y);
        b2xy( 18, c, x, y);
        cv::Point p2( x,y);
        int color = 0; int width = 1;
        cv::line( dst, p1, p2, color, width, CV_AA);
    }
} // draw()

// Board row and col (0-18) to pixel coord
//--------------------------------------------------------------
void DrawBoard::b2xy( int boardrow, int boardcol,
                     int &x, int &y) // out
{
    const int rightmarg = leftmarg;
    const float boardwidth  = dst.cols - leftmarg - rightmarg;
    const float boardheight = boardwidth;
    x = ROUND( leftmarg + boardcol * boardwidth / (board_sz-1));
    y = ROUND( topmarg + boardrow * boardheight / (board_sz-1));
} // b2xy()
