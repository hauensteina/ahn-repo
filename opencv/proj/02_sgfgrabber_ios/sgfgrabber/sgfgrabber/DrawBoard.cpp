//
//  DrawBoard.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-12-14.
//  Copyright © 2017 AHN. All rights reserved.
//

#include "DrawBoard.hpp"
#include "BlackWhiteEmpty.hpp"


// Draw a board from a list of B,W,E
//--------------------------------------------------
void DrawBoard::draw( std::vector<int> diagram)
{
    int x,y;
    m_dst = 127;
    // The lines
    RLOOP (m_board_sz) {
        b2xy( r, 0, x, y);
        cv::Point p1( x,y);
        b2xy( r, 18, x, y);
        cv::Point p2( x,y);
        cv::Scalar color(0); int width = 1;
        cv::line( m_dst, p1, p2, color, width, CV_AA);
    }
    CLOOP (m_board_sz) {
        b2xy( 0, c, x, y);
        cv::Point p1( x,y);
        b2xy( 18, c, x, y);
        cv::Point p2( x,y);
        cv::Scalar color(0); int width = 1;
        cv::line( m_dst, p1, p2, color, width, CV_AA);
    }
    // The stones
    int rightmarg = m_dst.cols - m_corners[1].x;
    int leftmarg =  m_corners[0].x;
    const float boardwidth  = m_dst.cols - leftmarg - rightmarg;
    int r = ROUND( 0.5 * boardwidth / (m_board_sz - 1)) +1;
    ISLOOP (diagram) {
        int row = i / m_board_sz;
        int col = i % m_board_sz;
        b2xy( row, col, x, y);
        if (diagram[i] == BlackWhiteEmpty::BBLACK) {
            cv::Scalar color(0);
            cv::circle( m_dst, cv::Point(x,y), r, color, -1);
        }
        else if (diagram[i] == BlackWhiteEmpty::WWHITE) {
            cv::Scalar color(255);
            cv::circle( m_dst, cv::Point(x,y), r, color, -1);
        }
    }
    // Warp board to the corners we got
    //const float leftmarg  = m_corners[0].x;
    const float topmarg   = m_corners[0].y;
    //const float rightmarg = leftmarg;
    //const float boardwidth = m_dst.cols - leftmarg - rightmarg;
    const float right  = leftmarg + boardwidth;
    const float bottom = topmarg + boardwidth;
    Point2f tl( leftmarg, topmarg);
    Point2f tr( right, topmarg);
    Point2f br( right, bottom);
    Point2f bl( leftmarg, bottom);
    Points2f sq_corners = { tl, tr, br, bl };
    cv::Mat M = cv::getPerspectiveTransform( sq_corners, m_corners);
    cv::warpPerspective( m_dst, m_dst, M, cv::Size( m_dst.cols, m_dst.rows));
} // draw()

// Board row and col (0-18) to pixel coord, on a square board
//---------------------------------------------------------------
void DrawBoard::b2xy( int boardrow, int boardcol,
                     int &x, int &y) // out
{
    int leftmarg = m_corners[0].x;
    int topmarg  = m_corners[0].y;
    int rightmarg = leftmarg;
    const float boardwidth  = m_dst.cols - leftmarg - rightmarg;
    const float boardheight = boardwidth;
    x = ROUND( leftmarg + boardcol * boardwidth / (m_board_sz-1));
    y = ROUND( topmarg + boardrow * boardheight / (m_board_sz-1));
} // b2xy()
