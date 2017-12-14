//
//  DrawBoard.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-12-14.
//  Copyright © 2017 AHN. All rights reserved.
//

#ifndef DrawBoard_hpp
#define DrawBoard_hpp

#include <iostream>
#include "Common.hpp"
#include "Ocv.hpp"

class DrawBoard
//=================
{
public:
    // Constructor
    DrawBoard( cv::Mat &dst, Points2f corners, int board_sz):
    m_dst(dst), m_corners(corners), m_board_sz(board_sz)
    {}
    // Draw board and position
    void draw( std::vector<int> diagram);
private:
    cv::Mat   &m_dst;
    const Points2f m_corners;
    const int m_board_sz;
    
    // Board coords to screen coords
    void b2xy( int boardrow, int boardcol,
              int &x, int &y); // out

}; // class DrawBoard


#endif /* DrawBoard_hpp */
