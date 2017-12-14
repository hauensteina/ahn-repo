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
    DrawBoard( cv::Mat &dst_, int topmarg_, int leftmarg_, int board_sz_):
    dst(dst_), topmarg(topmarg_), leftmarg(leftmarg_), board_sz(board_sz_)
    {}
    // Draw board and position
    void draw( std::vector<int> diagram);
private:
    cv::Mat &dst;
    const int topmarg;
    const int leftmarg;
    const int board_sz;
    
    // Board coords to screen coords
    void b2xy( int boardrow, int boardcol,
              int &x, int &y); // out

}; // class DrawBoard


#endif /* DrawBoard_hpp */
