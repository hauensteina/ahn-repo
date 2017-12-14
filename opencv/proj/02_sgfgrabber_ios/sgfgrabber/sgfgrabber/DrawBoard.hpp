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
    void draw( const cv::Mat &img, cv::Mat &dst, std::vector<int> diagram,
              int topmarg = 0, int leftmarg = 0, int board_sz = 19);
private:
    void b2xy( const cv::Mat &img, int boardrow, int boardcol,
              int topmarg, int leftmarg, int board_sz,
              int &x, int &y); // out

}; // class DrawBoard


#endif /* DrawBoard_hpp */
