//
//  LineFinder.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

// Class to find horizontal and vertical lines in a cloud of points.

#ifndef LineFinder_hpp
#define LineFinder_hpp

#include "Ocv.hpp"

class LineFinder
//================
{
public:
    LineFinder( Points cloud, int boardsize, cv::Size imgSize) {
        m_cloud = cloud;
        m_boardsize = boardsize;
        m_imgSize = imgSize;
    }
    void find_lines( std::vector<cv::Vec4f> &horizontal_lines, std::vector<cv::Vec4f> &vertical_lines);
//private:
    // Methods
    //---------
    void find_slopes( cv::Vec4f &hslope, cv::Vec4f &vslope);
    // Data
    //-------
    Points m_cloud;
    int m_boardsize;
    cv::Size m_imgSize;
}; // class LineFinder


#endif /* LineFinder_hpp */
