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
    LineFinder() {
        m_boardsize = 0;
        m_imgSize = cv::Size(0,0);
    }
    LineFinder( Points cloud, int boardsize, cv::Size imgSize) {
        m_cloud = cloud;
        m_boardsize = boardsize;
        m_imgSize = imgSize;
    }

    // Methods
    //---------
    void cluster();
    float dy_rat( cv::Vec2f &ratline, float &dy, int &idx);

    // Data
    //-------
    Points m_cloud;
    int m_boardsize;
    cv::Size m_imgSize;
    std::vector<Points> m_horizontal_clusters;
    std::vector<cv::Vec2f> m_horizontal_lines;
}; // class LineFinder


#endif /* LineFinder_hpp */
