//
//  LineFinder.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

// Class to find horizontal and vertical lines in a cloud of points.

#include "LineFinder.hpp"

//----------------------------------------------------------------------
void LineFinder::find_lines( std::vector<cv::Vec4f> &horizontal_lines,
                            std::vector<cv::Vec4f> &vertical_lines)
{
    cv::Vec4f hslope, vslope;
    find_slopes( hslope, vslope);
    ISLOOP (m_cloud) {
        std::cout << "xx";
    }
}

// Find two line segments representing the typical slope of horiz and vert
// lines. Used later for distance sorting.
//-------------------------------------------------------------------------
void LineFinder::find_slopes( cv::Vec4f &hslope, cv::Vec4f &vslope)
{
    // Find Hough lines in the detected intersections and stones
    cv::Mat canvas = cv::Mat::zeros( m_imgSize, CV_8UC1 );
    ISLOOP (m_cloud) {
        draw_point( m_cloud[i], canvas,1, cv::Scalar(255));
    }
    std::vector<cv::Vec2f> lines;
    std::vector<std::vector<cv::Vec2f> > horiz_vert_other_lines;
    std::vector<int> vote_thresholds = { 10, 9, 8, 7, 6, 5 };
    
    ISLOOP (vote_thresholds) {
        int votes = vote_thresholds[i];
        std::cerr << "trying " << votes << " hough line votes\n";
        HoughLines(canvas, lines, 1, CV_PI/180, votes, 0, 0 );
        
        // Separate horizontal, vertical, and other lines
        horiz_vert_other_lines = partition( lines, 3,
                                           [](cv::Vec2f &line) {
                                               const float thresh = 10.0;
                                               float theta = line[1] * (180.0 / CV_PI);
                                               if (fabs(theta - 180) < thresh) return 1;
                                               else if (fabs(theta) < thresh) return 1;
                                               else if (fabs(theta-90) < thresh) return 0;
                                               else return 2;
                                           });
        // Sort by Rho (distance of line from origin)
        std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
        std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
        if (hlines.size() >= 2 && vlines.size() >= 2) break;
    }
    std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
    std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
    hslope = avg_slope_line( hlines);
    vslope = avg_slope_line( vlines);
}
