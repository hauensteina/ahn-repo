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
    void get_lines( std::vector<cv::Vec4f> &horizontal_lines, std::vector<cv::Vec4f> &vertical_lines);
//private:
    // Methods
    //---------
    void find_slopes( cv::Vec4f &hslope, cv::Vec4f &vslope);
    void find_rhythm( const std::vector<Points> &clusters,
                                 float &wavelength,
                                 float &delta_wavelength,
                                 float &slope,
                                 float &median_rho,
                                 std::vector<cv::Vec4f> &lines,
                                 std::vector<float> &dists
                                 );
    void find_lines( int max_rho,
                    float wavelength_,
                    float delta_wavelength,
                    float slope,
                    float median_rho,
                    std::vector<cv::Vec4f> &lines);
    void best_two_horiz_lines( int &idx1, int &idx2,     // index in m_horizontal_clusters
                              cv::Vec4f &line1,
                              cv::Vec4f &line2,
                              float &rho1, float &rho2,  // distance from top
                              float &d1, float &d2);     // distance between intersections on that line
    float dy_rat( cv::Vec2f &ratline, float &dy);

    // Data
    //-------
    Points m_cloud;
    int m_boardsize;
    cv::Size m_imgSize;
    std::vector<Points> m_horizontal_clusters;
    std::vector<cv::Vec2f> m_horizontal_lines;
    std::vector<cv::Vec2f> m_vertical_lines;
    std::vector<Points> m_vertical_clusters;
    float m_wavelen_h, m_wavelen_v;
}; // class LineFinder


#endif /* LineFinder_hpp */
