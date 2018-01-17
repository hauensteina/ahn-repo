//
//  Helpers.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-09.
//  Copyright © 2018 AHN. All rights reserved.
//

// Collection of app specific standalone C++ functions

#ifndef Helpers_hpp
#define Helpers_hpp
#ifdef __cplusplus

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "Common.hpp"

// Apply inverse thresh and dilate grayscale image.
//-------------------------------------------------------------------------------------------
inline void thresh_dilate( const cv::Mat &img, cv::Mat &dst, int thresh = 8)
{
    cv::adaptiveThreshold( img, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          5 /* 11 */ ,  // neighborhood_size
                          thresh);  // threshold
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate( dst, dst, element );
}

////----------------------------------------------------------------------------------
//std::string generate_sgf( const std::string &title, const std::vector<int> diagram)
//{
//    if (!SZ(diagram)) return "";
//    int boardsz = ROUND( sqrt( SZ(diagram)));
//    return "";
//    
//} // generate_sgf()

#endif /* __clusplus */
#endif /* Helpers_hpp */
