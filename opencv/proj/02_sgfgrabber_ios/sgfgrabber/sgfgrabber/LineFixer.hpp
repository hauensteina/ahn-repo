//
//  LineFixer.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

#ifndef LineFixer_hpp
#define LineFixer_hpp

#include "Ocv.hpp"

class LineFixer
//================
{
public:
    LineFixer() {}
    void fix( const std::vector<cv::Vec4f> &lines, const std::vector<Points> &clusters,
             std::vector<cv::Vec4f> &lines_out);
}; // class LineFixer


#endif /* LineFixer_hpp */
