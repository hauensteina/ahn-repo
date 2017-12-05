//
//  BlackWhiteEmpty.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-16.
//  Copyright © 2017 AHN. All rights reserved.
//

// Classify board intersection into Black, White, Empty

#ifndef BlackWhiteEmpty_hpp
#define BlackWhiteEmpty_hpp

#include <iostream>
#include "Common.hpp"
#include "Ocv.hpp"

class BlackWhiteEmpty
//=====================
{
public:
    enum { BBLACK=0, EEMPTY=1, WWHITE=2, DONTKNOW=3 };
    
    //----------------------------------------------------------------------------------
    inline static std::vector<int> classify( const cv::Mat &img, // small, color
                                            const Points2f &intersections,
                                            float dx, // approximate dist between lines
                                            float dy)
    {
        cv::Mat gray;
        cv::cvtColor( img, gray, cv::COLOR_BGR2GRAY);
        std::vector<int> res(SZ(intersections), DONTKNOW);
        
        cv::Mat gray_normed;
        normalize_plane( gray, gray_normed);
        //cv::Mat &gray_normed(gray);
        //equalizeHist( gray, gray_normed );
        
        // Compute features for each board intersection
        std::vector<float> black_features;
        get_black_features( gray_normed, intersections, dx, dy, black_features);

        std::vector<float> center_brightness;
        center_bright( gray_normed, intersections, dx, dy, center_brightness);

        // Black stones
        float black_median = vec_median( black_features);
        ISLOOP( black_features) {
            //float black_median = get_neighbor_med( i, 3, black_features);
            float bthresh = black_median * 0.66; // larger means more Black stones
            if (black_features[i] < bthresh /* && black_features[i] - tt_feat[i] < 8 */ ) {
                res[i] = BBLACK;
            }
        }
        // White places
        ISLOOP( black_features) {
            //float black_median = get_neighbor_med( i, 3, black_features);
            float wthresh = black_median * 1.2; // larger means less White stones
            if (black_features[i] > wthresh  /* && black_features[i] - center_brightness[i] < 0 */ ) {
                res[i] = WWHITE;
            }
        }
        // Empty places
        ISLOOP( res) {
            if (res[i] != WWHITE && res[i] != BBLACK) {
                res[i] = EEMPTY;
            }
        }
        return res;
    } // classify()
    
private:
    
    // Get median of a neighborhood of size n around idx
    //-------------------------------------------------------------------------------------------------
    inline static float get_neighbor_med( int idx, int n, const std::vector<float> &feat )
    {
        int board_sz = sqrt(SZ(feat));
        std::vector<float> vals;
        int row = idx / board_sz;
        int col = idx % board_sz;
        for (int r = row-n; r <= row+n; r++) {
            if (r < 0 || r >= board_sz) continue;
            for (int c = col-n; c <= col+n; c++) {
                if (c < 0 || c >= board_sz) continue;
                int pos = r * board_sz + col;
                vals.push_back( feat[pos]);
            }
        }
        return vec_median( vals);
    }
    
    // Average pixel value around center of each intersection is black indicator.
    //---------------------------------------------------------------------------------
    inline static void get_black_features( const cv::Mat &img, // gray
                                          const Points2f &intersections,
                                          float dx_, float dy_,
                                          std::vector<float> &res )
    {        
        int dx = ROUND( dx_/4.0);
        int dy = ROUND( dy_/4.0);
        
        res.clear();
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            if (0 <= rect.x &&
                0 <= rect.width &&
                rect.x + rect.width <= img.cols &&
                0 <= rect.y &&
                0 <= rect.height &&
                rect.y + rect.height <= img.rows)
            {
                cv::Mat hood = cv::Mat( img, rect);
                float brightness = channel_median( hood);
                res.push_back( brightness);
            }
        } // for intersections
    } // get_black_features()

    // If there are contours, it's probably empty
    //----------------------------------------------------------------------------------------
    inline static void get_empty_features( const cv::Mat &img, // gray
                                          const Points2f &intersections,
                                          float dx_, float dy_,
                                          std::vector<float> &res )
    {
        int dx = ROUND( dx_/4.0);
        int dy = ROUND( dy_/4.0);
        
        cv::Mat edges;
        cv::Canny( img, edges, 30, 70);
        
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            if (0 <= rect.x &&
                0 <= rect.width &&
                rect.x + rect.width <= img.cols &&
                0 <= rect.y &&
                0 <= rect.height &&
                rect.y + rect.height <= img.rows)
            {
                cv::Mat hood = cv::Mat( edges, rect);
                float area = hood.rows * hood.cols;
                cv::Scalar ssum = cv::sum( hood);
                float crossness = ssum[0] / area;
                res.push_back( crossness);
            }
        } // for intersections
    } // get_empty_features()

    // Brightness in a 3x3 window
    //---------------------------------------------------------------------
    inline static void center_bright( const cv::Mat &img, // gray
                                     const Points2f &intersections,
                                     float dx_, float dy_,
                                     std::vector<float> &res )
    {
        int dx = 1;
        int dy = 1;

        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            if (0 <= rect.x &&
                0 <= rect.width &&
                rect.x + rect.width <= img.cols &&
                0 <= rect.y &&
                0 <= rect.height &&
                rect.y + rect.height <= img.rows)
            {
                cv::Mat hood = cv::Mat( img, rect);
                float brightness = channel_median( hood);
                res.push_back( brightness);
            }
        } // for intersections
    } // center_bright()

}; // class BlackWhiteEmpty
    

#endif /* BlackWhiteEmpty_hpp */
