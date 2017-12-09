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

cv::Mat mat_dbg;  // debug image to viz intermediate reults

class BlackWhiteEmpty
//=====================
{
public:
    enum { BBLACK=0, EEMPTY=1, WWHITE=2, DONTKNOW=3 };
    
    //----------------------------------------------------------------------------------
    inline static std::vector<int> classify( const cv::Mat &gray, // gray, normalized
                                            const Points2f &intersections,
                                            float dx, // approximate dist between lines
                                            float dy)
    {
        std::vector<int> res(SZ(intersections), DONTKNOW);
        
        // Compute features for each board intersection
        std::vector<float> brightness;
        get_brightness( gray, intersections, dx, dy, brightness);

        //std::vector<float> center_sum;
        //get_center_sum( gray, intersections, dx, dy, center_sum);

        std::vector<float> crossness;
        get_crossness( gray, intersections, dx, dy, crossness);

        // Black stones
        float black_median = vec_median( brightness);
        ISLOOP( brightness) {
            //float black_median = get_neighbor_med( i, 3, black_features);
            //float bthresh = black_median * 0.66; // larger means more Black stones
            float bthresh = black_median * 0.5; // larger means more Black stones
            if (brightness[i] < bthresh /* && black_features[i] - tt_feat[i] < 8 */ ) {
                res[i] = BBLACK;
            }
        }
        // White places
        ISLOOP( brightness) {
            //float black_median = get_neighbor_med( i, 3, brightness);
            float wthresh = black_median * 1.0; // larger means less White stones
            if (brightness[i] > wthresh  &&  crossness[i] < 0.35  )  {
                res[i] = WWHITE;
                //PLOG( ">>>>>>> WHITE crossness %f\n", crossness[i]);
            }
        }
        // Empty places
        ISLOOP( res) {
            if (res[i] != WWHITE && res[i] != BBLACK) {
                res[i] = EEMPTY;
                //PLOG( "######## EMPTY center sum %f\n", center_sum[i]);
            }
        }
        return res;
    } // classify()
    
//private:
    
    // Check if a rectangle makes sense
    //---------------------------------------------------------------------
    inline static bool check_rect( const cv::Rect &r, int rows, int cols )
    {
        if (0 <= r.x && r.x < 1e6 &&
            0 <= r.width && r.width < 1e6 &&
            r.x + r.width <= cols &&
            0 <= r.y &&  r.y < 1e6 &&
            0 <= r.height &&  r.height < 1e6 &&
            r.y + r.height <= rows)
        {
            return true;
        }
        return false;
    }
    
//    // Get median of a neighborhood of size n around idx
//    //-------------------------------------------------------------------------------------------------
//    inline static float get_neighbor_med( int idx, int n, const std::vector<float> &feat )
//    {
//        int board_sz = sqrt(SZ(feat));
//        std::vector<float> vals;
//        int row = idx / board_sz;
//        int col = idx % board_sz;
//        for (int r = row-n; r <= row+n; r++) {
//            if (r < 0 || r >= board_sz) continue;
//            for (int c = col-n; c <= col+n; c++) {
//                if (c < 0 || c >= board_sz) continue;
//                int pos = r * board_sz + col;
//                vals.push_back( feat[pos]);
//            }
//        }
//        return vec_median( vals);
//    }
    
    // Average pixel value around center of each intersection is black indicator.
    //---------------------------------------------------------------------------------
    inline static void get_brightness( const cv::Mat &img, // gray
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
            if (check_rect( rect, img.rows, img.cols)) {
                cv::Mat hood = cv::Mat( img, rect);
                float brightness = channel_median( hood);
                res.push_back( brightness);
            }
        } // for intersections
    } // get_brightness()

    //-------------------------------------------------------------------
    inline static void get_crossness( const cv::Mat &img, // gray
                                     const Points2f &intersections,
                                     float dx_, float dy_,
                                     std::vector<float> &res ) 
    {
        //mat_dbg = img.clone();
        res.clear();
        ISLOOP (intersections) {
            float feat = cross_feature( img, intersections[i], dx_, dy_);
            res.push_back( feat);
        } // for intersections
    } // get_crossness()


    // Look whether the cross pixels are set
    //----------------------------------------------------------------------------------------
    inline static float cross_feature( const cv::Mat &img, Point2f p_, float dx_, float dy_ )
    {
        float res = 0;
        int dx = ROUND(dx_/2.0);
        int dy = ROUND(dy_/2.0);
        cv::Mat threshed;
        cv::Point p(ROUND(p_.x), ROUND(p_.y));
        cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
        if (check_rect( rect, img.rows, img.cols)) {
            cv::Mat hood = img(rect);
            inv_thresh_avg( hood, threshed);
            int mid_y = ROUND(threshed.rows / 2.0);
            int mid_x = ROUND(threshed.cols / 2.0);
            float ssum = 0;
            int n = 0;
            const int marg = 6;
            CLOOP (threshed.cols) {
                if (c < marg) continue;
                if (c >= threshed.cols - marg ) continue;
                //ssum += threshed.at<uint8_t>(mid_y, c); n++;
                ssum += threshed.at<uint8_t>(mid_y-1, c); n++;
                ssum += threshed.at<uint8_t>(mid_y-2, c); n++;
            }
            RLOOP (threshed.rows) {
                if (r < marg) continue;
                if (r >= threshed.cols - marg ) continue;
                ssum += threshed.at<uint8_t>(r, mid_x); n++;
                //ssum += threshed.at<uint8_t>(r, mid_x-1); n++;
                //ssum += threshed.at<uint8_t>(r, mid_x+1); n++;
            }
            ssum /= n;
            res = ssum;
        }
        return res;
    } // cross_feature

    // Look whether cross pixels are set in neighborhood of p_.
    // img should be binary, 0 or 1, from an adaptive threshold operation.
    // The result is close to 0 for empty intersections, else close to 1,
    // because that looks nicer when visualizing.
    //---------------------------------------------------------------------------------------------
    inline static float cross_feature_new( const cv::Mat &img, Point2f p_, float dx_, float dy_ )
    {
        float res = 1.0;
        int dx = ROUND(dx_/2.0);
        int dy = ROUND(dy_/2.0);
        cv::Mat threshed;
        cv::Point p(ROUND(p_.x), ROUND(p_.y));
        int w = 2*dx+1;
        int h = 2*dy+1;
        cv::Rect rect( p.x - dx, p.y - dy, w, h );
        if (check_rect( rect, img.rows, img.cols)) {
            cv::Mat hood = img(rect);
            int mid_y = ROUND(hood.rows / 2.0);
            int mid_x = ROUND(hood.cols / 2.0);
            float ssum = 0;
            int n = 0;
            const int marg = 5;
            // Look for horizontal line in the middle
            CLOOP (hood.cols) {
                if (c < marg) continue;
                if (c >= hood.cols - marg ) continue;
                ssum += hood.at<uint8_t>(mid_y, c); n++;
            }
            // Look for vertical line in the middle
            RLOOP (hood.rows) {
                if (r < marg) continue;
                if (r >= hood.cols - marg ) continue;
                ssum += hood.at<uint8_t>(r, mid_x); n++;
            }
            // Total sum of darkness
            float totsum = 0;
            RLOOP (hood.rows) {
                CLOOP (hood.cols) {
                      totsum += hood.at<uint8_t>(r, c);
                }
                    //PLOG( ">>>>>>> dx %d dy %d\n", dx, dy);
            }
            //PLOG( "========\n");
            //PLOG( "%3.2f %3.2f %3.2f %3.2f\n", ssum, totsum, RAT( ssum, totsum), (totsum-ssum)/area);
            ssum = RAT( ssum, totsum);
            ssum *= 30;
            res = 1-ssum;
            if (res < 0) res = 0;
        }
        return res;
    } // cross_feature

}; // class BlackWhiteEmpty
    

#endif /* BlackWhiteEmpty_hpp */
