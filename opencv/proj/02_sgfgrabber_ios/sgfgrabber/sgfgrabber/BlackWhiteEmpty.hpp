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
std::vector<float> BWE_brightness;
std::vector<float> BWE_crossness_new;

class BlackWhiteEmpty
//=====================
{
public:
    enum { BBLACK=0, EEMPTY=1, WWHITE=2, DONTKNOW=3 };
    
    //----------------------------------------------------------------------------------
    inline static std::vector<int> classify( const cv::Mat &gray,
                                            const cv::Mat &threshed,
                                            const Points2f &intersections,
                                            float dx, // approximate dist between lines
                                            float dy)
    {
        std::vector<int> res(SZ(intersections), EEMPTY);
        
        // Compute features for each board intersection
        get_brightness( gray, intersections, dx, dy, BWE_brightness);
        
        int r=3;
        get_feature( threshed, intersections, r, cross_feature_new, BWE_crossness_new);

        // Black stones
        float black_median = vec_median( BWE_brightness);
        ISLOOP( BWE_brightness) {
            float bthresh = black_median * 0.5; // larger means more Black stones
            if (BWE_brightness[i] < bthresh /* && black_features[i] - tt_feat[i] < 8 */ ) {
                res[i] = BBLACK;
            }
        }
        // White places
        ISLOOP( BWE_crossness_new) {
            if (BWE_crossness_new[i] < 50 &&  res[i] != BBLACK)  {
                res[i] = WWHITE;
            }
        }
        return res;
    } // classify()

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
    
    // Generic way to get any feature for all intersections
    //-----------------------------------------------------------------------------------------
    template <typename F>
    inline static void get_feature( const cv::Mat &img, const Points2f &intersections, int r,
                                   F Feat,
                                   std::vector<float> &res,
                                   float yshift = 0)
    {
        res.clear();
        float feat = 0;
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - r, p.y - r + yshift, 2*r + 1, 2*r + 1 );
            if (check_rect( rect, img.rows, img.cols)) {
                const cv::Mat &hood( img(rect));
                feat = Feat( hood);
            }
            res.push_back( feat);
        } // for intersections
        vec_scale( res, 255);
    } // get_feature
    
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
        float brightness = 0;
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            if (check_rect( rect, img.rows, img.cols)) {
                const cv::Mat &hood( img(rect));
                brightness = channel_median( hood);
            }
            res.push_back( brightness);
        } // for intersections
    } // get_brightness()

//    //-------------------------------------------------------------------
//    inline static void get_crossness( const cv::Mat &img, // gray
//                                     const Points2f &intersections,
//                                     float dx_, float dy_,
//                                     std::vector<float> &res )
//    {
//        //mat_dbg = img.clone();
//        res.clear();
//        ISLOOP (intersections) {
//            //float feat = cross_feature( img, intersections[i], dx_, dy_);
//            float feat = cross_feature_new( img, intersections[i], dx_, dy_);
//            res.push_back( feat);
//        } // for intersections
//    } // get_crossness()
    
    // Look whether the cross pixels are set
    //----------------------------------------------------------------------------------------
    inline static float center_sum( const cv::Mat &img, Point2f p_, float dx_, float dy_ )
    {
        float res = 0;
        int dx_inner = ROUND(dx_/4.0);
        int dy_inner = ROUND(dy_/4.0);
        int dx_outer = ROUND(dx_/2.0);
        int dy_outer = ROUND(dy_/2.0);
        float area = dx_*dy_;
        float innersum=0, outersum=0;
        const int NSHIFT = 3;
        for (int shift=0; shift < NSHIFT; shift++) {
            cv::Point p(ROUND(p_.x), ROUND(p_.y - shift));
            cv::Rect rect_inner( p.x - dx_inner, p.y - dy_inner, 2*dx_inner+1, 2*dy_inner+1 );
            cv::Rect rect_outer( p.x - dx_outer, p.y - dy_outer, 2*dx_outer+1, 2*dy_outer+1 );
            if (check_rect( rect_inner, img.rows, img.cols) &&
                check_rect( rect_outer, img.rows, img.cols))
            {
                cv::Mat hood_inner = img(rect_inner);
                cv::Mat hood_outer = img(rect_outer);
                innersum = cv::sum(hood_inner)[0];
                outersum = cv::sum(hood_outer)[0];
                res += (innersum - outersum)/area;
            }
        } // for shift
        res /= NSHIFT;
        return -res;
    } // center_sum


    //------------------------------------------------------------------------
    inline static void get_whiteness( const cv::Mat &threshed,
                                     const Points2f &intersections,
                                     float dx_, float dy_,
                                     std::vector<float> &res )
    {
        int dx = ROUND(dx_/4.0);
        int dy = ROUND(dy_/4.0);
        float area = (2*dx+1) * (2*dy+1);

        const int tsz = 15;
        uint8_t tmpl[tsz*tsz] = {
            1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
            1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
            1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
            1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,
            1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,
            1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,
            1,1,1,1,1,1,0,0,0,1,1,1,1,1,1
        };
//        uint8_t tmpl[tsz*tsz] = {
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
//            1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
//            1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
//            1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
//            1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//        };
        cv::Mat mtmpl = 255 * cv::Mat(tsz, tsz, CV_8UC1, tmpl);
        cv::Mat mtt;
        cv::copyMakeBorder( threshed, mtt, tsz/2, tsz/2, tsz/2, tsz/2, cv::BORDER_REPLICATE, cv::Scalar(0));
        cv::Mat dst;
        cv::matchTemplate( mtt, mtmpl, dst, CV_TM_SQDIFF);
        cv::normalize( dst, dst, 0 , 255, CV_MINMAX, CV_8UC1);
        res.clear();
        float wness = 0;
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y-2));
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            if (check_rect( rect, threshed.rows, threshed.cols)) {
                cv::Mat hood = dst(rect);
                float wness = cv::sum(hood)[0] / area; // 0 .. 255
                wness /= 255.0; // 0 .. 1; best match is 0
                wness = -log(wness); // 0 .. inf
            }
            res.push_back( wness);
        } // for intersections
        mat_dbg = dst.clone();
    } // get_whiteness()

    //------------------------------------------------------------------------
    inline static void get_crossness( const cv::Mat &threshed,
                                     const Points2f &intersections,
                                     float dx_, float dy_,
                                     std::vector<float> &res )
    {
        int dx = 2; // ROUND(dx_/.0);
        int dy = 2; // ROUND(dy_/5.0);
        float area = (2*dx+1) * (2*dy+1);

        const int tsz = 15;
//        uint8_t tmpl[tsz*tsz] = {
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//            0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,
//        };
//        uint8_t tmpl[tsz*tsz] = {
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//            0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
//        };
        uint8_t tmpl[tsz*tsz] = {
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,
        };
//        uint8_t tmpl[tsz*tsz] = {
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//            0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,
//        };
        cv::Mat mtmpl = 255 * cv::Mat(tsz, tsz, CV_8UC1, tmpl);
        cv::Mat mtt;
        cv::copyMakeBorder( threshed, mtt, tsz/2, tsz/2, tsz/2, tsz/2, cv::BORDER_REPLICATE, cv::Scalar(0));
        cv::Mat dst;
        cv::matchTemplate( mtt, mtmpl, dst, CV_TM_SQDIFF);
        cv::normalize( dst, dst, 0 , 255, CV_MINMAX, CV_8UC1);
        res.clear();
        float cness = 255;
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - dx, p.y - dy-2, 2*dx+1, 2*dy+1 );
            if (check_rect( rect, threshed.rows, threshed.cols)) {
                cv::Mat hood = dst(rect);
                cness = cv::sum(hood)[0] / area;
            }
            res.push_back( 255 - cness);
        } // for intersections
        vec_scale( res, 255);
        //PLOG( "min cross: %.0f\n", vec_min( res));
        mat_dbg = dst.clone();
    } // get_crossness()
    

    
//    // Look whether the cross pixels are set
//    //------------------------------------------------------------------------------------------
//    inline static float cross_feature( const cv::Mat &img, Point2f p_, float r, float yshift=0)
//    {
//        float res = 0;
//        int dx = ROUND(r/2.0);
//        int dy = ROUND(r/2.0);
//        cv::Mat threshed;
//        cv::Point p(ROUND(p_.x), ROUND(p_.y));
//        cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
//        if (check_rect( rect, img.rows, img.cols)) {
//            cv::Mat hood = img(rect);
//            inv_thresh_avg( hood, threshed);
//            int mid_y = ROUND(threshed.rows / 2.0);
//            int mid_x = ROUND(threshed.cols / 2.0);
//            float ssum = 0;
//            int n = 0;
//            const int marg = 6;
//            CLOOP (threshed.cols) {
//                if (c < marg) continue;
//                if (c >= threshed.cols - marg ) continue;
//                //ssum += threshed.at<uint8_t>(mid_y, c); n++;
//                ssum += threshed.at<uint8_t>(mid_y-1, c); n++;
//                ssum += threshed.at<uint8_t>(mid_y-2, c); n++;
//            }
//            RLOOP (threshed.rows) {
//                if (r < marg) continue;
//                if (r >= threshed.cols - marg ) continue;
//                ssum += threshed.at<uint8_t>(r, mid_x); n++;
//                //ssum += threshed.at<uint8_t>(r, mid_x-1); n++;
//                //ssum += threshed.at<uint8_t>(r, mid_x+1); n++;
//            }
//            ssum /= n;
//            res = ssum;
//        }
//        return fabs(res);
//    } // cross_feature()

    // Look whether cross pixels are set in neighborhood of p_.
    // hood should be binary, 0 or 1, from an adaptive threshold operation.
    //---------------------------------------------------------------------------------
    inline static float cross_feature_new( const cv::Mat &hood)
    {
        int mid_y = ROUND(hood.rows / 2.0);
        int mid_x = ROUND(hood.cols / 2.0);
        float ssum = 0;
        int n = 0;
        // Look for horizontal line in the middle
        CLOOP (hood.cols) {
            ssum += hood.at<uint8_t>(mid_y, c); n++;
        }
        // Look for vertical line in the middle
        RLOOP (hood.rows) {
            ssum += hood.at<uint8_t>(r, mid_x); n++;
        }
        // Total sum of darkness
        float totsum = 0;
        RLOOP (hood.rows) {
            CLOOP (hood.cols) {
                totsum += hood.at<uint8_t>(r, c);
            }
        }
        ssum = RAT( ssum, totsum);
        return fabs(ssum);
    } // cross_feature_new()

}; // class BlackWhiteEmpty
    

#endif /* BlackWhiteEmpty_hpp */
