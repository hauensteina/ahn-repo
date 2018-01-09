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
#include "Helpers.hpp"
#include "Ocv.hpp"

//extern cv::Mat mat_dbg;  // debug image to viz intermediate results
static std::vector<float> BWE_brightmatch;
static std::vector<float> BWE_darkmatch;
static std::vector<float> BWE_sum;
static std::vector<float> BWE_sum_inner;
static std::vector<float> BWE_outer_minus_inner;
static std::vector<float> BWE_sigma;
static std::vector<float> BWE_crossness_new;
static std::vector<float> BWE_white_templ_score;
static std::vector<float> BWE_black_templ_score;
static std::vector<float> BWE_empty_templ_score;
static std::vector<float> BWE_ringmatch;
static std::vector<float> BWE_crossmatch;
static std::vector<float> BWE_black_holes;
static std::vector<float> BWE_white_holes;
static std::vector<float> BWE_graymean;
static std::vector<float> BWE_centerspot;
const static std::string WHITE_TEMPL_FNAME = "white_templ.yml";
const static std::string BLACK_TEMPL_FNAME = "black_templ.yml";
const static std::string EMPTY_TEMPL_FNAME = "empty_templ.yml";

class BlackWhiteEmpty
//=====================
{
public:
    enum { BBLACK=0, EEMPTY=1, WWHITE=2, DONTKNOW=3 };
    enum { RING_R = 12 };

    //----------------------------------------------------------------------------------
    inline static std::vector<int> classify( const cv::Mat &pyr,
                                            const cv::Mat &gray,
                                            const Points2f &intersections,
                                            float &match_quality)
    {
        cv::Mat pyrgray, gray_threshed, black_holes, white_holes, bright_places;
        cv::cvtColor( pyr, pyrgray, cv::COLOR_RGB2GRAY);
        thresh_dilate( gray, gray_threshed, 4);

        // Catch false positives. White stones must be bright.
        cv::adaptiveThreshold( gray, bright_places, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 101, -50);

        // Black stones
        cv::Mat blurred, dark_places;
        cv::GaussianBlur( gray, blurred, cv::Size(9,9),0,0);
        cv::adaptiveThreshold( blurred, dark_places, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 50);

        //cv::adaptiveThreshold( pyrgray, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 80); // good

        //auto thresholds = find_thresholds( gray, intersections);
        // The White stones become black holes, all else is white
        int nhood_sz =  25;
        float thresh = -32; //8;
        cv::adaptiveThreshold( pyrgray, white_holes, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                              nhood_sz, thresh);
        // The Black stones become black holes, all else is white
        nhood_sz = 25;
        thresh = 16; // 8;
        cv::adaptiveThreshold( pyrgray, black_holes, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                              nhood_sz, thresh);
        
        // white_holes mean small => white
        // black_holes mean small => black
        // else empty
        int r;
        const int yshift = 0;
        // We scale features to 0..255 to allow hardcoded thresholds.
        const bool scale = true;
        const bool dontscale = false;
//        r=3;
//        get_feature( black_holes, intersections, r,
//                    [](const cv::Mat &hood) { return cv::mean(hood)[0]; },
//                    BWE_black_holes, yshift, scale);
        cv::Mat emptyMask7( 7, 7, CV_8UC1, cv::Scalar(0));
        cv::Mat fullMask7( 7, 7, CV_8UC1, cv::Scalar(255));
        //cv::Mat darkMask3( 3, 3, CV_8UC1, cv::Scalar(0));
        cv::Mat crossMask = crossmask(2,3);
        cv::Mat crossMaskInv = 255 - crossMask;

        int wiggle = 1;
        match_mask_near_points( black_holes, emptyMask7, intersections, wiggle, BWE_black_holes);
        match_mask_near_points( white_holes, emptyMask7, intersections, wiggle, BWE_white_holes);
        match_mask_near_points( gray_threshed, emptyMask7, intersections, wiggle, BWE_sum_inner);
        match_mask_near_points( bright_places, fullMask7, intersections, wiggle, BWE_brightmatch);
        match_mask_near_points( dark_places, fullMask7, intersections, wiggle, BWE_darkmatch);
        //match_mask_near_points( gray, crossMaskInv, intersections, 2, BWE_centerspot);
        int tt=42;
        
//        r = 3;
//        get_feature( white_holes, intersections, r,
//                    [](const cv::Mat &hood) { return cv::mean(hood)[0]; },
//                    BWE_white_holes, yshift, scale);

        // Gray mean
        r = 4;
        get_feature( pyrgray, intersections, r,
                    [](const cv::Mat &hood) { return cv::mean(hood)[0]; },
                    BWE_graymean, yshift, dontscale);
//        r=3;
//        get_feature( threshed, intersections, r,
//                    [](const cv::Mat &hood) { return cv::sum( hood)[0]; },
//                    BWE_sum_inner, yshift, dontscale);
        
        std::vector<int> res( SZ(intersections), EEMPTY);
        ISLOOP (BWE_black_holes) {
            float blackness   = BWE_black_holes[i];
            float whiteness   = BWE_white_holes[i];
            float brightmatch = BWE_brightmatch[i];
            float darkmatch   = BWE_darkmatch[i];
            float brightness  = BWE_graymean[i];
            float white_glare = BWE_sum_inner[i];
            //float cs = BWE_centerspot[i];
            //PLOG(">>>>>> %5d %.0f %.0f %.0f\n", i, wh, bh, bh-wh);
            if (darkmatch < 100) {
                res[i] = BBLACK;
            }
//            if ( darkness < 100 &&  whiteness < 80) res[i] = WWHITE; // frozen
//            if ( brightness > 200 &&  white_glare < 15) res[i] = WWHITE;
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
    } // check_rect()
    
    // Take neighborhoods around points and average them, resulting in a
    // template for matching.
    //--------------------------------------------------------------------------------------------
    inline static void avg_hoods( const cv::Mat &img, const Points2f &pts, int r, cv::Mat &dst)
    {
        dst = cv::Mat( 2*r + 1, 2*r + 1, CV_32FC1);
        int n = 0;
        ISLOOP (pts) {
            cv::Point p( ROUND(pts[i].x), ROUND(pts[i].y));
            cv::Rect rect( p.x - r, p.y - r, 2*r + 1, 2*r + 1 );
            if (!check_rect( rect, img.rows, img.cols)) continue;
            cv::Mat tmp;
            img( rect).convertTo( tmp, CV_32FC1);
            dst = dst * (n/(float)(n+1)) + tmp * (1/(float)(n+1));
            n++;
        }
    } // avg_hoods
    
    // Generic way to get any feature for all intersections
    //-----------------------------------------------------------------------------------------
    template <typename F>
    inline static void get_feature( const cv::Mat &img, const Points2f &intersections, int r,
                                   F Feat,
                                   std::vector<float> &res,
                                   float yshift = 0, bool scale_flag=true)
    {
        res.clear();
        float feat = 0;
        ISLOOP (intersections) {
            cv::Point p(ROUND(intersections[i].x), ROUND(intersections[i].y));
            cv::Rect rect( p.x - r, p.y - r + yshift, 2*r + 1, 2*r + 1 );
            feat = 0;
            if (check_rect( rect, img.rows, img.cols)) {
                const cv::Mat &hood( img(rect));
                feat = Feat( hood);
            }
            res.push_back( feat);
        } // for intersections
        if (scale_flag) {
            vec_scale( res, 255);
        }
    } // get_feature
    
    // Median of pixel values. Used to find B stones.
    //---------------------------------------------------------------------------------
    inline static float brightness_feature( const cv::Mat &hood)
    {
        return channel_median(hood);
    } // brightness_feature()

    // Median of pixel values. Used to find B stones.
    //---------------------------------------------------------------------------------
    inline static float sigma_feature( const cv::Mat &hood)
    {
        cv::Scalar mmean, sstddev;
        cv::meanStdDev( hood, mmean, sstddev);
        return sstddev[0];
    } // sigma_feature()

//    // Sum all pixels in hood.
//    //---------------------------------------------------------------------------------
//    inline static float sum_feature( const cv::Mat &hood)
//    {
//        return cv::sum( hood)[0];
//    } // sum_feature()
    
    // Look whether cross pixels are set in neighborhood of p_.
    // hood should be binary, 0 or 1, from an adaptive threshold operation.
    //---------------------------------------------------------------------------------
    inline static float cross_feature( const cv::Mat &hood)
    {
        int mid_y = ROUND(hood.rows / 2.0);
        int mid_x = ROUND(hood.cols / 2.0);
        float ssum = 0;
        // Look for horizontal line in the middle
        CLOOP (hood.cols) {
            ssum += hood.at<uint8_t>(mid_y, c);
        }
        // Look for vertical line in the middle
        RLOOP (hood.rows) {
            ssum += hood.at<uint8_t>(r, mid_x); 
        }
        float totsum = cv::sum(hood)[0];
        ssum = RAT( ssum, totsum);
        return ssum;
    } // cross_feature()
    
    // Return a ring shaped mask used to detect W stones in threshed gray img.
    // For some reason, this is much worse than outer_minus_inner.
    //-------------------------------------------------------------------------
    inline static cv::Mat& ringmask()
    {
        static cv::Mat mask;
        if (mask.rows) { return mask; }
        
        // Build the mask, once.
        const int r = RING_R;
        //const int middle_r = 8;
        const int inner_r = 3;
        const int width = 2*r + 1;
        const int height = width;
        mask = cv::Mat( height, width, CV_8UC1);
        mask = 0;
        cv::Point center( r, r);
        cv::circle( mask, center, r, 255, -1);
        //cv::circle( mask, center, middle_r, 127, -1);
        cv::circle( mask, center, inner_r, 0, -1);
        
        return mask;
    }

    // Return a cross shaped mask.
    // thickness is weird: 1->1, 2->3, 3->5, 4->5, 5->7, 6->7, ...
    //-------------------------------------------------------------------------
    inline static cv::Mat& crossmask( const int thickness_=5, const int r_=12)
    {
        static cv::Mat mask;
        static int thickness=0;
        static int r=0;
        if (r != r_ || thickness != thickness_) {
            r = r_;
            thickness = thickness_;
        }
        else {
            return mask;
        }
        // Build the mask, once.
        mask = cv::Mat( 2*r+1, 2*r+1, CV_8UC1);
        mask = 0;
        cv::Point center( r, r);
        // horiz
        cv::line( mask, cv::Point( 0, r), cv::Point( 2*r+1, r), 255, thickness);
        // vert
        cv::line( mask, cv::Point( r, 0), cv::Point( r, 2*r+1), 255, thickness);

        return mask;
    }
    
    // Match a mask to all points around p within a square of radius r. Return best match.
    // Image and mask are float mats with values 0 .. 255.0 .
    // The result is in the range 0..255. Smaller numbers indicate better match.
    // Mask dimensions must be odd.
    //---------------------------------------------------------------------------------------------------
    inline static int match_mask_near_point( const cv::Mat &img, const cv::Mat &mask, Point2f pf, int r)
    {
        assert( mask.rows % 2);
        assert( mask.cols % 2);
        int dx = mask.cols / 2;
        int dy = mask.rows / 2;
        cv::Point p = pf2p( pf);
        double mindiff = 1E9;
        for (int x = p.x - r; x <= p.x + r; x++) {
            for (int y = p.y - r; y <= p.y + r; y++) {
                cv::Point q( x, y);
                cv::Rect rect( q.x - dx, q.y - dy, mask.cols, mask.rows);
                if (!check_rect( rect, img.rows, img.cols)) continue;
                cv::Mat diff = cv::abs( mask - img(rect));
                double ssum = cv::sum( diff)[0];
                if (ssum < mindiff) { mindiff = ssum; }
            } // for y
        } // for x
        mindiff /= (mask.rows * mask.cols);
        mindiff = ROUND(mindiff);
        //if (mindiff > 255) mindiff = 255;
        return mindiff;
    } // match_mask_near_point()

    // Match a mask to all intersections. Find best match for each intersection within a radius.
    //--------------------------------------------------------------------------------------------
    inline static void match_mask_near_points( const cv::Mat &img_, const cv::Mat mask_,
                                              const Points2f &intersections, int r,
                                              std::vector<float> &res)
    {
        res.clear();
        cv::Mat img, mask;
        img_.convertTo( img, CV_32FC1);
        mask_.convertTo( mask, CV_32FC1);
        ISLOOP (intersections) {
            float feat = match_mask_near_point( img, mask, intersections[i], r);
            res.push_back( feat);
        }
    } // match_mask_near_points()

    // Find a threshold for each intersection. Our own adaptive threshold.
    //-----------------------------------------------------------------------------------------------------------------------------
    inline static std::vector<float> find_thresholds( const cv::Mat &img, const Points2f &intersections)
    {
        int rad = img.cols * 0.05;
        std::vector<float> xvals = vec_extract( intersections,  [](const Point2f &p) { return p.x;} );
        std::vector<float> yvals = vec_extract( intersections,  [](const Point2f &p) { return p.y;} );
        float leftedge  = vec_min( xvals);
        float rightedge = vec_max( xvals);
        float topedge = vec_min( yvals);
        float botedge = vec_max( yvals);
        std::vector<float> res;
        ISLOOP (intersections) {
            cv::Point p = pf2p( intersections[i]);
            int left = p.x - rad;
            if (left < leftedge) left = leftedge;
            int right = p.x + rad;
            if (right > rightedge) right = rightedge;
            int top = p.y - rad;
            if (top < topedge) top = topedge;
            int bot = p.y + rad;
            if (bot > botedge) bot = botedge;
            cv::Rect rect( left, top, right - left, bot - top);
            double mmin, mmax;
            cv::Point minloc, maxloc;
            cv::minMaxLoc( img(rect), &mmin, &mmax, &minloc, &maxloc);
            //res.push_back( (mmax + mmin) / 2.0);
            res.push_back( mmax);
        } // ISLOOP
        return res;
    } // find_thresholds
    

}; // class BlackWhiteEmpty































































#endif /* BlackWhiteEmpty_hpp */
