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

//extern cv::Mat mat_dbg;  // debug image to viz intermediate results
static std::vector<float> BWE_brightness;
static std::vector<float> BWE_sum;
static std::vector<float> BWE_sum_inner;
static std::vector<float> BWE_outer_minus_inner;
static std::vector<float> BWE_sigma;
static std::vector<float> BWE_crossness_new;
static std::vector<float> BWE_white_templ_score;
static std::vector<float> BWE_black_templ_score;
static std::vector<float> BWE_empty_templ_score;
static std::vector<float> BWE_ringmatch;
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
    inline static std::vector<int> classify( const cv::Mat &gray,
                                            const cv::Mat &threshed,
                                            const Points2f &intersections,
                                            float &match_quality)
    {
        //static cv::Mat default_empty_template = readMat( )
        int r, yshift;
        std::vector<int> res(SZ(intersections), EEMPTY);
        
        // Compute features for each board intersection
        r=3;
        get_feature( gray, intersections, r, brightness_feature, BWE_brightness);
        //float max_brightness = vec_max( BWE_brightness);
        
        r=10; yshift=0;
        get_feature( threshed, intersections, r, sum_feature, BWE_sum, yshift);
        //float max_sum = vec_max( BWE_sum);

        r=3; yshift=0;
        get_feature( threshed, intersections, r, sum_feature, BWE_sum_inner, yshift);

        r=RING_R ; yshift=0;
        get_feature( threshed, intersections, r,
                    [](const cv::Mat &hood) { return mat_dist( ringmask(), hood); },
                    BWE_ringmatch, yshift);

        BWE_outer_minus_inner = BWE_sum;
        // Looking for a ring
        vec_sub( BWE_outer_minus_inner, BWE_sum_inner); // Yes, do this twice
        vec_sub( BWE_outer_minus_inner, BWE_sum_inner);
        //float max_outer_minus_inner = vec_max( BWE_outer_minus_inner);

        // Black stones
        ISLOOP( BWE_brightness) {
            float bthresh = 35; // larger means more Black stones
            if (BWE_brightness[i] < bthresh ) {
                res[i] = BBLACK;
            }
        }
        // White places, first guess
        //float sum_thresh    = 80;  // smaller means more White stones
        //float sum_thresh    = 0.8 * max_sum;  // smaller means more White stones
        //float bright_thresh = 0.9 * max_brightness; // smaller means more White stones
        //float ring_thresh = 0.9 * max_outer_minus_inner;
        ISLOOP( BWE_brightness) {
            if ( 1
                //&& BWE_brightness[i] > bright_thresh
                //&& BWE_crossness_new[i] < 100
                //&& BWE_sum[i] > sum_thresh
                && BWE_outer_minus_inner[i] > 0
                //&& BWE_ringmatch[i] < 100
                && res[i] != BBLACK)
            {
                res[i] = WWHITE;
            }
        }
#define BOOTSTRAP
#ifdef BOOTSTRAP
        // Bootstrap.
        // Train templates on preliminary classification result, then reclassify,
        // repeat. This should compensate for highlights and changes in the environment.
        const int NITER = 10; // Not sure about the best number
        //const int WMAGIC = 800; // larger means less W stones
        const int EMAGIC = 0;   // larger means more W stones
        NLOOP (NITER) {
            // Make a template for white places
            Points2f white_intersections;
            ISLOOP( res) {
                if (res[i] != WWHITE) continue;
                white_intersections.push_back( intersections[i]);
            }
            cv::Mat white_template;
            r = 10;
            avg_hoods( threshed, white_intersections, r, white_template);
            
            // Make a template for black places
            Points2f black_intersections;
            ISLOOP( res) {
                if (res[i] != BBLACK) continue;
                black_intersections.push_back( intersections[i]);
            }
            cv::Mat black_template;
            r = 10;
            avg_hoods( threshed, black_intersections, r, black_template);
            
            // Make a template for empty places
            Points2f empty_intersections;
            ISLOOP( res) {
                if (res[i] != EEMPTY) continue;
                empty_intersections.push_back( intersections[i]);
            }
            cv::Mat empty_template;
            r = 10;
            avg_hoods( threshed, empty_intersections, r, empty_template);
            
            // Get the template scores
            get_feature( threshed, intersections, r,
                        [white_template](const cv::Mat &hood) {
                            cv::Mat tmp;
                            hood.convertTo( tmp, CV_32FC1);
                            float res = MAX( 1e-5, mat_dist( tmp, white_template));
                            return -res;
                        },
                        BWE_white_templ_score, 0, false);
            get_feature( threshed, intersections, r,
                        [empty_template](const cv::Mat &hood) {
                            cv::Mat tmp;
                            hood.convertTo( tmp, CV_32FC1);
                            float res = MAX( 1e-5, mat_dist( tmp, empty_template));
                            return -res;
                        },
                        BWE_empty_templ_score, 0, false);
            get_feature( threshed, intersections, r,
                        [black_template](const cv::Mat &hood) {
                            cv::Mat tmp;
                            hood.convertTo( tmp, CV_32FC1);
                            float res = MAX( 1e-5, mat_dist( tmp, black_template));
                            return -res;
                        },
                        BWE_black_templ_score, 0, false);
            
            // Re-classify based on templates
            ISLOOP( res) {
                if (res[i] == BBLACK) {
                    if (BWE_empty_templ_score[i] > BWE_black_templ_score[i]) {
                        res[i] = EEMPTY;
                    }
                }
                //PLOG(" Template dist W-E: %.2f\n", BWE_white_templ_score[i] - BWE_empty_templ_score[i] );
//                else if (BWE_white_templ_score[i] - WMAGIC > BWE_empty_templ_score[i]) {
//                    res[i] = WWHITE;
//                }
                else if (BWE_empty_templ_score[i] - EMAGIC > BWE_white_templ_score[i]  ) {
                    res[i] = EEMPTY;
                }
            } // ISLOOP
//            // Save templates to file
//            std::string path;
//            path = g_docroot + "/" + WHITE_TEMPL_FNAME;
//            cv::FileStorage efilew( path, cv::FileStorage::WRITE);
//            efilew << "white_template" << white_template;
//
//            path = g_docroot + "/" + BLACK_TEMPL_FNAME;
//            cv::FileStorage efileb( path, cv::FileStorage::WRITE);
//            efileb << "black_template" << black_template;
//
//            path = g_docroot + "/" + EMPTY_TEMPL_FNAME;
//            cv::FileStorage efilee( path, cv::FileStorage::WRITE);
//            efilee << "empty_template" << empty_template;
        } // NLOOP
#endif
        // Get an overall match quaity score
        std::vector<float> best_matches = BWE_white_templ_score;
        ISLOOP (best_matches) {
            if (BWE_black_templ_score[i] > best_matches[i]) {
                best_matches[i] = BWE_black_templ_score[i];
            }
            if (BWE_empty_templ_score[i] > best_matches[i]) {
                best_matches[i] = BWE_empty_templ_score[i];
            }
        }
        match_quality = vec_sum( best_matches);
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

    // Sum all pixels in hood.
    //---------------------------------------------------------------------------------
    inline static float sum_feature( const cv::Mat &hood)
    {
        return cv::sum( hood)[0];
    } // sum_feature()
    
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

}; // class BlackWhiteEmpty































































#endif /* BlackWhiteEmpty_hpp */
