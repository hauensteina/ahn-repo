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

cv::Mat mat_dbg;  // debug image to viz intermediate results
std::vector<float> BWE_brightness;
std::vector<float> BWE_sum;
std::vector<float> BWE_sigma;
std::vector<float> BWE_crossness_new;
std::vector<float> BWE_white_templ_score;
std::vector<float> BWE_empty_templ_score;

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
        int r, yshift;
        std::vector<int> res(SZ(intersections), EEMPTY);
        
        // Compute features for each board intersection
        r=3;
        get_feature( gray, intersections, r, brightness_feature, BWE_brightness);
        
        r=11; yshift = 0;
        get_feature( threshed, intersections, r, sum_feature, BWE_sum, yshift);
        
        // Black stones
        ISLOOP( BWE_brightness) {
            float bthresh = 35; // larger means more Black stones
            if (BWE_brightness[i] < bthresh ) {
                res[i] = BBLACK;
            }
        }
        // White places, first guess
        float sum_thresh    = 80;  // smaller means more White stones
        float bright_thresh = 200; // smaller means more White stones
        ISLOOP( BWE_brightness) {
            if ( BWE_brightness[i] > bright_thresh
                //&& BWE_crossness_new[i] < 100
                && BWE_sum[i] > sum_thresh
                && res[i] != BBLACK)  {
                res[i] = WWHITE;
            }
        }
        
        // Bootstrap.
        // Train templates on preliminary classification result, then reclassify,
        // repeat. This should compensate for highlights and changes in the environment.
        const int NITER = 2;
        const int MAGIC = 400;
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
            
            // Re-classify based on templates
            ISLOOP( res) {
                if (res[i] == BBLACK) continue;
                PLOG(" Template dist W-E: %.2f\n", BWE_white_templ_score[i] - BWE_empty_templ_score[i] );
                if (BWE_white_templ_score[i] > BWE_empty_templ_score[i]) {
                    if (res[i] != WWHITE) {
                        PLOG( "Switch E->W\n");
                    }
                    res[i] = WWHITE;
                }
                else if (BWE_white_templ_score[i] < BWE_empty_templ_score[i] + MAGIC ) {
                    if (res[i] != EEMPTY) {
                        PLOG( "Switch W->E\n");
                    }
                    res[i] = EEMPTY;
                }
            } // ISLOOP
        } // NLOOP
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
    
    // Take neighborhoods around points and average them, reulting in a
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
    
}; // class BlackWhiteEmpty
    

#endif /* BlackWhiteEmpty_hpp */
