//
//  Boardness.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-12-19.
//  Copyright © 2017 AHN. All rights reserved.
//

// Class to find out how likely a board starts at left upper corner r,c
// in the grid of intersections.

#ifndef Boardness_hpp
#define Boardness_hpp

#include <iostream>
#include "Common.hpp"
#include "Ocv.hpp"

extern cv::Mat mat_dbg;

class Boardness
//=================
{
public:
    // Data
    //--------
    
    // From constructor
    const Points2f &m_intersections; // all intersections
    const cv::Mat  &m_pyr;           // pyramid filtered color image
    const Points2f m_blobs;         // Potential intersections found by BlobDetector
    const std::vector<cv::Vec2f> &m_horiz_lines; // Horizontal lines
    const std::vector<cv::Vec2f> &m_vert_lines; // Horizontal lines
    const int m_boardsz;
    
    // Internal
    cv::Mat m_pyrpix;  // A pixel per intersection, from m_pyr
    std::vector<bool> m_blobflags; // For each intersection, is there a blob
    cv::Mat m_edgeness;  // Prob of board at r,c by edgeness
    cv::Mat m_blobness;  // Prob of board at r,c by blobness

    // Methods
    //----------
    
    //---------------------------------------------------------------------------------------------------
    Boardness( const Points2f &intersections, const Points &blobs, const cv::Mat &pyr, int boardsz,
              const std::vector<cv::Vec2f> &horiz_lines, const std::vector<cv::Vec2f> &vert_lines) :
    m_intersections(intersections), m_pyr(pyr), m_blobs(points2float(blobs)), m_boardsz(boardsz),
    m_horiz_lines(horiz_lines), m_vert_lines(vert_lines)
    {
        // Pyr pixel value at each intersection as an image
        m_pyrpix = cv::Mat::zeros( SZ(horiz_lines), SZ(vert_lines), CV_8UC3);
        int i=0;
        RSLOOP (horiz_lines) {
            CSLOOP (vert_lines) {
                Point2f pf = intersections[i++];
                cv::Point p = pf2p(pf);
                if (p.x < m_pyr.cols && p.y < m_pyr.rows && p.x >= 0 && p.y >= 0) {
                    m_pyrpix.at<cv::Vec3b>(r,c) = m_pyr.at<cv::Vec3b>(p);
                }
            } // CSLOOP
        } // RSLOOP
    } // constructor

    // Color Difference between first lines and the backgroud
    //------------------------------------------------------------------------
    cv::Mat& edgeness()
    {
        const cv::Mat &m = m_pyrpix;
        cv::Mat tmp = cv::Mat::zeros( SZ(m_horiz_lines), SZ(m_vert_lines), CV_32FC1);
        float mmax = -1E9;
        
        RSLOOP (m_horiz_lines) {
            CSLOOP (m_vert_lines) {
                float ssum = 0;
                if (!p_on_img( cv::Point( c - 1,       r ), m)) continue;
                if (!p_on_img( cv::Point( c + m_boardsz, r ), m)) continue;
                if (!p_on_img( cv::Point( c,           r - 1), m)) continue;
                if (!p_on_img( cv::Point( c,           r + m_boardsz), m)) continue;
                
                // Left and right edge
                for (int rr = r; rr < r + m_boardsz; rr++) {
                    auto b_l = m.at<cv::Vec3b>( rr, c); // on the board
                    auto b_r = m.at<cv::Vec3b>( rr, c + m_boardsz -1);
                    auto o_l = m.at<cv::Vec3b>( rr, c - 1); // just outside the board
                    auto o_r = m.at<cv::Vec3b>( rr, c + m_boardsz);
                    ssum += cv::norm( b_l, o_l);
                    ssum += cv::norm( b_r, o_r);
                }
                // Top and bottom edge
                for (int cc = c; cc < c + m_boardsz; cc++) {
                    auto b_t = m.at<cv::Vec3b>( r, cc); // on the board
                    auto b_b = m.at<cv::Vec3b>( r + m_boardsz - 1, cc);
                    auto o_t = m.at<cv::Vec3b>( r - 1, cc); // just outside the board
                    auto o_b = m.at<cv::Vec3b>( r + m_boardsz, cc);
                    ssum += cv::norm( b_t, o_t);
                    ssum += cv::norm( b_b, o_b);
                }
                tmp.at<float>(r,c) = ssum;
                if (ssum > mmax) mmax = ssum;
            } // CSLOOP
        } // RSLOOP
        double scale = 255.0 / mmax;
        tmp.convertTo( m_edgeness, CV_8UC1, scale);
        return m_edgeness;
    } // edgeness()
    
    // Percentage of blobs captured by the board
    //---------------------------------------------
    cv::Mat &blobness()
    {
        const cv::Mat &m = m_pyrpix;
        cv::Mat tmp = cv::Mat::zeros( SZ(m_horiz_lines), SZ(m_vert_lines), CV_32FC1);
        float mmax = -1E9;

        if (!SZ(m_blobflags)) { fill_m_blobflags(); }
        RSLOOP (m_horiz_lines) {
            CSLOOP (m_vert_lines) {
                float ssum = 0;
                if (!p_on_img( cv::Point( c - 1,       r ), m)) continue;
                if (!p_on_img( cv::Point( c + m_boardsz, r ), m)) continue;
                if (!p_on_img( cv::Point( c,           r - 1), m)) continue;
                if (!p_on_img( cv::Point( c,           r + m_boardsz), m)) continue;
                for (int rr = r; rr < r + m_boardsz; rr++) {
                    for (int cc = c; cc < c + m_boardsz; cc++) {
                        int idx = rc2idx( rr,cc); 
                        if (m_blobflags[idx]) { ssum += 1; }
                    }
                }
                tmp.at<float>(r,c) = ssum;
                if (ssum > mmax) mmax = ssum;
            } // CSLOOP
        } // RSLOOP
        double scale = 255.0 / mmax;
        tmp.convertTo( m_blobness, CV_8UC1, scale);
        return m_blobness;
    } // blobness
private:
    // Fill m_blobflags. For each intersection, is there a blob.
    //------------------------------------------------------------
    void fill_m_blobflags()
    {
        const int EPS = 2.0;
        typedef struct { int idx; float d; } Idxd;
        // All points on horiz lines
        std::vector<Idxd> blob_to_horiz( SZ(m_blobs), {-1,1E9});
        ISLOOP (m_horiz_lines) {
            KSLOOP (m_blobs) {
                auto p = m_blobs[k];
                float d = fabs(dist_point_line( p, m_horiz_lines[i]));
                if (d < blob_to_horiz[k].d) {
                    blob_to_horiz[k].idx = i;
                    blob_to_horiz[k].d = d;
                }
            }
        } // ISLOOP
        // All points on vert lines
        std::vector<Idxd> blob_to_vert( SZ(m_blobs), {-1,1E9});
        ISLOOP (m_vert_lines) {
            KSLOOP (m_blobs) {
                auto p = m_blobs[k];
                float d = fabs(dist_point_line( p, m_vert_lines[i]));
                if (d < blob_to_vert[k].d) {
                    blob_to_vert[k].idx = i;
                    blob_to_vert[k].d = d;
                }
            }
        } // ISLOOP

        m_blobflags = std::vector<bool>(SZ(m_intersections),false);
        mat_dbg = cv::Mat::zeros( SZ(m_horiz_lines), SZ(m_vert_lines), CV_8UC3);
        KSLOOP (m_blobs) {
            int blobrow = blob_to_horiz[k].idx;
            int blobcol = blob_to_vert[k].idx;
            float blobd_h = blob_to_horiz[k].d;
            float blobd_v = blob_to_horiz[k].d;
            if (blobrow >= 0 && blobcol >= 0 && blobd_h < EPS && blobd_v < EPS) {
                m_blobflags[rc2idx(blobrow, blobcol)] = true;
                auto col = get_color();
                mat_dbg.at<cv::Vec3b>(blobrow,blobcol) = cv::Vec3b( col[0],col[1],col[2]);
            }
        } // KSLOOP
    } // fill_m_blobflags()
    
    // Convert r,c of intersection into linear index
    //--------------------------------------------------
    int rc2idx( int r, int c)
    {
        return r * SZ(m_vert_lines) + c;
    }
                    
}; // class Boardness


#endif /* Boardness_hpp */
