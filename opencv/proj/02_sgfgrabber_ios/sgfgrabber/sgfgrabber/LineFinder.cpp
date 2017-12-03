//
//  LineFinder.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

// Class to find horizontal and vertical lines in a cloud of points.

#include "LineFinder.hpp"
#include "Clust1D.hpp"

// Use a model to find out whereabouts we expect the lines to be.
// Later, use the result to cluster points and fix lines in LineFixer.
//----------------------------------------------------------------------
void LineFinder::cluster()
{
    cv::Vec2f hslope, vslope;
    hslope = {0, PI/2};
    vslope = {0, 0};
    const float wwidth = 1.0;
    const int min_points = 4;
    std::vector<float> horizontal_cuts = Clust1D::cluster( m_cloud, wwidth,
                                                          [hslope](cv::Point p) { return fabs(dist_point_line(p, hslope)); });
    Clust1D::classify( m_cloud, horizontal_cuts, min_points,
                      [hslope](cv::Point p) { return fabs(dist_point_line(p, hslope)); },
                      m_horizontal_clusters);
    ISLOOP (m_horizontal_clusters) {
        rem_dups_x( m_horizontal_clusters[i], 5);
    }
} // get_lines()

// Get the ratio between successive horizontal line distances.
// Should be > 1.0; closer line is the denominator.
// Ratline is the line with the median ratio.
// In dy, return the current line distance at the ratline.
//----------------------------------------------------------------
float LineFinder::dy_rat( cv::Vec2f &ratline, float &dy, int &idx)
{
    m_horizontal_lines.clear();
    ISLOOP (m_horizontal_clusters) {
        cv::Vec4f line =  fit_line( m_horizontal_clusters[i]);
        cv::Vec2f pline = segment2polar(line);
        m_horizontal_lines.push_back( pline);
    }
    float middle_x = m_imgSize.width / 2.0;
    
    // Find distances from previous
    typedef struct { int idx; float dist; } DistIdx;
    std::vector<DistIdx> dists;
    ISLOOP (m_horizontal_lines) {
        if (i==0) continue;
        DistIdx di = { i, y_from_x( middle_x, m_horizontal_lines[i]) - y_from_x( middle_x, m_horizontal_lines[i-1]) };
        dists.push_back( di);
    }
    
    // Find ratios
    std::vector<DistIdx> rats;
    ISLOOP (dists) {
        if (i==0) continue;
        float r = RAT( dists[i].dist, dists[i-1].dist);
        DistIdx rat = { dists[i].idx, r};
        rats.push_back( rat);
    }
    std::sort( rats.begin(), rats.end(), [](DistIdx a, DistIdx b){ return a.dist < b.dist; });
    DistIdx med = vec_median( rats, [](DistIdx di) { return di.dist; });
    ratline = m_horizontal_lines[med.idx];
    dy = dists[med.idx-1].dist;
    assert( dists[med.idx-1].idx == med.idx);
    float res = med.dist;
    idx = med.idx;
    return res;
} // dy_rat()


