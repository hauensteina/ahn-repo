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
void LineFinder::get_lines( std::vector<cv::Vec4f> &horizontal_lines,
                           std::vector<cv::Vec4f> &vertical_lines)
{
    cv::Vec4f hslope, vslope;
    hslope = PI/2;
    vslope = 0;
    // Find predominant slopes
    //find_slopes( hslope, vslope);
    // Cluster points by distance from predominant slope lines.
    //    m_horizontal_clusters = cluster( m_cloud, m_boardsize,
    //                                    [hslope](cv::Point &p) { return dist_point_line(p, hslope); });
    //    m_vertical_clusters = cluster( m_cloud, m_boardsize,
    //                                  [vslope](cv::Point &p) { return dist_point_line(p, vslope); });
    //const float wwidth = 5.0;
    //const float wwidth = 2.5;
    //const float wwidth = 100;
    const float wwidth = 1.0;
    const int min_points = 4;
    std::vector<float> horizontal_cuts = Clust1D::cluster( m_cloud, wwidth,
                                                          [hslope](cv::Point p) { return fabs(dist_point_line(p, hslope)); });
    Clust1D::classify( m_cloud, horizontal_cuts, min_points,
                      [hslope](cv::Point p) { return fabs(dist_point_line(p, hslope)); },
                      m_horizontal_clusters);
    
    std::vector<float> vertical_cuts   = Clust1D::cluster( m_cloud, wwidth,
                                                          [vslope](cv::Point p) { return fabs(dist_point_line(p, vslope)); });
    
    Clust1D::classify( m_cloud, vertical_cuts, min_points,
                      [vslope](cv::Point p) { return fabs(dist_point_line(p, vslope)); },
                      m_vertical_clusters);
    
    float delta_wavelen_h, slope_h, median_rho_h;
    std::vector<cv::Vec4f> lines_h;
    std::vector<float> dists_h;
    find_rhythm( m_horizontal_clusters,
                m_wavelen_h,
                delta_wavelen_h,
                slope_h,
                median_rho_h,
                lines_h,
                dists_h);
    
    float delta_wavelen_v, slope_v, median_rho_v;
    std::vector<cv::Vec4f> lines_v;
    std::vector<float> dists_v;
    find_rhythm( m_vertical_clusters,
                m_wavelen_v,
                delta_wavelen_v,
                slope_v,
                median_rho_v,
                lines_v,
                dists_v);
    
    std::vector<cv::Vec4f> lines;
    find_lines( m_imgSize.height, m_wavelen_h, delta_wavelen_h, slope_h, median_rho_h, horizontal_lines);
    find_lines( m_imgSize.width , m_wavelen_v, delta_wavelen_v, slope_v, median_rho_v, vertical_lines);
    
} // get_lines()


// Find two line segments representing the typical slope of horiz and vert
// lines. Used later for distance sorting.
//-------------------------------------------------------------------------
void LineFinder::find_slopes( cv::Vec4f &hslope, cv::Vec4f &vslope)
{
    // Find Hough lines in the detected intersections and stones
    cv::Mat canvas = cv::Mat::zeros( m_imgSize, CV_8UC1 );
    ISLOOP (m_cloud) {
        draw_point( m_cloud[i], canvas,1, cv::Scalar(255));
    }
    std::vector<cv::Vec2f> lines;
    std::vector<std::vector<cv::Vec2f> > horiz_vert_other_lines;
    std::vector<int> vote_thresholds = { 10, 9, 8, 7, 6, 5 };
    
    ISLOOP (vote_thresholds) {
        int votes = vote_thresholds[i];
        std::cerr << "trying " << votes << " hough line votes\n";
        HoughLines(canvas, lines, 1, CV_PI/180, votes, 0, 0 );
        
        // Separate horizontal, vertical, and other lines
        horiz_vert_other_lines = partition( lines, 3,
                                           [](cv::Vec2f &line) {
                                               const float thresh = 10.0;
                                               float theta = line[1] * (180.0 / CV_PI);
                                               if (fabs(theta - 180) < thresh) return 1;
                                               else if (fabs(theta) < thresh) return 1;
                                               else if (fabs(theta-90) < thresh) return 0;
                                               else return 2;
                                           });
        // Sort by Rho (distance of line from origin)
        std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
        std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
        if (hlines.size() >= 2 && vlines.size() >= 2) break;
    }
    std::vector<cv::Vec2f> &hlines = horiz_vert_other_lines[0];
    std::vector<cv::Vec2f> &vlines = horiz_vert_other_lines[1];
//    hslope = avg_slope_line( hlines);
//    vslope = avg_slope_line( vlines);
    hslope = median_slope_line( hlines);
    vslope = median_slope_line( vlines);
} // find_slopes()

// Find phase, wavelength etc of a family of lines.
// Each cluster has a bunch of points which are probably on the same line.
//----------------------------------------------------------------------------
void LineFinder::find_rhythm( const std::vector<Points> &clusters,
                             float &wavelength,
                             float &delta_wavelength,
                             float &slope,
                             float &median_rho,
                             std::vector<cv::Vec4f> &lines,
                             std::vector<float> &dists
                             )
{
    typedef struct { float dist; float slope; } DistSlope;
    //std::vector<cv::Vec4f> lines;
    // Lines through the clusters
    lines.clear();
    ISLOOP (clusters) {
        cv::Vec4f line =  fit_line( m_horizontal_clusters[i]);
        cv::Vec2f pline;
        segment2polar(line,pline);        // Kludge to make the line long enough to see
        polar2segment(pline, line);
        lines.push_back( line);
    }
    // Slopes of the lines
    std::vector<float> slopes;
    ISLOOP( lines) {
        cv::Vec4f line = lines[i];
        float dx = line[2] - line[0];
        float dy = line[3] - line[1];
        if (fabs(dx) > fabs(dy)) { // horizontal
            if (dx < 0) { dx *= -1; dy *= -1; }
        }
        else { // vertical
            if (dy > 0) { dx *= -1; dy *= -1; }
        }
        float theta = atan2( dy, dx);
        //NSLog(@"dx dy theta %.2f %.2f %.2f", dx, dy, theta );
        slopes.push_back( theta);
    }
    slope = vec_median( slopes);
    // A polar line with the median slope
    cv::Vec2f median_hline(0, slope + CV_PI/2.0);
    
    // For each cluster, get the median dist from the median slope line
    std::vector<DistSlope> distSlopes( clusters.size());
    dists.clear();
    ISLOOP (clusters) {
        std::vector<float> ds;
        JSLOOP (clusters[i]) {
            cv::Point p = clusters[i][j];
            float d = dist_point_line( p, median_hline);
            ds.push_back( d);
        }
        float dist = vec_median( ds);
        dists.push_back(dist);
        distSlopes[i].dist = dist;
        distSlopes[i].slope = slopes[i];
    }
    
    // Get the rhythm (wavelength of line distances)
    std::sort( distSlopes.begin(), distSlopes.end(), [](DistSlope a, DistSlope b){ return a.dist < b.dist; });
    median_rho = distSlopes[distSlopes.size() / 2].dist;
    std::vector<float> delta_dists;
    ISLOOP (distSlopes) {
        if (!i) continue;
        delta_dists.push_back( distSlopes[i].dist - distSlopes[i-1].dist);
    }
    delta_wavelength = vec_median_delta( delta_dists);
    wavelength = vec_median( delta_dists);
} // find_rhythm()

// Start in the middle with the medians, expand to both sides
// while adjusting with delta_slope and delta_rho.
//---------------------------------------------------------------
void LineFinder::find_lines( int max_rho,
                            float wavelength_,
                            float delta_wavelength,
                            float slope,
                            float median_rho,
                            std::vector<cv::Vec4f> &lines)
{
    if (wavelength_ < 5) {
        PLOG( "find_lines(): Wavelen %.2f too low", wavelength_);
        return; }
    float theta, rho, wavelength;
    std::vector<cv::Vec2f> hlines;
    
    // center to lower rho
    wavelength = wavelength_;
    theta = slope + CV_PI/2;
    rho = median_rho - wavelength;
    while (rho > 0 && wavelength > 0) {
        hlines.push_back( cv::Vec2f ( rho, theta));
        rho -= wavelength;
        wavelength -= delta_wavelength;
    }
    // center to higher rho
    wavelength = wavelength_;
    theta = slope + CV_PI/2;
    rho = median_rho;
    while (rho < max_rho && wavelength > 0) {
        hlines.push_back( cv::Vec2f ( rho, theta));
        rho += wavelength;
        wavelength += delta_wavelength;
    }
    // convert to segments
    lines.clear();
    ISLOOP (hlines) {
        cv::Vec4f line;
        polar2segment( hlines[i], line);
        lines.push_back( line);
    }
} // find_lines()

// Get some characteristics of the best two horizontal lines
//-----------------------------------------------------------------------------------------------------
void LineFinder::best_two_horiz_lines( int &idx1, int &idx2,     // index in m_horizontal_clusters
                                      cv::Vec4f &line1,
                                      cv::Vec4f &line2,
                                      float &rho1, float &rho2,  // distance from top
                                      float &d1, float &d2)      // distance between intersections on that line
{
    std::vector<int> indexes;
    ISLOOP (m_horizontal_clusters ) {
        indexes.push_back(i);
    }
    // Sort indexes by number of points matching board size
    //int bs = m_boardsize;
    auto &hc(m_horizontal_clusters);
    int bs = m_boardsize;
    // size() is evil. Need to cast to int.
    std::sort( indexes.begin(), indexes.end(),
              [hc,bs](int a, int b)
    {
        return (fabs( bs - (int)hc[a].size()) <
        fabs( bs - (int)hc[b].size()));
    });
    
    PLOG( "cluster sizes");
    ISLOOP (indexes) {
        PLOG( "%ld\n", m_horizontal_clusters[indexes[i]].size());
        PLOG( "%f\n", fabs( bs - (int)hc[indexes[i]].size()));
    }
    PLOG("========");
    
    idx1 = indexes[0];
    idx2 = indexes[1];
    Points &cl1( m_horizontal_clusters[idx1]);
    Points &cl2( m_horizontal_clusters[idx2]);
    cv::Vec2f pl1, pl2;
    line1 = fit_line( cl1);
    segment2polar( line1, pl1);
    polar2segment( pl1, line1);
    line2 = fit_line( cl2);
    segment2polar( line2, pl2);
    polar2segment( pl2, line2);
    rho1 = pl1[0];
    rho2 = pl2[0];
    std::vector<int> x1( cl1.size());
    ISLOOP (cl1) { x1[i] = cl1[i].x; }
    std::vector<int> x2( cl2.size());
    ISLOOP (cl2) { x2[i] = cl2[i].x; }
    d1 = vec_median( x1);
    d2 = vec_median( x2);
}


