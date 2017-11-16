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
    // Find predominant slopes
    find_slopes( hslope, vslope);
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
                      [hslope](cv::Point p) { return dist_point_line(p, hslope); },
                      m_horizontal_clusters);
    
    std::vector<float> vertical_cuts   = Clust1D::cluster( m_cloud, wwidth,
                                                          [vslope](cv::Point p) { return fabs(dist_point_line(p, vslope)); });
    Clust1D::classify( m_cloud, vertical_cuts, min_points,
                      [vslope](cv::Point p) { return dist_point_line(p, vslope); },
                      m_vertical_clusters);

    float delta_wavelen_h, slope_h, median_rho_h;
    find_rhythm( m_horizontal_clusters,
                m_wavelen_h,
                delta_wavelen_h,
                slope_h,
                median_rho_h);

    float delta_wavelen_v, slope_v, median_rho_v;
    find_rhythm( m_vertical_clusters,
                m_wavelen_v,
                delta_wavelen_v,
                slope_v,
                median_rho_v);
    
    std::vector<cv::Vec4f> lines;
    find_lines( m_imgSize.height, m_wavelen_h, delta_wavelen_h, slope_h, median_rho_h, horizontal_lines);
    find_lines( m_imgSize.width , m_wavelen_v, delta_wavelen_v, slope_v, median_rho_v, vertical_lines);

}

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
    hslope = avg_slope_line( hlines);
    vslope = avg_slope_line( vlines);
} // find_slopes()

// Find phase, wavelength etc of a family of lines.
// Each cluster has a bunch of points which are probably on the same line.
//----------------------------------------------------------------------------
void LineFinder::find_rhythm( const std::vector<Points> &clusters,
                             float &wavelength,
                             float &delta_wavelength,
                             float &slope,
                             float &median_rho
                             )
{
    typedef struct { float dist; float slope; } DistSlope;
    std::vector<cv::Vec4f> lines;
    // Lines through the clusters
    ISLOOP (clusters) {
        lines.push_back( fit_line( clusters[i]));
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
    ISLOOP (clusters) {
        std::vector<float> ds;
        JSLOOP (clusters[i]) {
            cv::Point p = clusters[i][j];
            float d = dist_point_line( p, median_hline);
            ds.push_back( d);
        }
        float dist = vec_median( ds);
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


