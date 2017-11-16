//
//  LineFixer.cpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-15.
//  Copyright © 2017 AHN. All rights reserved.
//

#include "LineFixer.hpp"


// Replace each synthetic line with a close cluster line.
// If none found, interpolate rho and theta from predecessor.
//----------------------------------------------------------------------------------------------
void LineFixer::fix( const std::vector<cv::Vec4f> &lines, const std::vector<Points> &clusters,
                    std::vector<cv::Vec4f> &lines_out)
{
    char func[] = "clean_lines()";
    std::vector<cv::Vec4f> res_lines;
    
    // Lines through the clusters
    std::vector<cv::Vec4f> clines;
    ISLOOP (clusters) {
        clines.push_back( fit_line( clusters[i]));
    }
    // Convert cluster lines to polar
    std::vector<cv::Vec2f> chlines;
    ISLOOP (clines) {
        cv::Vec2f hline;
        segment2polar( clines[i], hline);
        chlines.push_back( hline);
    }
    // Sort by rho
    std::sort( chlines.begin(), chlines.end(), [](cv::Vec2f a, cv::Vec2f b) { return a[0] < b[0]; });
    
    // Convert our synthetic lines to polar
    std::vector<cv::Vec2f> hlines;
    ISLOOP (lines) {
        cv::Vec2f hline;
        segment2polar( lines[i], hline);
        hlines.push_back( hline);
    }
    // Sort by rho
    std::sort( hlines.begin(), hlines.end(), [](cv::Vec2f a, cv::Vec2f b) { return a[0] < b[0]; });
    
    const float EPS = 7.0; // Could take the median dist here
    float delta_rho = -1;
    int takenj = -1;
    std::vector<bool> match(hlines.size(), false);
    // Replace each hline with a close chline, if you find one
    ISLOOP (hlines) {
        cv::Vec2f hline = hlines[i];
        int minidx = -1;
        float mindist = 1E9;
        JSLOOP (chlines) {
            cv::Vec2f chline = chlines[j];
            if (fabs( chline[0] - hline[0]) < mindist && takenj < j) {
                mindist = fabs( chline[0] - hline[0]);
                minidx = j;
            }
        } // for chlines
        PLOG( "mindist: %.0f", mindist);
        if (mindist < EPS) {
            takenj = minidx;
            PLOG( "%s: replaced line %d with %d", func, i, minidx );
            hlines[i] = chlines[minidx];
            match[i] = true;
        }
    } // for hlines
    
    // Interpolate whoever didn't find a match, low to high rho
    bool init = false;
    float theta = 0;
    delta_rho = -1;
    float old_rho = 0;
    ISLOOP (match) {
        if (i > 0 && match[i] && match[i-1]) {
            init = true;
            theta = hlines[i][1];
            delta_rho = hlines[i][0] - hlines[i-1][0];
            old_rho = hlines[i][0];
            continue;
        }
        if (!init) continue;
        if (!match[i]) {
            PLOG( "Forward interpolated line %d", i);
            hlines[i][1] = theta;
            hlines[i][0] = old_rho + delta_rho;
            match[i] = true;
        }
        old_rho = hlines[i][0];
    }
    
    // Interpolate whoever didn't find a match, high to low rho
    init = false;
    theta = 0;
    delta_rho = -1;
    old_rho = 0;
    const int lim = (int)match.size()-1;
    for (int i = lim; i >= 0; i--) {
        if (i < lim && match[i] && match[i+1]) {
            init = true;
            theta = hlines[i][1];
            delta_rho = hlines[i+1][0] - hlines[i][0];
            old_rho = hlines[i][0];
            continue;
        }
        if (!init) continue;
        if (!match[i]) {
            PLOG( "Backward interpolated line %d", i);
            hlines[i][1] = theta;
            hlines[i][0] = old_rho - delta_rho;
            match[i] = true;
        }
        old_rho = hlines[i][0];
    }
    
    // Convert back to segment
    ISLOOP (hlines) {
        cv::Vec4f line;
        polar2segment( hlines[i], line);
        res_lines.push_back( line);
    }
    lines_out = res_lines;
} // fix()


