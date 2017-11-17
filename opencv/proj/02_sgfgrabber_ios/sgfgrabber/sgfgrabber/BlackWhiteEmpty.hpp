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

class BlackWhiteEmpty
//=====================
{
    
    // Type to hold a feature vector at a board position
    typedef struct feat {
        std::string key;
        int x,y;     // Pixel pos
        std::vector<float> features;
    } Feat;
    
public:
    enum {BBLACK=-1, EEMPTY=0, WWHITE=1 }; // Black Empty White
    //---------------------------------------------------------------------------
    static inline std::vector<int> get_diagram( const cv::Mat &img, // small, color
                                               std::vector<cv::Vec4f> horizontal_lines, // sorted cleaned lines (by LineFixer)
                                               float wavelen_h, // approximate dist between lines
                                               std::vector<cv::Vec4f> vertical_lines,
                                               float wavelen_v,
                                               int boardsize, // 9,13,19
                                               Points board_corners,
                                               Points &diagram_intersections) // out
    {
        std::vector<int> diagram(boardsize*boardsize, EEMPTY);
        diagram_intersections = Points(boardsize*boardsize);
        cv::Mat gray;
        cv::cvtColor( img, gray, cv::COLOR_BGR2GRAY);
        
        // Get pixel pos for each potential board intersection
        std::map<std::string, cv::Point> intersections;
        RSLOOP (horizontal_lines) {
            CSLOOP (vertical_lines) {
                Point2f pf = intersection( horizontal_lines[r], vertical_lines[c]);
                cv::Point p( ROUND(pf.x), ROUND(pf.y));
                intersections[rc_key(r,c)] = p;
            }
        }
        // Compute features for each potential board intersection
        std::map<std::string, Feat> black_features;
        get_black_features( gray, intersections, wavelen_h, wavelen_v, black_features);

        // Find the best grid
        cv::Point2f bcenter = get_center( board_corners);
        double mindist = 1E9;
        int minr = -1, minc = -1;
        RSLOOP (horizontal_lines) {
            CSLOOP (vertical_lines) {
                cv::Point2f gridcenter;
                if (!subgrid_center( r, c, boardsize, intersections, gridcenter)) continue;
                double d = cv::norm( bcenter - gridcenter);
                if (d < mindist) {
                    mindist = d;
                    minr = r; minc = c;
                }
            }
        }
        std::vector<Feat> black_features_subgrid;
        get_subgrid_features( minr, minc, boardsize, black_features, black_features_subgrid);
        
        // Black stones
        if (black_features_subgrid.size()) {
            Feat minelt = *(std::min_element( black_features_subgrid.begin(), black_features_subgrid.end(),
                                             [](Feat &a, Feat &b){ return a.features[0] < b.features[0]; } )) ;
            float thresh = minelt.features[0] * 4;
            ISLOOP( black_features_subgrid) {
                Feat &f(black_features_subgrid[i]);
                diagram_intersections[i] = cv::Point(f.x, f.y);
                if (f.features[0] < thresh) {
                    diagram[i] = BBLACK;
                }
            }
        } // if black_features
        return diagram;
    }
    
private:
    // Average pixel value around center of each intersection is black indicator.
    //--------------------------------------------------------------------------------------
    inline static void get_black_features( const cv::Mat &img, std::map<std::string, cv::Point> intersections,
                                          int wavelen_h, int wavelen_v,
                                          std::map<std::string, Feat> &res )
    {
        int dx = ROUND( wavelen_h/4.0);
        int dy = ROUND( wavelen_v/4.0);

        for (auto &inter: intersections)
        {
            std::string key = inter.first;
            cv::Point p = inter.second;
            Feat f;
            f.key = key; f.x = p.x; f.y = p.y;
            cv::Rect rect( p.x - dx, p.y - dy, 2*dx+1, 2*dy+1 );
            f.features.clear();
            if (0 <= rect.x &&
                0 <= rect.width &&
                rect.x + rect.width <= img.cols &&
                0 <= rect.y &&
                0 <= rect.height &&
                rect.y + rect.height <= img.rows)
            {
                cv::Mat hood = cv::Mat( img, rect);
                float area = hood.rows * hood.cols;
                cv::Scalar ssum = cv::sum( hood);
                float brightness = ssum[0] / area;
                f.features.push_back( brightness);
                res[key] = f;
            }
        } // for intersections
    } // get_black_features()
    
    // Dictionary key for r,c
    //----------------------------------------------
    inline static std::string rc_key (int r, int c)
    {
        static char buf[100];
        sprintf( buf, "%d_%d", r,c);
        return std::string (buf);
    }

//    // Linear index for key "r_c"
//    //-------------------------------------------------------------
//    inline static int linear (std::string rc_key, int boardsize)
//    {
//        std::istringstream f(rc_key);
//        std::string s;
//        getline( f, s, '_');
//        int r = atoi( s.c_str());
//        getline( f, s, '_');
//        int c = atoi( s.c_str());
//        int res = r * boardsize + c;
//        return res;
//    }

    // Get a list of features for the intersections in a subgrid
    // with r,c as upper left corner.
    //------------------------------------------------------------
    inline static bool get_subgrid_features( int top_row, int left_col, int boardsize,
                              std::map<std::string, Feat> &features,
                              std::vector<Feat> &subgrid)
    {
        subgrid.clear();
        RLOOP (boardsize) {
            CLOOP (boardsize) {
                std::string key = rc_key( top_row + r, left_col + c);
                if (!features.count( key)) {
                    return false;
                }
                subgrid.push_back( features[key]);
            }
        }
        return true;
    } // get_subgrid_features()
    
    // Try to get subgrid center with r,c as upper left corner.
    //------------------------------------------------------------
    inline static bool subgrid_center( int top_row, int left_col, int boardsize,
                                      std::map<std::string, cv::Point> &intersections,
                                      cv::Point2f &center)
    {
        double avg_x, avg_y;
        avg_x = 0; avg_y = 0;
        RLOOP (boardsize) {
            CLOOP (boardsize) {
                std::string key = rc_key( top_row + r, left_col + c);
                if (!intersections.count( key)) {
                    return false;
                }
                avg_x += intersections[key].x;
                avg_y += intersections[key].y;
            }
        }
        center.x = avg_x / (boardsize*boardsize);
        center.y = avg_y / (boardsize*boardsize);
        return true;
    } // subgrid_center()
}; // class BlackWhiteEmpty
    

#endif /* BlackWhiteEmpty_hpp */
