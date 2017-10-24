//
//  GrabFuncs.mm
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-10-21.
//  Copyright © 2017 AHN. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "Common.h"
#import "GrabFuncs.h"

typedef std::vector<std::vector<cv::Point> > Contours;
typedef std::vector<cv::Point> Contour;
typedef std::vector<cv::Point> Points;
static cv::RNG rng(12345);

@implementation GrabFuncs
//=========================

//=== General utility funcs ===
//=============================

#pragma mark - Utility funcs
//----------------------------
+ (NSString *) opencvVersion
{
    return [NSString stringWithFormat:@"OpenCV version: %s", CV_VERSION];
}

// Resize image such that min(width,height) = sz
//------------------------------------------------------
void resize(const cv::Mat &src, cv::Mat &dst, int sz)
{
    //cv::Size s;
    int width  = src.cols;
    int height = src.rows;
    float scale;
    if (width < height) scale = sz / (float) width;
    else scale = sz / (float) height;
    cv::resize( src, dst, cv::Size(int(width*scale),int(height*scale)), 0, 0, cv::INTER_AREA);
}

// Calculates the median value of a single channel
//-------------------------------------
int channel_median( cv::Mat channel )
{
    cv::Mat flat = channel.reshape(1,1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);
    double res = sorted.at<uchar>(sorted.size() / 2);
    return res;
}

// Calculates the median value of a vector of int
//-------------------------------------------------
int int_median( std::vector<int> ints )
{
    std::sort( ints.begin(), ints.end(), [](int a, int b) { return a < b; });
    int res = ints[ints.size() / 2];
    return res;
}

//-------------------------------------------------------
void draw_contours( const Contours cont, cv::Mat &dst)
{
    // Draw contours
    for( int i = 0; i< cont.size(); i++ )
    {
        cv::Scalar color = cv::Scalar( rng.uniform(50, 255), rng.uniform(50,255), rng.uniform(50,255) );
        drawContours( dst, cont, i, color, 2, 8);
    }
} // draw_contours()

// Automatic edge detection without parameters
//--------------------------------------------------------------------
void auto_canny( const cv::Mat &src, cv::Mat &dst, float sigma=0.33)
{
    double v = channel_median(src);
    int lower = int(fmax(0, (1.0 - sigma) * v));
    int upper = int(fmin(255, (1.0 + sigma) * v));
    cv::Canny( src, dst, lower, upper);
}

// Mark a point on an image
//--------------------------------------
void draw_point( cv::Point p, cv::Mat &img)
{
    cv::circle( img, p, 10, cv::Scalar(255,0,0), -1);
}

//=== Task specific helpers ===
//=============================

#pragma mark - Task Specific Helpers
//---------------------------------------------
Contours get_contours( const cv::Mat &img)
{
    Contours conts;
    std::vector<cv::Vec4i> hierarchy;
    // Edges
    cv::Mat m;
    auto_canny( img, m, 0.5);
    // Find contours
    findContours( m, conts, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    return conts;
} // get_contours()

// Try to eliminate small and wiggly contours
//---------------------------------------------------------------------
Contours filter_contours( const Contours conts, int width, int height)
{
    Contours large_conts;
    float minArea = width * height / 4000.0;
    std::copy_if( conts.begin(), conts.end(), std::back_inserter(large_conts),
                 [minArea](Contour c){return cv::contourArea(c) > minArea;} );
    
    Contours large_hullarea;
    std::copy_if( large_conts.begin(), large_conts.end(), std::back_inserter(large_hullarea),
                 [minArea](Contour c){
                     Contour hull;
                     cv::convexHull( c, hull);
                     return cv::contourArea(hull) > 0.001; });

    return large_hullarea;
}

// Find the center of the board, which is the median of the contours on the board
//----------------------------------------------------------------------------------
cv::Point get_board_center( const Contours conts, Points &centers)
{
    centers.resize( conts.size());
    int i = 0;
    std::generate( centers.begin(), centers.end(), [conts,&i] {
        Contour c = conts[i++];
        cv::Moments M = cv::moments( c);
        int cent_x = int(M.m10 / M.m00);
        int cent_y = int(M.m01 / M.m00);
        return cv::Point(cent_x, cent_y);
    });
    i=0;
    std::vector<int> cent_x( conts.size());
    std::generate( cent_x.begin(), cent_x.end(), [centers,&i] { return centers[i++].x; } );
    i=0;
    std::vector<int> cent_y( conts.size());
    std::generate( cent_y.begin(), cent_y.end(), [centers,&i] { return centers[i++].y; } );
    int x = int_median( cent_x);
    int y = int_median( cent_y);
    return cv::Point(x,y);
}

//# Remove spurious contours outside the board
//#--------------------------------------------------
//def cleanup_squares(centers, square_cnts, board_center, width, height):
//# Store distance from center for each contour
//# sqdists = [ {'cnt':sq, 'dist':np.linalg.norm( centers[idx] - board_center)}
//#             for idx,sq in enumerate(square_cnts) ]
//# distsorted = sorted( sqdists, key = lambda x: x['dist'])
//
//#ut.show_contours( g_frame, square_cnts)
//sqdists = [ {'cnt':sq, 'dist':ut.contour_maxdist( sq, board_center)}
//           for sq in square_cnts ]
//distsorted = sorted( sqdists, key = lambda x: x['dist'])
//
//# Remove contours if there is a jump in distance
//lastidx = len(distsorted)
//for idx,c in enumerate(distsorted):
//if not idx: continue
//delta = c['dist'] - distsorted[idx-1]['dist']
//#print(c['dist'], delta)
//#ut.show_contours( g_frame, [c['cnt']])
//#print ('dist:%f delta: %f' % (c['dist'],delta))
//if delta > min(width,height) / 10.0:
//lastidx = idx
//break
//
//res = [x['cnt'] for x in distsorted[:lastidx]]
//return res
//

// Remove spurious contours outside the board
//-------------------------------------------------------------------------------
Contours filter_outside_contours( const Points &centers, const Contours &conts,
                                 cv::Point board_center,
                                 int width, int height)
{
    typedef struct dist_idx {
        int idx;
        float dist;
    } dist_idx_t;
    
    std::vector<dist_idx_t> sqdists( conts.size());
    int i=0;
    std::generate( sqdists.begin(), sqdists.end(), [conts,board_center,&i] {
        Contour c = conts[i++];
        float dist = 0; int idx = -1;
        for (cv::Point p: c) {
            float d = sqrt( (p.x - board_center.x)*(p.x - board_center.x) +
                           (p.y - board_center.y)*(p.y - board_center.y));
            if (d > dist) { dist = d; idx = i-1; }
        }
        dist_idx_t res;
        res.dist = dist; res.idx = idx;
        return res;
    });
    
    std::sort( sqdists.begin(), sqdists.end(), [](dist_idx_t a, dist_idx_t b) { return a.dist < b.dist; });
    size_t lastidx = sqdists.size();
    i=0;
    for (dist_idx_t di: sqdists) {
        if (i) {
            float delta = di.dist - sqdists[i-1].dist;
            assert(delta >= 0);
            if (delta > fmin( width, height) / 10.0) {
                lastidx = i;
                break;
            }
        }
    }
    Contours res( lastidx);
    for (i=0; i < lastidx; i++) {
        res[i] = conts[sqdists[i].idx];
    }
    return res;
} // filter_outside_contours()
    

//-----------------------------------------
- (UIImage *) findBoard:(UIImage *)img
{
    // Convert UIImage to Mat
    cv::Mat m;
    UIImageToMat( img, m);
    // Resize
    resize( m, m, 500);
    int width = m.cols; int height = m.rows;
    // Grayscale
    cv::cvtColor( m, m, cv::COLOR_BGR2GRAY);
    // Contours
    cv::Mat drawing = cv::Mat::zeros( m.size(), CV_8UC3 );
    Contours contours = get_contours(m);
    // Straight large ones only
    Contours filtered = filter_contours( contours, width, height);
    draw_contours( filtered, drawing);
    // Only contours on the board
    Points centers;
    cv::Point board_center = get_board_center( filtered, centers);
    draw_point( board_center, drawing);
    Contours inside = filter_outside_contours( centers, filtered, board_center, width, height);
    drawing = cv::Mat::zeros( m.size(), CV_8UC3 );
    draw_contours( inside, drawing);
    draw_point( board_center, drawing);

    // Convert back to UIImage
    UIImage *res = MatToUIImage( drawing);
    return res;
} // drawRectOnImage()



@end
