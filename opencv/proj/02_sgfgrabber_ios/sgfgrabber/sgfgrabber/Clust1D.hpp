//
//  Clust1D.hpp
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2017-11-16.
//  Copyright © 2017 AHN. All rights reserved.
//

// Cluster 1D numbers using simple KDE (Kernel density estimation) approach

#ifndef Clust1D_hpp
#define Clust1D_hpp

#include <iostream>
#include "Common.hpp"

class Clust1D
//================
{
public:
    // One dim clustering. Return the cutting points.
    //---------------------------------------------------------------------------
    template <typename T, typename G>
    static inline std::vector<float> cluster( const std::vector<T> &seq_, float width, G getter)
    {
        const float SMOOTH = 3.0;
        typedef float(*WinFunc)(float,float,float);
        WinFunc winf = bell;
        
        std::vector<float> vals;
        ISLOOP (seq_) { vals.push_back( getter(seq_[i] )); }
        std::sort( vals.begin(), vals.end(), [](float a, float b) { return a<b; });
        std::vector<float> freq(vals.size());
        
        // Distance weighted sum of number of samples to the left and to the right
        ISLOOP (vals) {
            float sum = 0; int j;
            j=i;
            while( j < vals.size() && winf( vals[j], vals[i], width) > 0) {
                sum += winf( vals[j], vals[i], width);
                j++;
            }
            j=i;
            while( j >= 0 && winf( vals[j], vals[i], width) > 0) {
                sum += winf( vals[j], vals[i], width);
                j--;
            }
            freq[i] = sum;
        } // ISLOOP
        
        // Convert to discrete pdf, missing values set to -1
        int mmax = ROUND( vec_max( vals)) + 1;
        std::vector<float> pdf(mmax,-1);
        ISLOOP (freq) {
            pdf[ROUND(vals[i])] = freq[i];
        }
        pdf = smooth( pdf,SMOOTH);

        std::vector<float> maxes;
        ISLOOP (pdf) {
            if (i < 1) continue;
            if (i >= pdf.size()-1) continue;
            if (pdf[i] >= pdf[i-1] && pdf[i] > pdf[i+1]) {
                maxes.push_back( i);
            }
        }
        std::vector<float> cuts;
        ISLOOP (maxes) {
            if (i==0) continue;
            cuts.push_back( (maxes[i] + maxes[i-1]) / 2.0);
        }
        return cuts;
        
    } // cluster()
    
    // Use the cuts returned by cluster() to classify new samples
    //---------------------------------------------------------------------------
    template <typename T, typename G>
    static inline void classify( std::vector<T> samples, const std::vector<float> &cuts, int minsz,
                   G getter,
                   std::vector<std::vector<T> > &parts)
    {
        std::sort( samples.begin(), samples.end(),
                  [getter](T a, T b) { return getter(a) < getter(b); });
        std::vector<std::vector<T> > res(cuts.size()+1);
        int cut=0;
        std::vector<T> part;
        ISLOOP (samples) {
            T s = samples[i];
            float x = getter(s);
            if (cut < cuts.size()) {
                if (x > cuts[cut]) {
                    res[cut] = part;
                    part.clear();
                    cut++;
                }
            }
            part.push_back( s);
        }
        res[cut] = part;
        
        // Eliminate small clusters
        std::vector<std::vector<T> > big;
        ISLOOP (res) {
            if (res[i].size() >= minsz) {
                big.push_back( res[i]);
            }
        }

        parts = big;
    } // classify()


private:
    // Smooth
    //---------
    static inline std::vector<float> smooth( const std::vector<float> &seq, float width = 3)
    {
        std::vector<float> res( seq.size());
        ISLOOP (seq) {
            float ssum = 0;
            for (int k = i-width; k <= i+width; k++) {
                if (k < 0) continue;
                if (k >= seq.size()) continue;
                float weight = triang( i, k, width);
                ssum += seq[k] * weight;
            }
            res[i] = ssum;
        }
        return res;
    }
    
    // Various window funcs. Bell works best.
    //=========================================
    // Triangle, 1.0 at the center, falling to both sides
    //----------------------------------------------------------
    static inline float triang( float val, float center, float width)
    {
        float d = fabs( center-val);
        float res = (1.0 - d / width);
        return res > 0 ? res : 0;
    }
    // Rectangle, 1.0 at the center, extends by width both sides
    //-----------------------------------------------------------
    static inline float rect( float val, float center, float width)
    {
        float d = fabs( center-val);
        float res = (1.0 - d / width);
        return res > 0 ? 1 : 0;
    }
    // Bell (Gaussian)
    //-----------------------------------------------------------
    static inline float bell( float val, float center, float sigma)
    {
        float d = center-val;
        float bell = exp(-(d*d) / 2*sigma);
        return bell;
    }
}; // class Clust1D


#endif /* Clust1D_hpp */
