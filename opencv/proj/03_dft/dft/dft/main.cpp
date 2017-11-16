//
//  main.cpp
//  dft
//
//  Created by Andreas Hauenstein on 2017-11-01.
//  Copyright © 2017 AHN. All rights reserved.
//

#include <iostream>
#include <vector>
#include <math.h>

#define ILOOP(n) for (int i=0; i < (n); i++ )
#define JLOOP(n) for (int j=0; j < (n); j++ )
#define KLOOP(n) for (int k=0; k < (n); k++ )
#define RLOOP(n) for (int r=0; r < (n); r++ )
#define CLOOP(n) for (int c=0; c < (n); c++ )

#define ISLOOP(n) for (int i=0; i < ((n).size()); i++ )
#define JSLOOP(n) for (int j=0; j < ((n).size()); j++ )
#define KSLOOP(n) for (int k=0; k < ((n).size()); k++ )
#define RSLOOP(n) for (int r=0; r < ((n).size()); r++ )
#define CSLOOP(n) for (int c=0; c < ((n).size()); c++ )

#define SIGN(x) (x)>=0?1:-1
#define PI M_PI

// Triangle, 1.0 at the center, falling to both sides
//---------------------------------------------------
float triang( float val, float center, float width)
{
    float d = fabs( center-val);
    float res = (1.0 - d / width);
    return res > 0 ? res : 0;
}

// One dim clustering. Return the cutting points.
//---------------------------------------------------------------------------
std::vector<float> triangleFreq( std::vector<float> seq, float width)
{
    std::sort( seq.begin(), seq.end(), [](float a, float b) { return a<b; });
    std::vector<float> freq(seq.size());

    ISLOOP (seq) {
        float sum = 0;
        int j=i;
        while( j < seq.size() && triang(seq[j], seq[i], width) > 0) {
            sum += triang(seq[j], seq[i], width);
            j++;
        }
        j=i;
        while( j >= 0 && triang(seq[j], seq[i], width) > 0) {
            sum += triang(seq[j], seq[i], width);
            j--;
        }
        freq[i] = sum;
    } // ISLOOP
    std::vector<float> maxes;
    ISLOOP (freq) {
        if (i==0) continue;
        if (i==freq.size()-1) continue;
        if (freq[i] >= freq[i-1] && freq[i] > freq[i+1]) {
            maxes.push_back( seq[i]);
        }
    }
    std::vector<float> cuts;
    ISLOOP (maxes) {
        if (i==0) continue;
        cuts.push_back( (maxes[i] + maxes[i-1]) / 2.0);
    }
    return cuts;
}

// Take samples and cut them into clusters by the cuts as returned from triangleFreq
//------------------------------------------------------------------------------------
template <typename T, typename G>
void partition( std::vector<T> samples, const std::vector<float> &cuts,
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
    parts = res;
}

//------------------------------------------------------------
template <typename T>
void print_parts( const std::vector<std::vector<T> > &parts)
{
    std::cout << '\n';
    ISLOOP (parts) {
        std::vector<float> part = parts[i];
        if (part.size() == 0) std::cout << "empty";
        for (auto x:part) {
            printf( "%7.2f", x);
        }
        std::cout << '\n';
    }
}

//----------
int main()
{
    std::vector<float> seq = { 1,2,3,10,11,12,100,101,102 };
    std::vector<float> cuts = triangleFreq( seq, 5);
    for (auto x:cuts) {
        printf( "%7.2f", x);
    }
    std::cout << '\n';
    std::vector<std::vector<float> > parts;

    std::vector<float> samples;
    samples = { 2,3,4, 7,10,20,30, 57 };
    partition( samples, cuts, [](float x){return x;}, parts);
    print_parts( parts);

    samples = { };
    partition( samples, cuts, [](float x){return x;}, parts);
    print_parts( parts);

    samples = { 57, 7,10,20,30,  };
    partition( samples, cuts, [](float x){return x;}, parts);
    print_parts( parts);

    return 0;
}

