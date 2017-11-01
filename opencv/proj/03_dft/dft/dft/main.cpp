//
//  main.cpp
//  dft
//
//  Created by Andreas Hauenstein on 2017-11-01.
//  Copyright © 2017 AHN. All rights reserved.
//

#include <iostream>
#include <complex>
#include <math.h>

double PI = M_PI;
typedef std::complex<double> cplx;
cplx I(0.0, 1.0);

//---------------------------------------------------
void _fft(cplx buf[], cplx out[], int n, int step)
{
    if (step < n) {
        _fft( out, buf, n, step * 2);
        _fft( out + step, buf + step, n, step * 2);
        
        for (int i = 0; i < n; i += 2 * step) {
            cplx t = exp( -I * PI * (cplx(i) / cplx(n))) * out[ i + step];
            buf[ i / 2]     = out[i] + t;
            buf[ (i + n)/2] = out[i] - t;
        }
    }
}

//---------------------------
void fft(cplx buf[], int n)
{
    cplx out[n];
    for (int i = 0; i < n; i++) out[i] = buf[i];
    
    _fft( buf, out, n, 1);
}

//----------------------------------------
void show(const char * s, cplx buf[])
{
    printf("%s", s);
    for (int i = 0; i < 8; i++)
        if (!buf[i].imag())
            printf("%g ", buf[i].real());
        else
            printf("(%g, %g) ", buf[i].real(), buf[i].imag());
}

//----------
int main()
{
    PI = atan2(1, 1) * 4;
    cplx buf[] = {{1, 1, 1, 1, 0, 0, 0, 0};
    
    show("Data: ", buf);
    fft(buf, 8);
    show("\nFFT : ", buf);
    
    return 0;
}

