// scale_legendre.h
#ifndef SCALE_LEGENDRE_H
#define SCALE_LEGENDRE_H

#include <stdio.h>

// Polynomial declarations
extern double P0[1];
extern double P1[2];
extern double P2[3];
extern double P3[4];
extern double P4[5];
extern double P5[6];
extern double P6[7];
extern double P7[8];
extern double P8[9];
extern double P9[10];
extern double P10[11];

// Products
extern double PP0[1];
extern double PP1[3];
extern double PP2[5];
extern double PP3[7];
extern double PP4[9];
extern double PP5[11];
extern double PP6[13];
extern double PP7[15];
extern double PP8[17];
extern double PP9[19];
extern double PP10[21];

// Integrals
extern double IPP0[2];
extern double IPP1[4];
extern double IPP2[6];
extern double IPP3[8];
extern double IPP4[10];
extern double IPP5[12];
extern double IPP6[14];
extern double IPP7[16];
extern double IPP8[18];
extern double IPP9[20];
extern double IPP10[22];

// Function prototypes
void PrintPoly(char *name, double *P, int N);
void ScalePoly(double scale, double *P, int N);
void FoilPoly(double *Pout, int *Nout, double *A, int NA, double *B, int NB);
void IntegratePoly(double *Pout, int *Nout, double *A, int NA);
double EvalPoly(double *A, int N, double x);

#endif // SCALE_LEGENDRE_H

