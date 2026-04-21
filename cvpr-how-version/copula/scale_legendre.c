#include <stdio.h>
#include <stdlib.h>

//
// Polynomials
//

//                  1      x     x2     x3     x4       x5      x6      x7      x8      x9      x10
double P0[1]   = { 1.0};
double P1[2]   = { 0.0,   1.0};
double P2[3]   = {-1.0,   0.0,   3.0};
double P3[4]   = { 0.0,  -3.0,   0.0,   5.0};
double P4[5]   = { 3.0,   0.0, -30.0,   0.0,  35.0};
double P5[6]   = { 0.0,  15.0,   0.0, -70.0,   0.0,    63};
double P6[7]   = {-5.0,   0.0, 105.0,   0.0, -315.0,   0.0,  231.0};
double P7[8]   = { 0.0,  -35.0,  0.0, 315.0,   0.0, -693.0,   0.0,  429.0};
double P8[9]   = {35.0,   0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0,   0.0,   6435.0};
double P9[10]  = { 0.0, 315.0,  0.0, -4620.0, 0.0,  18018.0,   0.0, -25740.0,   0.0,  12155.0};
double P10[11] = {-63.0,  0.0,  3465.0, 0.0, -30030.0, 0.0, 90090.0, 0.0, -109395.0,  0.0,   46189.0};

//
// Products
//
double PP0[1];
double PP1[3];
double PP2[5];
double PP3[7];
double PP4[9];
double PP5[11];
double PP6[13];
double PP7[15];
double PP8[17];
double PP9[19];
double PP10[21];
int NPP0 = 1;
int NPP1 = 3;
int NPP2 = 5;
int NPP3 = 7;
int NPP4 = 9;
int NPP5 = 11;
int NPP6 = 13;
int NPP7 = 15;
int NPP8 = 17;
int NPP9 = 19;
int NPP10 = 21;

//
// Integrals
//
double IPP0[2];
double IPP1[4];
double IPP2[6];
double IPP3[8];
double IPP4[10];
double IPP5[12];
double IPP6[14];
double IPP7[16];
double IPP8[18];
double IPP9[20];
double IPP10[22];
int INPP0 = 2;
int INPP1 = 4;
int INPP2 = 6;
int INPP3 = 8;
int INPP4 = 10;
int INPP5 = 12;
int INPP6 = 14;
int INPP7 = 16;
int INPP8 = 18;
int INPP9 = 20;
int INPP10 = 22;


void PrintPoly(char*name, double *P, int N)
{
	int i;
	printf("--------------\n");
	printf("%s   %d\n", name, N);

	for (i=0; i<N; i++)
		printf(" %f  x^%d\n", P[i], i);
}

void ScalePoly(double scale, double *P, int N)
{
	int i;
	for (i=0; i<N; i++)
		P[i] *= scale;
}

void FoilPoly(double *Pout, int *Nout, double *A, int NA, double *B, int NB)
{
	int i,a,b;
	int N = NA + NB - 1;
	for (i=0; i<N; i++)
		Pout[i] = 0;
		
	for (a=0; a<NA; a++) {
		for (b=0; b<NB; b++) {
			i = a+b;
			Pout[i] += A[a] * B[b];
		}
	}
			
	*Nout = N;
}


void IntegratePoly(double *Pout, int *Nout, double *A, int NA)
{
	int a;
	int N = NA + 1;
	for (a=0; a<NA; a++) {
		Pout[a+1] = A[a] / (a+1);
	}
	*Nout = N;
}

double EvalPoly(double *A, int N, double x)
{
	int i;
	double sum = 0.0;
	for (i=N-1; i>=0; i--) {
		sum = sum * x + A[i];
	}
	return sum;
}

void input() {
	printf("press enter\n");
	fgetc(stdin);
}

int main()
{
	printf("---------------------\n");
	printf(" Print the original polynomials\n");
	printf("---------------------\n");
	PrintPoly("P0", P0, 1);
	PrintPoly("P1", P1, 2);
	PrintPoly("P2", P2, 3);
	PrintPoly("P3", P3, 4);
	PrintPoly("P4", P4, 5);
	PrintPoly("P5", P5, 6);
	PrintPoly("P6", P6, 7);
	PrintPoly("P7", P7, 8);
	PrintPoly("P8", P8, 9);
	PrintPoly("P9", P9, 10);
	PrintPoly("P10", P10, 11);
	input();

	printf("---------------------\n");
	printf(" Rescale the polynomials\n");
	printf("---------------------\n");
	ScalePoly(1.0, P0, 1);
	ScalePoly(1.0, P1, 2);
	ScalePoly(1.0 / 2.0, P2, 3);
	ScalePoly(1.0 / 2.0, P3, 4);
	ScalePoly(1.0 / 8.0, P4, 5);
	ScalePoly(1.0 / 8.0, P5, 6);
	ScalePoly(1.0 / 16.0, P6, 7);
	ScalePoly(1.0 / 16.0, P7, 8);
	ScalePoly(1.0 / 128.0, P8, 9);
	ScalePoly(1.0 / 128.0, P9, 10);
	ScalePoly(1.0 / 256.0, P10, 11);
	input();

	printf("---------------------\n");
	printf(" Print the rescaled polynomials\n");
	printf("---------------------\n");
	PrintPoly("P0", P0, 1);
	PrintPoly("P1", P1, 2);
	PrintPoly("P2", P2, 3);
	PrintPoly("P3", P3, 4);
	PrintPoly("P4", P4, 5);
	PrintPoly("P5", P5, 6);
	PrintPoly("P6", P6, 7);
	PrintPoly("P7", P7, 8);
	PrintPoly("P8", P8, 9);
	PrintPoly("P9", P9, 10);
	PrintPoly("P10", P10, 11);
	input();

	printf("---------------------\n");
	printf(" Multiply the polynomials\n");
	printf("---------------------\n");
	FoilPoly(PP0,  &NPP0,  P0,  1,  P0,  1);
	FoilPoly(PP1,  &NPP1,  P1,  2,  P1,  2);
	FoilPoly(PP2,  &NPP2,  P2,  3,  P2,  3);
	FoilPoly(PP3,  &NPP3,  P3,  4,  P3,  4);
	FoilPoly(PP4,  &NPP4,  P4,  5,  P4,  5);
	FoilPoly(PP5,  &NPP5,  P5,  6,  P5,  6);
	FoilPoly(PP6,  &NPP6,  P6,  7,  P6,  7);
	FoilPoly(PP7,  &NPP7,  P7,  8,  P7,  8);
	FoilPoly(PP8,  &NPP8,  P8,  9,  P8,  9);
	FoilPoly(PP9,  &NPP9,  P9,  10, P9,  10);
	FoilPoly(PP10, &NPP10, P10, 11, P10, 11);
	input();

	printf("---------------------\n");
	printf(" Print the multiplied polynomials\n");
	printf("---------------------\n");
	PrintPoly("PP0",  PP0,  NPP0);
	PrintPoly("PP1",  PP1,  NPP1);
	PrintPoly("PP2",  PP2,  NPP2);
	PrintPoly("PP3",  PP3,  NPP3);
	PrintPoly("PP4",  PP4,  NPP4);
	PrintPoly("PP5",  PP5,  NPP5);
	PrintPoly("PP6",  PP6,  NPP6);
	PrintPoly("PP7",  PP7,  NPP7);
	PrintPoly("PP8",  PP8,  NPP8);
	PrintPoly("PP9",  PP9,  NPP9);
	PrintPoly("PP10", PP10, NPP10);
	input();

	printf("---------------------\n");
	printf(" Integrate the polynomials\n");
	printf("---------------------\n");
	IntegratePoly(IPP0,  &INPP0,  PP0,  NPP0);
	IntegratePoly(IPP1,  &INPP1,  PP1,  NPP1);
	IntegratePoly(IPP2,  &INPP2,  PP2,  NPP2);
	IntegratePoly(IPP3,  &INPP3,  PP3,  NPP3);
	IntegratePoly(IPP4,  &INPP4,  PP4,  NPP4);
	IntegratePoly(IPP5,  &INPP5,  PP5,  NPP5);
	IntegratePoly(IPP6,  &INPP6,  PP6,  NPP6);
	IntegratePoly(IPP7,  &INPP7,  PP7,  NPP7);
	IntegratePoly(IPP8,  &INPP8,  PP8,  NPP8);
	IntegratePoly(IPP9,  &INPP9,  PP9,  NPP9);
	IntegratePoly(IPP10, &INPP10, PP10, NPP10);
	input();


	printf("---------------------\n");
	printf(" Print the integrated polynomials\n");
	printf("---------------------\n");
	PrintPoly("IPP0",  IPP0,  INPP0);
	PrintPoly("IPP1",  IPP1,  INPP1);
	PrintPoly("IPP2",  IPP2,  INPP2);
	PrintPoly("IPP3",  IPP3,  INPP3);
	PrintPoly("IPP4",  IPP4,  INPP4);
	PrintPoly("IPP5",  IPP5,  INPP5);
	PrintPoly("IPP6",  IPP6,  INPP6);
	PrintPoly("IPP7",  IPP7,  INPP7);
	PrintPoly("IPP8",  IPP8,  INPP8);
	PrintPoly("IPP9",  IPP9,  INPP9);
	PrintPoly("IPP10", IPP10, INPP10);
	input();

	printf("---------------------\n");
	printf(" EVALUATE THE POLYNOMIALS\n");
	printf("---------------------\n");
	printf("mu0 %.16f\n",  EvalPoly(IPP0, INPP0, 1.0) - EvalPoly(IPP0, INPP0,-1.0));
	printf("mu1 %.16f\n",  EvalPoly(IPP1, INPP1, 1.0) - EvalPoly(IPP1, INPP1,-1.0));
	printf("mu2 %.16f\n",  EvalPoly(IPP2, INPP2, 1.0) - EvalPoly(IPP2, INPP2,-1.0));
	printf("mu3 %.16f\n",  EvalPoly(IPP3, INPP3, 1.0) - EvalPoly(IPP3, INPP3,-1.0));
	printf("mu4 %.16f\n",  EvalPoly(IPP4, INPP4, 1.0) - EvalPoly(IPP4, INPP4,-1.0));
	printf("mu5 %.16f\n",  EvalPoly(IPP5, INPP5, 1.0) - EvalPoly(IPP5, INPP5,-1.0));
	printf("mu6 %.16f\n",  EvalPoly(IPP6, INPP6, 1.0) - EvalPoly(IPP6, INPP6,-1.0));
	printf("mu7 %.16f\n",  EvalPoly(IPP7, INPP7, 1.0) - EvalPoly(IPP7, INPP7,-1.0));
	printf("mu8 %.16f\n",  EvalPoly(IPP8, INPP8, 1.0) - EvalPoly(IPP8, INPP8,-1.0));
	printf("mu9 %.16f\n",  EvalPoly(IPP9, INPP9, 1.0) - EvalPoly(IPP9, INPP9,-1.0));
	printf("mu10 %.16f\n", EvalPoly(IPP10,INPP10,1.0) - EvalPoly(IPP10,INPP0,-1.0));

	printf("Done!\n");
	return 0.0;
}
