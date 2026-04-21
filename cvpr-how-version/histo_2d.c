#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmap.h"

#define int64 long long int


// Easy string formatting
char path[4096];
char *cat(const char*str1, const char*str2) {
	(sprintf)(path, "%s%s", str1, str2);
	return path;
}

//---------
// Command Arguments
//---------
char *arg_infile;
char *arg_outfile;
float arg_minx;
float arg_maxx;
int   arg_nbin;
int   arg_maxC1;

//---------
// Histogram file
//---------
Binfile datafile;
float *X;
int64 N;
int64 C;
int64 sY;
int64 sX;
int64 CsYsX;
int64 sYsX;


//---------
// Output histogram
//---------
Binfile outhist;
int *histo;

int main(int argc, char**argv)
{
	int64 n,c,c1,c2,y,x,b1,b2;

	//----------------------
	// Read Command Arugments
	//----------------------
	if (argc<7) {
		printf("Usage:\n");
		printf("   ./histo_2d infile outfile minx maxx nbin maxC1\n");
		printf("\n");
		printf("   infile:      in .bin feature file\n");
		printf("   outfile:     output binary file\n");
		printf("   minx,maxx:   x-interval for histogram binning\n");
		printf("   nbin:        how many histogram bins\n");
		printf("\n");
		exit(1);
	}
	arg_infile  =      argv[1];
	arg_outfile =      argv[2];
	arg_minx    = atof(argv[3]);
	arg_maxx    = atof(argv[4]);
	arg_nbin    = atoi(argv[5]);
	arg_maxC1   = atoi(argv[6]);

	printf("arg_infile: %s\n",  arg_infile);
	printf("arg_outfile: %s\n", arg_outfile);
	printf("arg_minx %f\n", arg_minx);
	printf("arg_maxx %f\n", arg_maxx);
	printf("arg_nbin %d\n", arg_nbin);
	printf("arg_maxC1 %d\n", arg_maxC1);

	//----------------------
	// Memory map the data histogram file
	//----------------------
	datafile = MapBinfileR(arg_infile);
	X = (float*)datafile.data;
	N = datafile.shape[0];
	C = datafile.shape[1];
	sY = datafile.shape[2];
	sX = datafile.shape[3];
	CsYsX = C*sY*sX;
	sYsX = sY*sX;

	printf("N %lld C %lld sY %lld sX %lld CsYsX %lld sYsX %lld\n", N, C, sY, sX, CsYsX, sYsX);


	printf("----------------------\n");
	printf(" Memory map the histograms\n");
	printf("----------------------\n");
		
	int outhist_shape[5];
	outhist_shape[0] = arg_maxC1;
	outhist_shape[1] = C;
	outhist_shape[2] = arg_nbin;
	outhist_shape[3] = arg_nbin;
	outhist_shape[4] = 0;
	outhist = MapBinfileRW(arg_outfile, outhist_shape, INT32);
	int64 BB  = arg_nbin*arg_nbin;
	int64 CBB = C*BB;
	int64 CCBB = arg_maxC1*CBB;
	histo = (int*)outhist.data;
	
	printf("fd %d BB %lld CBB %lld CCBB %lld histo %p\n", outhist.fd, BB, CBB, CCBB, histo);
	
	printf("----------------------\n");
	printf(" Zero out the histogram\n");
	printf("----------------------\n");
	int64 i;
	for (i=0; i<CCBB; i++)
		histo[i] = 0;

	printf("----------------------\n");
	printf(" Compute the histograms\n");
	printf("----------------------\n");

	float scaling = (float)arg_nbin / (arg_maxx-arg_minx);

	// For every data element
	for (n=0; n<N; n++) {
		if (n%100==0) printf("n %lld/%lld\n", n, N);
	for (c1=0; c1<arg_maxC1; c1++) {
	for (c2=0; c2<C; c2++) {
	for (y=0; y<sY; y++) {
	for (x=0; x<sX; x++) {

		int64 idx1 = n*CsYsX + c1*sYsX + y*sX + x;
		int64 idx2 = n*CsYsX + c2*sYsX + y*sX + x;

		//-------------------
		// Identify bin 1
		//-------------------

		// Which bin (floating point) ?
		float val1 = X[idx1];
		float fbin1 = (val1-arg_minx) * scaling;
		if (fbin1<0) {
			//printf("WARNING fbin1 out of bounds val1 %f fbin1 %f\n", val1, fbin1);
			continue;
		}
		else if (fbin1>=arg_nbin) {
			//printf("WARNING fbin1 out of bounds val1 %f fbin1 %f\n", val1, fbin1);
			continue;
		}

		// Which bin (integer)
		int ibin1 = (int)fbin1;
		if (ibin1<0)
			ibin1=0;
		if (ibin1>arg_nbin-1)
			ibin1=arg_nbin-1;

		//-------------------
		// Identify bin 2
		//-------------------

		// Which bin (floating point) ?
		float val2 = X[idx2];
		float fbin2 = (val2-arg_minx) * scaling;
		if (fbin2<0) {
			//printf("WARNING fbin2 out of bounds val2 %f fbin2 %f\n", val2, fbin2);
			continue;
		}
		else if (fbin2>=arg_nbin) {
			//printf("WARNING fbin2 out of bounds val2 %f fbin2 %f\n", val2, fbin2);
			continue;
		}

		// Which bin (integer)
		int ibin2 = (int)fbin2;
		if (ibin2<0)
			ibin2=0;
		if (ibin2>arg_nbin-2)
			ibin2=arg_nbin-2;

		// Bin the value
		i64 outidx = c1*CBB + c2*BB + ibin1*arg_nbin + ibin2;
		histo[outidx]++;

	}}}}}


	printf("----------------------\n");
	printf(" Unmap the dataset features\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);

	printf("----------------------\n");
	printf(" Unmap the output histogram\n");
	printf("----------------------\n");
	UnmapBinfile(outhist);

	printf("Success\n");

	return 0;
}
