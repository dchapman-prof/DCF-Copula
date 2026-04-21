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

void *Calloc(size_t N, size_t size) {
	void *data = calloc(N,size);
	if (data==NULL) {
		printf("ERROR Calloc failed out of memory\n");
		exit(1);
	}
	return data;
}

//---------
// Command Arguments
//---------
char *arg_infile;
char *arg_outfile;
int   arg_nbin=100000;
int   arg_nquant=10000;

//---------
// Histogram file
//---------
Binfile datafile;
float *X;
int N;
int C;
int sY;
int sX;


int main(int argc, char**argv)
{
	int n,c,y,x,b,q;

	//----------------------
	// Read Command Arugments
	//----------------------
	if (argc<3) {
		printf("Usage:\n");
		printf("   ./quantile infile outfile (nbin nquant)\n");
		printf("\n");
		printf("   infile:      in .bin feature file\n");
		printf("   outfile:     out .csv file\n");
		printf("   nbin:	how many histogram bins (default 1000000)\n");
		printf("   nquant:      how many quantiles (defualt 10001)\n");
		printf("\n");
		exit(1);
	}
	arg_infile  =      argv[1];
	arg_outfile =      argv[2];
		if (argc>3)
			arg_nbin = atoi(argv[3]);
		if (argc>4)
			arg_nquant = atoi(argv[4]);

	printf("arg_infile: %s\n",  arg_infile);
	printf("arg_outfile: %s\n", arg_outfile);
	printf("arg_nbin %d\n", arg_nbin);
		printf("arg_nquant %d\n", arg_nquant);

	printf("----------------------\n");
	printf(" Memory map the .bin features file\n");
	printf("----------------------\n");
	datafile = MapBinfileR(arg_infile);
	X = (float*)datafile.data;
	N = datafile.shape[0];
	C = datafile.shape[1];
	sY = datafile.shape[2];
	sX = datafile.shape[3];

	printf("N %d C %d sY %d sX %d\n", N, C, sY, sX);


	printf("----------------------\n");
	printf(" Allocate the histograms\n");
	printf("----------------------\n");
	float *minx      = (float*)Calloc(C, sizeof(float));
	float *maxx      = (float*)Calloc(C, sizeof(float));
	double *scaling  = (double*)Calloc(C, sizeof(double));
	int64 *n_zero    = (int64*)Calloc(C, sizeof(int64));
	int64 *n_nonzero = (int64*)Calloc(C, sizeof(int64));
	float **quant = (float**)Calloc(C, sizeof(float*));
	for (c=0; c<C; c++)
		quant[c] = (float*)Calloc(arg_nquant, sizeof(float));
	int64 **histo = (int64**)Calloc(C, sizeof(int64*));
	for (c=0; c<C; c++)
		histo[c] = (int64*)Calloc(arg_nbin, sizeof(int64));

	printf("----------------------\n");
	printf(" Calculate min and max values per channel\n\n");
	printf("----------------------\n");

	// For every data element find min and max
	int64 idx=0;
	for (n=0; n<N; n++) {
	for (c=0; c<C; c++) {
	for (y=0; y<sY; y++) {
	for (x=0; x<sX; x++) {
		float val = X[idx];
		
		if (n==0 && y==0 && x==0) {
			minx[c] = val;
			maxx[c] = val;
		}
		else {
			if (val<minx[c])
				minx[c] = val;
			if (val > maxx[c])
				maxx[c] = val;
		}
		idx++;
	}}}}
	
	// calculate scaling values
	for (c=0; c<C; c++) {
		scaling[c] = (double)arg_nbin / (double)(maxx[c]-minx[c]);
		
		printf("c %d minx %f maxx %f scaling %f\n", c, minx[c], maxx[c], scaling[c]);
	}

	printf("----------------------\n");
	printf(" Compute the histograms\n");
	printf("----------------------\n");
	
	// For every data element
	idx=0;
	for (n=0; n<N; n++) {
	for (c=0; c<C; c++) {
	for (y=0; y<sY; y++) {
	for (x=0; x<sX; x++) {

		// Which bin (floating point) ?
		float val = X[idx];
		float fbin = (val-minx[c]) * scaling[c];
		//if (fbin<0) {
		//	printf("WARNING fbin out of bounds val %f fbin %f\n", val, fbin);
		//}
		//else if (fbin>=arg_nbin) {
		//	printf("WARNING fbin out of bounds val %f fbin %f\n", val, fbin);
		//}

		// Which bin (integer)
		int ibin = (int)fbin;
		if (ibin<0)
			ibin=0;
		if (ibin>arg_nbin-1)
			ibin=arg_nbin-1;

		// If this is a non-zero value
		if (val>=0.00001 || val<-0.00001) {

			// increase the histogram by one in that bin
			histo[c][ibin]++;
			n_nonzero[c]++;
		} else
			n_zero[c]++;    // otherwise it is a zero value

		// Next datafile element
		idx++;
	}}}}


	printf("----------------------\n");
	printf(" Convert histograms into CDFs\n");
	printf("----------------------\n");
	
	for (c=0; c<C; c++) {
		for (b=1; b<arg_nbin; b++) {
			histo[c][b] = histo[c][b] + histo[c][b-1];
		}
	}
	

	printf("----------------------\n");
	printf(" Convert CDFs into quantiles\n");
	printf("----------------------\n");
	
	for (c=0; c<C; c++) {
	
		// How many points ?
		int64 npnt = histo[c][arg_nbin-1];
		double inv_npnt = 1.0 / (double)npnt;
		
		// What are the start and end x and y values of the CDF
		b = 0;
		double xstep = (maxx[c]-minx[c]) / (double)arg_nbin;
		double x0 = minx[c];
		double x1 = minx[c]+xstep;
		double y0 = 0.0;
		double y1 = (double)histo[c][0] * inv_npnt;

		// Loop through all of the quantiles and bins
		double qscale = 1.0 / (arg_nquant-1);
		for (q=0; q<arg_nquant; q++)
		{
			// What is the y value of the quantile ?
			double qy = q * qscale;
			
			// Is this the best histogram bin to be looking at?
			while (b<arg_nbin && y1<qy) {
				b++;                       // move to next bin . . .
				x0 = x1;
				x1 = x1 + xstep;
				y0 = y1;
				y1 = (double)histo[c][b] * inv_npnt;
			}
			
			// Find the x quantile
			double qx = x0;     // avoid divide by zero
			if (y1-y0 > 0.00000001) {
				// Linear interpolation
				double delta = (qy-y0) / (y1-y0);
				qx = x0 + delta * (x1-x0);
			}
			
			// Assign the quantile value
			quant[c][q] = qx;
		}
	}

	printf("----------------------\n");
	printf(" Unmap the data histogram\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);

	printf("----------------------\n");
	printf(" Write out the quantiles\n");
	printf("----------------------\n");

	FILE *f = fopen(arg_outfile, "w");
	if (f==NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outfile);
		exit(1);
	}

	printf("Write out the data\n");
	for (c=0; c<C; c++) {
		for (q=0; q<arg_nquant; q++) {
			fprintf(f, "%.16f\t", quant[c][q]);
		}
		fprintf(f, "\n");
	}

	fclose(f);

	printf("----------------------\n");
	printf(" Write out the zero/nonzero files\n");
	printf("----------------------\n");
	f = fopen(cat(arg_outfile,".nonzero.csv"),"w");
	if (f==NULL) {
		printf("ERROR: cannot open %s for writing\n", path);
		exit(1);
	}

	// Write the header
	fprintf(f, "feature\tnum_nonzero\tnum_zero\tfrac_nonzero\tfrac_zero\t\n");

	// Write the zero/nonzero statistics
	for (c=0; c<C; c++) {
		int64 n = n_nonzero[c] + n_zero[c];
		float frac_nonzero = (float)n_nonzero[c] / (float)n;
		float frac_zero    = (float)n_zero[c] / (float)n;
		fprintf(f, "%d\t%lld\t%lld\t%f\t%f\t\n", c, n_nonzero[c], n_zero[c], frac_nonzero, frac_zero);
	}

	fclose(f);


	printf("Success\n");

	return 0;
}
