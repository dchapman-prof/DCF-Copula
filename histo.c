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
	int n,c,y,x,b;

	//----------------------
	// Read Command Arugments
	//----------------------
	if (argc<6) {
		printf("Usage:\n");
		printf("   ./histo infile outfile minx maxx nbin\n");
		printf("\n");
		printf("   infile:      in .bin feature file\n");
		printf("   outfile:     out .csv file\n");
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

	printf("arg_infile: %s\n",  arg_infile);
	printf("arg_outfile: %s\n", arg_outfile);
	printf("arg_minx %f\n", arg_minx);
	printf("arg_maxx %f\n", arg_maxx);
	printf("arg_nbin %d\n", arg_nbin);

	//----------------------
	// Memory map the data histogram file
	//----------------------
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
	int64 *n_zero    = (int64*)calloc(C, sizeof(int64));
	int64 *n_nonzero = (int64*)calloc(C, sizeof(int64));
	int64 **histo = (int64**)calloc(C, sizeof(int64*));
	for (c=0; c<C; c++)
		histo[c] = (int64*)calloc(arg_nbin, sizeof(int64));

	printf("----------------------\n");
	printf(" Compute the histograms\n");
	printf("----------------------\n");

	float scaling = (float)arg_nbin / (arg_maxx-arg_minx);

	// For every data element
	int64 idx=0;
	for (n=0; n<N; n++) {
	for (c=0; c<C; c++) {
	for (y=0; y<sY; y++) {
	for (x=0; x<sX; x++) {

		// Which bin (floating point) ?
		float val = X[idx];
		float fbin = (val-arg_minx) * scaling;
		if (fbin<0) {
			printf("WARNING fbin out of bounds val %f fbin %f\n", val, fbin);
			idx++;
			continue;
		}
		else if (fbin>=arg_nbin) {
			printf("WARNING fbin out of bounds val %f fbin %f\n", val, fbin);
			idx++;
			continue;
		}

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
			n_zero[c]++;	// otherwise it is a zero value

		// Next datafile element
		idx++;
	}}}}


	printf("----------------------\n");
	printf(" Unmap the data histogram\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);

	printf("----------------------\n");
	printf(" Write out the histogram\n");
	printf("----------------------\n");

	FILE *f = fopen(arg_outfile, "w");
	if (f==NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outfile);
		exit(1);
	}

	printf("Write out the ticks\n");
	float step = (arg_maxx-arg_minx)/(float)arg_nbin;
	float tick = arg_minx + 0.5*step;
	for (b=0; b<arg_nbin; b++) {
		fprintf(f, "%f\t", tick);
		tick += step;
	}
	fprintf(f, "\n");

	printf("Write out the data\n");
	for (c=0; c<C; c++) {
		for (b=0; b<arg_nbin; b++) {
			fprintf(f, "%lld\t", histo[c][b]);
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
