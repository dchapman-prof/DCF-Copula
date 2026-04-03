#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "../mmap.h"
//#include "scale_legendre.h"


#define MT_N 624
#define MT_M 397
#define MATRIX_A 0x9908b0df
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7fffffff

#define PI 3.14159265358979323846264
/*
uint32_t mt[MT_N];
int mti = MT_N + 1;

void init_genrand(uint32_t s) {
    printf("BEGIN init_genrand\n");
    mt[0] = s & 0xffffffff;
    for (int i = 1; i < MT_N; i++) {
        mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
    }
    printf("END init_genrand\n");
}

uint32_t genrand_int32(void) {
    if (mti >= MT_N) {

        int i;
        for (i = 0; i < MT_M; i++) {
            uint32_t y = (mt[i] & UPPER_MASK) | (mt[i+1] & LOWER_MASK);
            mt[i] = mt[i+MT_M] ^ (y >> 1) ^ (y & 1 ? MATRIX_A : 0);
        }
        for (; i < MT_N; i++) {
            uint32_t y = (mt[i] & UPPER_MASK) | (mt[i+1] & LOWER_MASK);
            mt[i] = mt[i-MT_M] ^ (y >> 1) ^ (y & 1 ? MATRIX_A : 0);
        }
        mt[MT_N-1] = mt[0] ^ (mt[MT_N-1] >> 1) ^ (mt[MT_N-1] & 1 ? MATRIX_A : 0);
        mti = 0;
    }
    uint32_t y = mt[mti];
    y ^= y >> 11;
    y ^= y << 7 & 0x9d2c5680;
    y ^= y << 15 & 0xefc60000;
    y ^= y >> 18;
    mti++;
    return y;
}
*/

#define int64 long long int

//#define RANDF  (double)((int)genrand_int32() / (float)((0x80000000)))

//#define RANDMAX 200000
//#define RANDI   (genrand_int32() % RANDMAX)
//#define RANDF   2.0*(RANDI/(double)RANDMAX) - 1.0

extern double P0[], P1[], P2[], P3[], P4[], P5[], P6[], P7[], P8[], P9[], P10[];

#define RANDF   ( rand() / (float)RAND_MAX )

// Easy string formatting
char path[4096];
char *cat(const char*str1, const char*str2) {
	(sprintf)(path, "%s%s", str1, str2);
	return path;
}

//--------------------------
// Command Arguments
//--------------------------
char *arg_infile;
char *arg_infile_test;
char *arg_outcsv;
char *arg_outgcf;
char *arg_outfgcf;
char *arg_outmom;
char *arg_outfmom;
char *arg_outloss;
int arg_featA;
int arg_featB;
int arg_featA_test;
int arg_featB_test;
float arg_min;
float arg_max;
int   arg_nbin;

//---------
// datafiles file (Train)
//---------
Binfile datafile;
float *data;
int64 data_N;
int64 data_C;
int64 data_sY;
int64 data_sX;
int64 data_CsYsX;
int64 data_sYsX;

//---------
// datafiles file (Test)
//---------
Binfile testfile;
float *data_test;
int64 data_N_test;
int64 data_C_test;
int64 data_sY_test;
int64 data_sX_test;
int64 data_CsYsX_test;
int64 data_sYsX_test;

//--------------------------
// Feature data (Train)
//--------------------------
float *A, *B;     // Original A and B
float *UA, *UB;   // Uniform A and B for Copula Analysis
float *A_sorted, *B_sorted;  // Sorted A and B for probability integral transform
float *feat_backup;   // for mergesort
int64 N_train;


//--------------------------
// Feature data (Test)
//--------------------------
float *A_test, *B_test;     // Test Data
float *UA_test, *UB_test;   // Uniform A_test and B_test for Copula Analysis
int64 N_test;


void Mergesort(float *X, float *back, int N) {

	int m = N/2;
	if (m==0)
		return;
		
	Mergesort(X,back,m);
	Mergesort(X+m,back,N-m);
	
	
	// Merge the values
	float *A = X;
	int  iA = 0;
	int  nA = m;
	float *B = X+m;
	int  iB = 0;
	int  nB = N-m;
	float *C = back;
	int  iC = 0;
	
	while (iA<nA && iB<nB) {
	
		if (A[iA] <= B[iB])
			C[iC++] = A[iA++];
		else
			C[iC++] = B[iB++]; 
	}
	
	while (iA<nA)
		C[iC++] = A[iA++];
	
	while (iB<nB)
		C[iC++] = B[iB++];
		
	// Copy back to array
	for (iC=0; iC<N; iC++)
		X[iC] = C[iC];
}

// Returns the index of val
int Binsearch(float val, float *X, int N)
{
	int a = 0;
	int b = N;
	while (b-a>1) {
		int m = (a+b)/2;
		float guess = X[m];
		if (val<guess)
			b=m;
		else
			a=m;
	}
	return a;
}

void normalize_histogram(int *histogram, double *normalized_histogram, int num_bins, double *norm) {
	
	float deltax = 2.0 / num_bins;
	float deltay = 2.0 / num_bins;
	
	printf("deltax: %f, deltay: %f\n", deltax, deltay);

	// Compute total sum of the histogram
	long total_sum = 0;
	
	for (int i = 0; i < num_bins * num_bins; i++) {
		total_sum += histogram[i];
		
	}
	printf("Total sum of histogram: %ld\n", total_sum);

	// Calculate normalization factor
	* norm = total_sum * deltax * deltay;
	printf("Normalization factor (norm): %f\n", *norm);

	// Normalize the histogram
	if (norm > 0) { 
		for (int i = 0; i < num_bins * num_bins; i++) {
			normalized_histogram[i] = (int)histogram[i] / (*norm);
			//printf("normalized_histogram[%d]: %f\n", i, normalized_histogram[i]);
		}
	}
}


int convert_bin_idx(float value, int num_bins) {
	// Shift from [-1, 1] to [0, 2], scale to [0, num_bins], and floor to get integer index
	int index = (int)floor(((value + 1.0) / 2.0) * num_bins);

	// Ensure the index is within [0, num_bins-1]
	if (index < 0) {
		index = 0;
	} else if (index >= num_bins) {
		index = num_bins - 1;
	}

	return index;
}


void check_empty_bins(int *histogram, int num_bins) {
	int empty_bins = 0;

	for (int i = 0; i < num_bins * num_bins; i++) {
		if (histogram[i] == 0) {
    			//printf("Warning: Bin %d is empty.\n", i);
    			empty_bins++;
		}
	}

	if (empty_bins == num_bins) {
		printf("All bins are Empty.\n");
		//fgetc(stdin);
	} else {
		//printf("Total empty bins: %d\n", empty_bins);
		printf("Total empty bins: %d\n", empty_bins);

	}

}

int main(int argc, char**argv)
{
	int i,n,y,x,mA,mB;
/*
	init_genrand(time(NULL));

	for (i=0; i<200; i++) {
		float val = RANDF;
		printf("val %f\n", val);
	}
	exit(0);
*/
/*
	float vals[10] = {3,2,5,1,4,2,7,5,0,1};
	float back[10];
	
	for (i=0; i<10; i++)
		printf("%f ", vals[i]);
	printf("\n");
	Mergesort(vals, back, 10);
	for (i=0; i<10; i++)
		printf("%f ", vals[i]);
	printf("\n");
	

	exit(1);
*/


	//-----
	// Read command arguments
	//-----
	if (argc<16) {
		printf("Usage\n");
		printf("  ./pit infile outcsv featA featB\n");
		printf("\n");
		printf(" infile    input binary file .bin\n");
		printf(" infile_test    input binary test file .bin\n");
		//printf(" outcsv    output csv for chebyshev moments\n");
		printf(" outcsv    output csv for histogram\n");
		printf(" outgcf    output gcf for Legendre pdf\n");
		printf(" outfgcf   output fgcf for Fourier pdf\n");
		printf(" outmom    output for Legendre moments\n");
		printf(" outfmom   output for Fourier moments\n");
		printf(" outloss   output for hist_loss and Legendre_loss and Fourier_loss\n");
		printf(" featA     first feature to compare\n");
		printf(" featB     second feature to compare\n");
		printf("   min,max:   x-interval for histogram binning\n");
		printf("   nbin:        how many histogram bins\n");
		printf("\n");
		exit(1);
	}
	arg_infile      =      argv[1];
	arg_infile_test =      argv[2];
	arg_outcsv      =      argv[3];
	arg_outgcf      =      argv[4];
	arg_outfgcf      =     argv[5];
	arg_outmom      =      argv[6];
	arg_outfmom      =     argv[7];
	arg_outloss     =      argv[8];
	arg_featA       = atoi(argv[9]);
	arg_featB       = atoi(argv[10]);
	arg_featA_test  = atoi(argv[11]);
	arg_featB_test  = atoi(argv[12]);
	arg_min         = atof(argv[13]);
	arg_max         = atof(argv[14]);
	arg_nbin        = atoi(argv[15]);
	
	printf("arg_infile %s\n", arg_infile);
	printf("arg_infile %s\n", arg_infile_test);
	printf("arg_outcsv %s\n", arg_outcsv);
	printf("arg_outgcf %s\n", arg_outgcf);
	printf("arg_outfgcf %s\n", arg_outfgcf);
	printf("arg_outmom %s\n", arg_outmom);
	printf("arg_outfmom %s\n", arg_outfmom);
	printf("arg_outloss %s\n", arg_outloss);
	printf("arg_featA %d\n", arg_featA);
	printf("arg_featB %d\n", arg_featB);
	printf("arg_featA_test %d\n", arg_featA_test);
	printf("arg_featB_test %d\n", arg_featB_test);
	printf("arg_min %f\n", arg_min);
	printf("arg_max %f\n", arg_max);
	printf("arg_nbin %d\n", arg_nbin);
	

	printf("-------------------------\n");
	printf(" Memory map the data file\n");
	printf("-------------------------\n");
	
	datafile = MapBinfileR(arg_infile);
	printf("Opening train file: %s\n", arg_infile);
	data = (float*)datafile.data;
	data_N = datafile.shape[0];
	data_C = datafile.shape[1];
	data_sY = datafile.shape[2];
	data_sX = datafile.shape[3];
	data_CsYsX = data_C*data_sY*data_sX;
	data_sYsX = data_sY*data_sX;
	
	printf("data_N %lld data_C %lld data_sY %lld data_sX %lld data_CsYsX %lld data_sYsX %lld\n", data_N, data_C, data_sY, data_sX, data_CsYsX, data_sYsX);
	
	printf("------------------------------\n");
	printf(" Memory map the test data file\n");
	printf("------------------------------\n");

	testfile = MapBinfileR(arg_infile_test);
	arg_infile_test = argv[2];
	printf("Opening test file: %s\n", arg_infile_test);
	
	data_test = (float*)testfile.data;
	data_N_test = testfile.shape[0];
	data_C_test = testfile.shape[1];
	data_sY_test = testfile.shape[2];
	data_sX_test = testfile.shape[3];
	data_CsYsX_test = data_C_test * data_sY_test * data_sX_test;
	data_sYsX_test = data_sY_test * data_sX_test;

	printf("data_N_test %lld data_C_test %lld data_sY_test %lld data_sX_test %lld data_CsYsX_test %lld data_sYsX_test %lld\n", 
		data_N_test, data_C_test, data_sY_test, data_sX_test, data_CsYsX_test, data_sYsX_test);


	
	
	printf("---------------------------------\n");
	printf(" Read the non-zero train features\n");
	printf("---------------------------------\n");
	int cA = arg_featA;
	int cB = arg_featB;
	
	printf("How many non-zero train features ?\n");
	N_train=0;
	for (n=0; n<data_N; n++) {
	for (y=0; y<data_sY; y++) {
	for (x=0; x<data_sX; x++) {
		int64 idxA = n*data_CsYsX + cA*data_sYsX + y*data_sX + x;
		int64 idxB = n*data_CsYsX + cB*data_sYsX + y*data_sX + x;
		
		if (data[idxA]!=0 && data[idxB]!=0 && !isnan(data[idxA]) && !isnan(data[idxB]) && !isinf(data[idxA]) && !isinf(data[idxB]))
			N_train++;
	}}}

	
	printf("num nonzero train features %lld\n", N_train);
	
	printf("Allocate nonzero train features\n");
	A = (float*)malloc(N_train*sizeof(float));
	B = (float*)malloc(N_train*sizeof(float));
	
	printf("Copy nonzero train features\n");
	N_train=0;
	for (n=0; n<data_N; n++) {
	for (y=0; y<data_sY; y++) {
	for (x=0; x<data_sX; x++) {
		int64 idxA = n*data_CsYsX + cA*data_sYsX + y*data_sX + x;
		int64 idxB = n*data_CsYsX + cB*data_sYsX + y*data_sX + x;
		
		if (data[idxA]!=0 && data[idxB]!=0 && !isnan(data[idxA]) && !isnan(data[idxB]) && !isinf(data[idxA]) && !isinf(data[idxB])) {
			A[N_train] = data[idxA];
			B[N_train] = data[idxB];
			//A[N] = rand() + 0.1*rand();
			//A[N] = RANDF;//idxA;
			//B[N] = RANDF;//rand() + 0.1*rand();
			N_train++;
		}
	}}}
/*
	N=0;
	for (y=0; y<1000; y++) {
	for (x=0; x<1000; x++) {
		A[N] = RANDF;
		B[N] = RANDF;
		printf("A %f B %f\n", A[N], B[N]);
		N++;
	}}
*/	
		
	printf("--------------------------------------------------------------------\n");
	printf(" Sort the non-zero features for Copula probability integral transform\n");
	printf("---------------------------------------------------------------------\n");
	
	printf("Allocate sorted arrays\n");
	A_sorted = (float*)malloc(N_train*sizeof(float));
	B_sorted = (float*)malloc(N_train*sizeof(float));
	feat_backup = (float*)malloc(N_train*sizeof(float));

	for (n=0; n<N_train; n++) {
		A_sorted[n] = A[n];
		B_sorted[n] = B[n];
	}

	printf("Sort the data\n");
	Mergesort(A_sorted, feat_backup, N_train);
	Mergesort(B_sorted, feat_backup, N_train);

	free(feat_backup);
	
	printf("Convert to uniform distribution [-1, 1]\n");
	UA = (float*)malloc(N_train*sizeof(float));
	UB = (float*)malloc(N_train*sizeof(float));
	float inv_N_train = 1.0/N_train;
	for (n=0; n<N_train; n++) {
		float valA = A[n];
		float valB = B[n];
		int idxA = Binsearch(valA, A_sorted, N_train);
		int idxB = Binsearch(valB, B_sorted, N_train);
		
		//printf("Accessing data[idxA=%d], data[idxB=%d]\n", idxA, idxB);
		UA[n] = 2.0*(idxA + 0.5)*inv_N_train - 1.0;     // uniform [-1, 1]
		UB[n] = 2.0*(idxB + 0.5)*inv_N_train - 1.0;     // uniform [-1, 1]
		//UA[n] = RANDF;
		//UB[n] = RANDF;
		
		
		//printf("A %.4f UA %.4f    B %.4f  UB %.4f\n", A[n], UA[n], B[n], UB[n]);
	}
	
	
	
	printf("--------------------------------\n");
	printf(" Read the non-zero test features\n");
	printf("--------------------------------\n");
	int cA_test = arg_featA_test;
	int cB_test = arg_featB_test;
	
	printf("How many non-zero test features ?\n");
	N_test=0;
	for (n=0; n<data_N_test; n++) {
	for (y=0; y<data_sY_test; y++) {
	for (x=0; x<data_sX_test; x++) {
		int64 idxA_test = n*data_CsYsX_test + cA_test*data_sYsX_test + y*data_sX_test + x;
		int64 idxB_test = n*data_CsYsX_test + cB_test*data_sYsX_test + y*data_sX_test + x;
		
		if (data[idxA_test]!=0 && data[idxB_test]!=0 && !isnan(data[idxA_test]) && !isnan(data[idxB_test]) && !isinf(data[idxA_test]) && !isinf(data[idxB_test]))
			N_test++;
	}}}

	
	printf("num nonzero test features %lld\n", N_test);
	
	printf("Allocate nonzero test features\n");
	A_test  = (float*)malloc(N_test*sizeof(float));  
	B_test  = (float*)malloc(N_test*sizeof(float));
	
	printf("Copy nonzero test features\n");
	N_test=0;
	for (n=0; n<data_N_test; n++) {
	for (y=0; y<data_sY_test; y++) {
	for (x=0; x<data_sX_test; x++) {
		int64 idxA_test = n*data_CsYsX_test + cA_test*data_sYsX_test + y*data_sX_test + x;
		int64 idxB_test = n*data_CsYsX_test + cB_test*data_sYsX_test + y*data_sX_test + x;
		
		if (data[idxA_test]!=0 && data[idxB_test]!=0 && !isnan(data[idxA_test]) && !isnan(data[idxB_test]) && !isinf(data[idxA_test]) && !isinf(data[idxB_test])) {
			A_test[N_test] = data[idxA_test];
			B_test[N_test] = data[idxB_test];
			N_test++;
		}
	}}}
	
	
	printf("-------------------------------------------------\n");
	printf("Project test_data to uniform distribution [-1, 1]\n");
	printf("-------------------------------------------------\n");
	
	UA_test = (float*)malloc(N_test*sizeof(float));
	UB_test = (float*)malloc(N_test*sizeof(float));
	
	if (A_test == NULL || B_test == NULL) {
		fprintf(stderr, "Memory allocation failed\n");
	return -1;  
	}

	float inv_N_test = 1.0/N_test;
	
	for (int n = 0; n < N_test; n++) {
		//printf("N_test: %d\n", N_test);

		float valA_test = A_test[n];  
		float valB_test = B_test[n]; 
	    

		// binary search to find indices of the test data
		int idxA_test = Binsearch(valA_test, A_sorted, N_train);
		int idxB_test = Binsearch(valB_test, B_sorted, N_train);

		
		if (idxA_test < 0 || idxA_test >= N_train || idxB_test < 0 || idxB_test >= N_train) {
			fprintf(stderr, "Invalid index from binary search: idxA_test=%d, idxB_test=%d\n", idxA_test, idxB_test);
			continue; 
		}

		// Map the indices to the uniform distribution [-1, 1]
		UA_test[n] = 2.0 * (idxA_test + 0.5) * inv_N_train - 1.0;  // uniform [-1, 1]
		UB_test[n] = 2.0 * (idxB_test + 0.5) * inv_N_train - 1.0;  // uniform [-1, 1]
    
	}


	printf("Projecting test data is completed.\n");

	
	printf("----------------------\n");
	printf(" Calculate covariance\n");
	printf("----------------------\n");
	double covAB = 0.0;
	for (n=0; n<N_train; n++) {
		covAB += (double)UA[n]*(double)UB[n];
	}
	covAB = covAB*inv_N_train;
	
	float corrAB = covAB * 3;
	
	printf("Covariance %f Correlation %f\n", covAB, corrAB);
	
	
	printf("---------------------------\n");
	printf(" Calculate Legendre Moments\n");
	printf("---------------------------\n");
	#define nMoments 11
	double  TA[64];
	double  TB[64];
	double   T[64][64];
	double mom[64][64];
	
	
	//char *arg_out;
	//arg_out = "moments.csv";
	//FILE *csvfile = fopen(arg_outcsv, "w");
	FILE *csvfile_mom = fopen(arg_outmom, "w");
	
	if (csvfile_mom == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outmom);
		
		exit(1);
	}
	fprintf(csvfile_mom, "mom[FeatureA][FeatureB]");  
	for (int x = 0; x < nMoments; x++) {
		fprintf(csvfile_mom, ",%d", x); 
	}
	fprintf(csvfile_mom, "\n");



	// Moment lengths array
	double moment_lengths[11] = {2.0, 0.6666666666666666, 0.4, 0.2857142857142858, 
                            0.2222222222222223, 0.1818181818181799, 
                            0.1538461538461604, 0.1333333333333635, 
                             0.1176470588234224, 0.1052631578950241, 
                             0.1081811814074172};

	// Scaling factors for normalized legendre polynomials
	double legendre_scaling[11];
	for (i=0; i<11; i++)
		legendre_scaling[i] = 1.0 / sqrt(moment_lengths[i]);


	for (mA=0; mA<nMoments; mA++)
		for (mB=0; mB<nMoments; mB++)
			mom[mA][mB] = 0.0;
	
	for (n=0; n<N_train; n++)
	{
		// Calculate Legendre of A
		double xA = UA[n];
	
		// Calculate Legendre of B
		double xB = UB[n];

		// Calculate Ledgendre polynomial function values
		TA[0] = 1;
		TA[1] = xA;
		for (mA=2; mA<nMoments; mA++)
			TA[mA] = ((2*mA-1.0)/mA)*xA*TA[mA-1] - ((mA-1.0)/mA)*TA[mA-2];
			
		TB[0] = 1;
		TB[1] = xB;
		for (mB=2; mB<nMoments; mB++)
			TB[mB] = ((2*mB-1.0)/mB)*xB*TB[mB-1] - ((mB-1.0)/mB)*TB[mB-2];

		// Rescale polynomials to unit length
		for (mA=0; mA<nMoments; mA++){
			TA[mA] *= legendre_scaling[mA];
		}
		for (mB=0; mB<nMoments; mB++){
			TB[mB] *= legendre_scaling[mB];		
		}
		// Calculate the Legendre of A,B and compute MoM
		for (mA=0; mA<nMoments; mA++) {
			for (mB=0; mB<nMoments; mB++) {
				T[mA][mB]    = TA[mA]*TB[mB];   // Calculate Legendre
				mom[mA][mB] += T[mA][mB];	// Add to MoM
			}
		}
	}
	
	
	// Compute expected value
	for (mA=0; mA<nMoments; mA++) {
		fprintf(csvfile_mom, "%d", mA);
		
		for (mB=0; mB<nMoments; mB++) {
			mom[mA][mB] *= inv_N_train;
			fprintf(csvfile_mom, ",%f", mom[mA][mB]);

			printf("MoM_A_%d_B_%d_,%f\n", mA, mB, mom[mA][mB]);
		}
		fprintf(csvfile_mom, "\n"); 
	}
	printf("MOMENTS are saved as 2D matrix to %s\n", arg_outmom);
	fclose(csvfile_mom);	
	
	
	printf("Covariance %f Correlation %f\n", covAB, corrAB);
	
	
	
	printf("--------------------------\n");
	printf(" Calculate Fourier Moments\n");
	printf("---------------------------\n");
	#define nMoments 11
	double  FTA[64];
	double  FTB[64];
	double   FT[64][64];
	double fmom[64][64];
	

	FILE *csvfile_fmom = fopen(arg_outfmom, "w");
	
	if (csvfile_fmom == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outfmom);
		
		exit(1);
	}
	fprintf(csvfile_fmom, "fouriermom[FeatureA][FeatureB]");  
	for (int x = 0; x < nMoments; x++) {
		fprintf(csvfile_fmom, ",%d", x); 
	}
	fprintf(csvfile_fmom, "\n");


	for (mA=0; mA<nMoments; mA++)
		for (mB=0; mB<nMoments; mB++)
			fmom[mA][mB] = 0.0;
	
	for (n=0; n<N_train; n++)
	{
		// Calculate Fourier of A
		double xA = UA[n];
	
		// Calculate Fourier of B
		double xB = UB[n];


		// Calculate Fourier function values
		FTA[0] = 0.70710678118;                       // sqrt(2)/2
		for (mA=1; mA<nMoments; mA++)
			FTA[mA] = cos( 0.5*mA*PI * (xA-1.0) );
		
		FTB[0] = 0.70710678118;
		for (mB=1; mB<nMoments; mB++)
			FTB[mB] = cos( 0.5*mB*PI * (xB-1.0) );
		
		// Calculate the Fourier of A,B and compute MoM
		for (mA=0; mA<nMoments; mA++) {
			for (mB=0; mB<nMoments; mB++) {
				FT[mA][mB]    = FTA[mA]*FTB[mB];   // Calculate Fourier
				fmom[mA][mB] += FT[mA][mB];	   // Add to MoM
			}
		}
	}
	
	
	// Compute expected value
	for (mA=0; mA<nMoments; mA++) {
		fprintf(csvfile_fmom, "%d", mA);
		
		for (mB=0; mB<nMoments; mB++) {
			fmom[mA][mB] *= inv_N_train;
			fprintf(csvfile_fmom, ",%f", fmom[mA][mB]);

			printf("Fourier MoM_A_%d_B_%d_,%f\n", mA, mB, fmom[mA][mB]);
		}
		fprintf(csvfile_fmom, "\n"); 
	}
	printf("Fourier MOMENTS are saved as 2D matrix to %s\n", arg_outfmom);
	fclose(csvfile_fmom);	
	
	printf("---------------------------------------------\n");
	printf(" Calculate Histogram of data  Range is [-1 1]\n");
	printf("---------------------------------------------\n");
	
	int num_bins = arg_nbin;
	float bin_width = (arg_max-arg_min)/num_bins;
	double norm = 0;

	
	int *histogram = (int*)calloc(num_bins*num_bins, sizeof(int));
	if (histogram == NULL) {
		 fprintf(stderr, "Memory allocation failed\n");
	}
	
	double *normalized_histogram = (double*)calloc(num_bins * num_bins, sizeof(double));
	if (normalized_histogram == NULL) {
    		fprintf(stderr, "Memory allocation failed for normalized histogram\n");
	}


	for (i=0; i<N_train; i++) {

		UA[i] = ((UA[i]+1)/2)*num_bins;
			
		UB[i] = ((UB[i]+1)/2)*num_bins;
				
        	int binA = (int)(UA[i]);

		int binB = (int)(UB[i]);
		
		if (binA < 0) 
			binA = 0;
		if (binA >= num_bins) 
			binA = num_bins - 1;
		if (binB < 0) 
			binB = 0;
		if (binB >= num_bins) 
			binB = num_bins - 1;

		histogram[binA * num_bins + binB]++;
		
		}
	normalize_histogram(histogram, normalized_histogram, num_bins, &norm);
	
	
	printf("----------------------------\n");
	printf(" Evaluation of the Histogram\n");
	printf("----------------------------\n");
 	
 	double cross_entropy_hist = 0.0;
 	double log_pdf_hist       = 0.0;
 	//double total_pdf_hist       = 0.0;  
 	
	for (int n=0; n<N_test; n++) {

		double UA_test_val = UA_test[n];  
		double UB_test_val = UB_test[n];  
			
		//printf("UA_test[%d] = %f ", n, UA_test_val);
		//printf("UB_test[%d] = %f ", n, UB_test_val);
		
		// Convert UA_test_val and UB_test_val to indices in the range [0, num_bins-1]
		int UA_index = (int)(((UA_test_val + 1) / 2) * (num_bins));
		int UB_index = (int)(((UB_test_val + 1) / 2) * (num_bins));
		
		// Ensure indices are within bounds
		if (UA_index < 0) UA_index = 0;
		if (UA_index >= num_bins) UA_index = num_bins - 1;
		if (UB_index < 0) UB_index = 0;
		if (UB_index >= num_bins) UB_index = num_bins - 1;

		//printf("UA_test[%d] = %f, Index = %d ", n, UA_test_val, UA_index);
		//printf("UB_test[%d] = %f, Index = %d\n", n, UB_test_val, UB_index);
		
		//printf("------------------------\n");
		//printf(" \n");
		//printf("------------------------\n");
		
		int hist_index = UA_index * num_bins + UB_index;      // Convert to 1D index
		//printf("hist_index = %i, \n", hist_index);
		
		double pdf_hist = normalized_histogram[hist_index];   // Look up the PDF value
		printf("pdf_hist = %f, \n", pdf_hist);
		check_empty_bins(histogram, arg_nbin);
		//fgetc(stdin);
		
		//total_pdf_hist += pdf_hist;
		if(pdf_hist == 0){
			printf("Warning: Empty bin at index %d\n", hist_index);
			pdf_hist = 0.0001;
			
		}
		log_pdf_hist += log(pdf_hist);
		
		//if (log_pdf_hist != 0) {
			
			//double log_pdf_hist = log(total_pdf_hist);
			cross_entropy_hist =  log_pdf_hist * (-1.0)/N_test;
			//printf("n %d, Cross-Entropy of histogram Estimation: %f\n", n, cross_entropy_hist);
		//}
		//else {
		//	printf("Warning: PDF is zero or negative \n", x, y);
		//}
	
	}

	
	printf("------------------------\n");
	printf(" write out the Histogram\n");
	printf("------------------------\n");
		
	// Write the histogram as a 2D matrix

	FILE *csvfile = fopen(arg_outcsv, "w");

	if (csvfile == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outcsv);
		exit(1);
	}


	fprintf(csvfile, "BinA/BinB");  
	for (int x = 0; x < num_bins; x++) {
		fprintf(csvfile, ",%d", x); 
	}
	fprintf(csvfile, "\n");


	for (int binA = 0; binA < num_bins; binA++) {
		fprintf(csvfile, "%d", binA);  
		for (int binB = 0; binB < num_bins; binB++) {
	      		
			fprintf(csvfile, ",%f", normalized_histogram[binA * num_bins + binB]);
	

		}
		fprintf(csvfile, "\n");  
	}

	printf("Histogram saved as 2D matrix to %s\n", arg_outcsv);
	fclose(csvfile);
	
	

	printf("--------------------------------------\n");
	printf(" Calculate Legendre Characteristic PDF\n");
	printf("--------------------------------------\n");
	
	
	FILE *gcffile = fopen(arg_outgcf, "w");
	
	if (gcffile==NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outgcf);
		exit(1);
	}

	fprintf(gcffile, "BinA/BinB");
	for (x=0; x<num_bins; x++)
		fprintf(gcffile, ",%d", x);
	fprintf(gcffile, "\n");

	double inv_num_bins = 1.0/num_bins;
	
	for (y=0; y<num_bins; y++) {
		fprintf(gcffile, "%d", y);
		
		for (x=0; x<num_bins; x++) {
			double xA = 2.0 * (y+0.5) * inv_num_bins - 1.0;
			double xB = 2.0 * (x+0.5) * inv_num_bins - 1.0;
		
			// Calculate Ledgendre polynomial function values
			TA[0] = 1;
			TA[1] = xA;
			for (mA=2; mA<nMoments; mA++)
				TA[mA] = ((2*mA-1.0)/mA)*xA*TA[mA-1] - ((mA-1.0)/mA)*TA[mA-2];
			TB[0] = 1;
			TB[1] = xB;
			for (mB=2; mB<nMoments; mB++)
				TB[mB] = ((2*mB-1.0)/mB)*xB*TB[mB-1] - ((mB-1.0)/mB)*TB[mB-2];
			
			// Rescale polynomials to unit length
			for (mA=0; mA<nMoments; mA++)
				TA[mA] *= legendre_scaling[mA];
			for (mB=0; mB<nMoments; mB++)
				TB[mB] *= legendre_scaling[mB];		

			// Calculate pdf at that test point
			double pdf = 0.0;
			for (mA=0; mA<nMoments; mA++){
				for (mB=0; mB<nMoments; mB++){
					double cont = mom[mA][mB]*TA[mA]*TB[mB];
					pdf += cont;
				}
			}
			fprintf(gcffile, ",%f", pdf);
		}
		fprintf(gcffile, "\n");
	}
	
	fclose(gcffile);



	printf("-------------------------------------\n");
	printf(" Calculate Fourier Characteristic PDF \n");
	printf("--------------------------------------\n");
	
	//arg_outcsv = "gcf.csv";
	FILE *fgcffile = fopen(arg_outfgcf, "w");
	
	if (gcffile==NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outfgcf);
		exit(1);
	}

	fprintf(fgcffile, "BinA/BinB");
	for (x=0; x<num_bins; x++)
		fprintf(fgcffile, ",%d", x);
	fprintf(fgcffile, "\n");

	//double inv_num_bins = 1.0/num_bins;
	
	for (y=0; y<num_bins; y++) {
		fprintf(fgcffile, "%d", y);
		
		for (x=0; x<num_bins; x++) {
			double xA = 2.0 * (y+0.5) * inv_num_bins - 1.0;
			double xB = 2.0 * (x+0.5) * inv_num_bins - 1.0;
		
			// Calculate Fourier function values
			FTA[0] = 0.70710678118;                       // sqrt(2)/2
			for (mA=1; mA<nMoments; mA++)
				FTA[mA] = cos( 0.5*mA*PI * (xA-1.0) );
			
			FTB[0] = 0.70710678118;
			for (mB=1; mB<nMoments; mB++)
				FTB[mB] = cos( 0.5*mB*PI * (xB-1.0) );
			
			
			// Calculate pdf at that test point
			double pdf = 0.0;
			for (mA=0; mA<nMoments; mA++){
				for (mB=0; mB<nMoments; mB++){
					double cont = fmom[mA][mB]*FTA[mA]*FTB[mB];
					pdf += cont;
				}
			}
			fprintf(fgcffile, ",%f", pdf);
		}
		fprintf(fgcffile, "\n");
	}
	
	fclose(fgcffile);



	printf("-------------------------------\n");
	printf(" Evaluation of the Legendre PDF\n");
	printf("--------------------------------\n");


	double cross_entropy_legendre= 0.0;

	double log_pdf_legendre = 0.0;
	  
	  
	for (int n = 0; n < N_test; n++) {
		
		// Use indexed values of UA_test and UB_test arrays
		double UA_test_val = UA_test[n];  
		double UB_test_val = UB_test[n];  
		
		//printf("UA_test[%d] = %f\n", n, UA_test_val);
		//printf("UB_test[%d] = %f\n", n, UB_test_val);
		
		double xA = UA_test_val;
		double xB = UB_test_val;
		
		// Calculate Ledgendre polynomial function values
		TA[0] = 1;
		TA[1] = xA;
		for (mA=2; mA<nMoments; mA++)
			TA[mA] = ((2*mA-1.0)/mA)*xA*TA[mA-1] - ((mA-1.0)/mA)*TA[mA-2];
		TB[0] = 1;
		TB[1] = xB;
		for (mB=2; mB<nMoments; mB++)
			TB[mB] = ((2*mB-1.0)/mB)*xB*TB[mB-1] - ((mB-1.0)/mB)*TB[mB-2];

		// Rescale polynomials to unit length
		for (mA=0; mA<nMoments; mA++){
			TA[mA] *= legendre_scaling[mA];
		}
		for (mB=0; mB<nMoments; mB++){
			TB[mB] *= legendre_scaling[mB];		
		}
		// Calculate pdf at that point
		double pdf_legendre = 0.0;
		int num = 0;
		for (mA=0; mA<nMoments; mA++){
			for (mB=0; mB<nMoments; mB++){
			
				if (mB >= 0 && mB < nMoments) {
					// Safe to access TB[mB]
				} else {
					printf("Invalid index for mB: %d\n", mB);
					fgetc(stdin);
					// Handle error or continue to the next iteration
				}
			
				//pdf += mom[mA][mB]*TA[mA]*TB[mB];
				double cont = mom[mA][mB]*TA[mA]*TB[mB];
				//cont /= moment_lengths[mA];
				//cont /= moment_lengths[mB];
				pdf_legendre += cont;
								
			}
		}
		if (pdf_legendre<0.0001)
			pdf_legendre=0.0001;
		
		
		log_pdf_legendre += log(pdf_legendre);
		
		if (pdf_legendre<=0.0) {
			//num +=1;
			printf("WARNING n %d pdf_legendre %f\n", n, pdf_legendre);
			printf("UA_test_val %f\n", UA_test_val);
			printf("UB_test_val %f\n", UB_test_val);
			printf("xA %f\n", xA);
			printf("xB %f\n", xB);
			printf("N_test %lld\n", N_test);
			
			
			

			fgetc(stdin);
		}

		// Calculate cross-entropy using the log(pdf)
		cross_entropy_legendre =  log_pdf_legendre * (-1.0)/N_test;
	}	



	printf("------------------------------\n");
	printf(" Evaluation of the Fourier PDF\n");
	printf("------------------------------\n");


	double cross_entropy_fourier= 0.0;

	double log_pdf_fourier = 0.0;
	  
	  
	for (int n = 0; n < N_test; n++) {
		
		// Use indexed values of UA_test and UB_test arrays
		double UA_test_val = UA_test[n];  
		double UB_test_val = UB_test[n];  
		
		//printf("UA_test[%d] = %f\n", n, UA_test_val);
		//printf("UB_test[%d] = %f\n", n, UB_test_val);
		
		double xA = UA_test_val;
		double xB = UB_test_val;
		
		// Calculate Fourier function values
		FTA[0] = 0.70710678118;    // sqrt(2)/2
		for (mA=1; mA<nMoments; mA++)
			FTA[mA] = cos( 0.5*mA*PI * (xA-1.0) );
		FTB[0] = 0.70710678118;
		for (mB=1; mB<nMoments; mB++)
			FTB[mB] = cos( 0.5*mB*PI * (xB-1.0) );


		// Calculate pdf at that point
		double pdf_fourier = 0.0;
		int num = 0;
		for (mA=0; mA<nMoments; mA++){
			for (mB=0; mB<nMoments; mB++){
			
				if (mB >= 0 && mB < nMoments) {
					// Safe to access TB[mB]
				} else {
					printf("Invalid index for mB: %d\n", mB);
					fgetc(stdin);
					// Handle error or continue to the next iteration
				}
			
				//pdf += mom[mA][mB]*TA[mA]*TB[mB];
				double cont = fmom[mA][mB]*FTA[mA]*FTB[mB];
				pdf_fourier += cont;				
			}
		}
		if (pdf_fourier<0.0001)
			pdf_fourier=0.0001;
		
		
		log_pdf_fourier += log(pdf_fourier);
		
		if (pdf_fourier<=0.0) {
			//num +=1;
			printf("WARNING n %d pdf_fourier %f\n", n, pdf_fourier);
			printf("UA_test_val %f\n", UA_test_val);
			printf("UB_test_val %f\n", UB_test_val);
			printf("xA %f\n", xA);
			printf("xB %f\n", xB);
			printf("N_test %lld\n", N_test);

			
			

			fgetc(stdin);
		}

		// Calculate cross-entropy using the log(pdf)
		cross_entropy_fourier =  log_pdf_fourier * (-1.0)/N_test;
	}	


	printf("Cross-Entropy of histogram Estimation: %f\n", cross_entropy_hist);
	printf("Cross-Entropy of Legendre Polynomial Estimation: %f\n", cross_entropy_legendre);	
	printf("Cross-Entropy of Fourier Estimation: %f\n", cross_entropy_fourier);	
	

	FILE *csvfile_loss = fopen(arg_outloss, "a");
	//FILE *csv_file_loss = fopen("cross_entropy_results.csv", "a"); 

	if (csvfile_loss == NULL) {
		printf("Error opening file!\n");
	return 1;
	}

	// Write header only once if the file is empty
	fseek(csvfile_loss, 0, SEEK_END);
	long size = ftell(csvfile_loss);
	if (size == 0) {
		
		fprintf(csvfile_loss, "Feature A_test,Feature B_test,Cross-Entropy Histogram,Cross-Entropy Legendre,Cross-Entropy Fourier\n");
	}

	fprintf(csvfile_loss, "%d,%d,%f,%f,%f\n", arg_featA_test,arg_featB_test, cross_entropy_hist, cross_entropy_legendre, cross_entropy_fourier);
	
	printf("Losses are saved as 2D matrix to %s\n", arg_outloss);
	fclose(csvfile_loss);


	printf("----------------------\n");
	printf(" Unmap the dataset features\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);	



	
	return 0;
}
