#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include "../mmap.h"
#include "archimedian.h"
//#include "scale_legendre.h"


#define OVER_2PI 0.15915494309
#define PI 3.14159265358979323846264
#define RANDF  ( (double)rand() / (double)RAND_MAX )
#define int64 long long int
//#define nMoments 11
//#define MIN(a,b)  ( (a)<(b) ? (a) : (b) )
//#define MAX(a,b)  ( (a)<(b) ? (b) : (a) )
#define MINMAX(x,min,max)    MIN((max),MAX((x),(min)))

// Easy string formatting
char path[4096];
char *cat(const char*str1, const char*str2) {
	(sprintf)(path, "%s%s", str1, str2);
	return path;
}

// Complex numbers
typedef struct complex {
	double r,i;
} complex;

complex complex_mul(complex A, complex B) {
	complex res;
	res.r = A.r*B.r - A.i*B.i;
	res.i = A.r*B.i + A.i*B.r;
	return res;
}


//--------------------------
// Command Arguments
//--------------------------
char arg_infile[4096];
char arg_infile_test[4096];
char arg_outcsv[4096];
char arg_outgcf[4096];
char arg_outfgcf[4096];
char arg_outecf[4096];
char arg_outmom[4096];
char arg_outfmom[4096];
char arg_norm_ecf[4096];
char arg_outecfr[4096];
char arg_outecfi[4096];
//char arg_outloss[6144];

int arg_featA;
int arg_featB;
int arg_featA_test;
int arg_featB_test;
int arg_nbin;
float arg_min;
float arg_max;
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
double *A, *B;     // Original A and B
float *UA, *UB;   // Uniform A and B for Copula Analysis
double *A_sorted, *B_sorted;  // Sorted A and B for probability integral transform
double *feat_backup;   // for mergesort
int64 N_train;


//--------------------------
// Feature data (Test)
//--------------------------
double *A_test, *B_test;     // Test Data
float *UA_test, *UB_test;   // Uniform A_test and B_test for Copula Analysis
int64 N_test;






void Mergesort(double *X, double *back, int N) {

	int m = N/2;
	if (m==0)
		return;
		
	Mergesort(X,back,m);
	Mergesort(X+m,back,N-m);
	
	
	// Merge the values
	double *A = X;
	int  iA = 0;
	int  nA = m;
	double *B = X+m;
	int  iB = 0;
	int  nB = N-m;
	double *C = back;
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
int Binsearch(double val, double *X, int N)
{
	int a = 0;
	int b = N;
	while (b-a>1) {
		int m = (a+b)/2;
		double guess = X[m];
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
	
	//printf("deltax: %f, deltay: %f\n", deltax, deltay);

	// Compute total sum of the histogram
	long total_sum = 0;
	
	for (int i = 0; i < num_bins * num_bins; i++) {
		total_sum += histogram[i];
		
	}
	//printf("Total sum of histogram: %ld\n", total_sum);

	// Calculate normalization factor
	* norm = total_sum * deltax * deltay;
	//printf("Normalization factor (norm): %f\n", *norm);

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
		//printf("All bins are Empty.\n");
		//fgetc(stdin);
	} else {
		//printf("Total empty bins: %d\n", empty_bins);
		printf("Total empty bins: %d\n", empty_bins);

	}

}



char cmd[8192];

int System(char *_cmd) {
	int rt=system(_cmd);
	if (rt/256 != 0) {
		printf("WARNING: could not run command %s\n", _cmd);
	}
	return rt/256;
}


void Mkdir(char *dir) {
	sprintf(cmd, "mkdir %s", dir);
	printf("%s\n", cmd);
	
	System(cmd);
}

//-----
// Looking for Dead Channels
//-----

void analyze_channels(float *data, int64 N, int64 C, int64 sY, int64 sX, double jitter, int *dead_mask) {
    int64 CsYsX = C * sY * sX;
    int64 sYsX = sY * sX;

	for (int c = 0; c < C; c++) {
		float max_val = -INFINITY;
		for (int n = 0; n < N; n++) {
			for (int y = 0; y < sY; y++) {
				for (int x = 0; x < sX; x++) {
					int64 idx = n * CsYsX + c * sYsX + y * sX + x;
					float val = data[idx];
					if (val > max_val) max_val = val;
				}
			}
		}
// Mark channel as dead if max ≤ jitter
		dead_mask[c] = (max_val <= jitter) ? 1 : 0;
	}
}



//int main(int argc, char**argv){
int Run(const char *dataset, const char *model, int seed, const char *layer, int featA, int featB, int nMoments, int nbin, const char *outloss_file, double pi_multiplier) {

	int i,n,y,x,mA,mB;


	// Assign to global vars
	arg_featA = featA;
	arg_featB = featB;
	arg_featA_test = featA;
	arg_featB_test = featB;
	//arg_nbin = nbin;
	//arg_min = min_val;
	//arg_max = max_val;

	// Construct file paths
	
	char output_dir[4096];
	sprintf(output_dir, "pit_archi/%s_%s_%d_%s", dataset, model, seed, layer);
	mkdir("pit_archi", 0777);            // Ensure base directory exists
	mkdir(output_dir, 0777);             // Create specific output folder

	sprintf(arg_infile,      "../features/%s_%s_%d/%s_train.bin", dataset, model, seed, layer);
	sprintf(arg_infile_test, "../features/%s_%s_%d/%s_test.bin", dataset, model, seed, layer);
	
	sprintf(arg_outcsv,  "%s/density_histogram.csv", output_dir);
	sprintf(arg_outcsv,  "%s/hist_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outgcf,  "%s/density_legendre.csv",  output_dir);
	sprintf(arg_outfgcf, "%s/density_fourier.csv",   output_dir);
	sprintf(arg_outmom,  "%s/moments_legendre.csv",  output_dir);
	sprintf(arg_outfmom, "%s/moments_fourier.csv",   output_dir);
	//sprintf(outloss_file, "%s/losses.csv",            output_dir);
	sprintf(arg_outecf, "%s/density_ecf.csv",   output_dir);
	//sprintf(arg_outecfr, "%s/ecf_real_%d_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outecfr, "%s/ecf_real_pi%.1f_%d_%d.csv", output_dir, pi_multiplier, featA, featB);
	//sprintf(arg_outecfi, "%s/ecf_imag_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outecfi, "%s/ecf_imag_pi%.1f_%d_%d.csv", output_dir, pi_multiplier, featA, featB);
	
	sprintf(arg_outcsv,  "%s/hist_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outgcf,  "%s/leg_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outfgcf, "%s/fou_%d_%d.csv", output_dir, featA, featB);
	//sprintf(arg_outecf,  "%s/ecf_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outecf,  "%s/ecf_pi%.1f_%d_%d.csv", output_dir, pi_multiplier, featA, featB);
	//sprintf(arg_norm_ecf, "%s/norm_ecf_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_norm_ecf, "%s/norm_ecf_pi%.1f_%d_%d.csv", output_dir, pi_multiplier, featA, featB);
	sprintf(arg_outmom,  "%s/mom_leg_%d_%d.csv", output_dir, featA, featB);
	sprintf(arg_outfmom, "%s/mom_fou_%d_%d.csv", output_dir, featA, featB);





	
	printf("  [INFO] Using file: %s\n", arg_infile);
	printf("  [INFO] Outputting to: %s\n", output_dir);
	//fgetc(stdin);
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
	//arg_infile_test = argv[2];
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
	printf(" Read Train features\n");
	printf("---------------------------------\n");
	int cA = arg_featA;
	int cB = arg_featB;
	double jitter = 0.0000001;
	
	printf("How many ntrain features ?\n");
	N_train=data_N * data_sY * data_sX;

	printf("num train features %lld\n", N_train);
	
	printf("Allocate train features\n");
	A = (double*)malloc(N_train*sizeof(double));
	B = (double*)malloc(N_train*sizeof(double));
	
	printf("Copy train features plus random jitter\n");
	i=0;
	for (n=0; n<data_N; n++) {
		//printf("n %d/%d\n", n, data_N);
	for (y=0; y<data_sY; y++) {
	for (x=0; x<data_sX; x++) {
		int64 idxA = n*data_CsYsX + cA*data_sYsX + y*data_sX + x;
		int64 idxB = n*data_CsYsX + cB*data_sYsX + y*data_sX + x;
		
//if (n==9467){

//printf("y %d x %d idxA %lld idxB %lld\n", y, x, idxA, idxB); 
//printf(" data[idxA] %f\n", data[idxA]);
//printf(" data[idxB] %f\n", data[idxB]);
//}
		
		
		
		if (data[idxA]!=0 && data[idxB]!=0 && !isnan(data[idxA]) && !isnan(data[idxB]) && !isinf(data[idxA]) && !isinf(data[idxB])) {
			A[i] = data[idxA] + jitter*RANDF;
			B[i] = data[idxB] + jitter*RANDF;
			i++;
		}
	}}}
	N_train = i;
	
	
	
	printf("--------------------------------------------------------------------\n");
	printf(" Sort the features for Copula probability integral transform\n");
	printf("---------------------------------------------------------------------\n");
	
	printf("Allocate sorted arrays\n");
	A_sorted = (double*)malloc(N_train*sizeof(double));
	B_sorted = (double*)malloc(N_train*sizeof(double));
	feat_backup = (double*)malloc(N_train*sizeof(double));

	for (n=0; n<N_train; n++) {
		A_sorted[n] = A[n];
		B_sorted[n] = B[n];
	}

	printf("Sort the data\n");
	Mergesort(A_sorted, feat_backup, N_train);
	Mergesort(B_sorted, feat_backup, N_train);

	free(feat_backup);
	
	printf("Convert to uniform distribution [0, 1]\n");
	UA = (float*)malloc(N_train*sizeof(float));
	UB = (float*)malloc(N_train*sizeof(float));
	double inv_N_train = 1.0/N_train;
	for (n=0; n<N_train; n++) {
		double valA = A[n];
		double valB = B[n];
		int idxA = Binsearch(valA, A_sorted, N_train);
		int idxB = Binsearch(valB, B_sorted, N_train);
		
		//printf("Accessing data[idxA=%d], data[idxB=%d]\n", idxA, idxB);
		//UA[n] = 2.0*(idxA + 0.5)*inv_N_train - 1.0;     // uniform [-1, 1]
		//UB[n] = 2.0*(idxB + 0.5)*inv_N_train - 1.0;     // uniform [-1, 1]
		UA[n] = (idxA + 0.5)*inv_N_train;     // uniform [0, 1]
		UB[n] = (idxB + 0.5)*inv_N_train;     // uniform [0, 1]
	}
	
	
	
	printf("--------------------------------\n");
	printf(" Read the test features\n");
	printf("--------------------------------\n");
	int cA_test = arg_featA_test;
	int cB_test = arg_featB_test;
	
	printf("How many test features ?\n");
	N_test=data_N_test * data_sY_test * data_sX_test;
	
	printf("num test features %lld\n", N_test);
	
	printf("Allocate test features\n");
	A_test  = (double*)malloc(N_test*sizeof(double));  
	B_test  = (double*)malloc(N_test*sizeof(double));
	
	printf("Copy nonzero test features\n");
	i=0;
	for (n=0; n<data_N_test; n++) {
	for (y=0; y<data_sY_test; y++) {
	for (x=0; x<data_sX_test; x++) {
		int64 idxA_test = n*data_CsYsX_test + cA_test*data_sYsX_test + y*data_sX_test + x;
		int64 idxB_test = n*data_CsYsX_test + cB_test*data_sYsX_test + y*data_sX_test + x;
		
		if (data[idxA_test]!=0 && data[idxB_test]!=0 && !isnan(data[idxA_test]) && !isnan(data[idxB_test]) && !isinf(data[idxA_test]) && !isinf(data[idxB_test])) {
			A_test[i] = data[idxA_test] + jitter*RANDF;
			B_test[i] = data[idxB_test] + jitter*RANDF;
			
			i++;
		}
	}}}
	N_test = i;
	
	
	printf("-------------------------------------------------\n");
	printf(" Project test_data to uniform distribution [0, 1]\n");
	printf("-------------------------------------------------\n");
	
	UA_test = (float*)malloc(N_test*sizeof(float));
	UB_test = (float*)malloc(N_test*sizeof(float));
	
	if (A_test == NULL || B_test == NULL) {
		fprintf(stderr, "Memory allocation failed\n");
		return;// -1;  
	}

	double inv_N_test = 1.0/N_test;
	
	for (int n = 0; n < N_test; n++) {
		//printf("N_test: %d\n", N_test);

		double valA_test = A_test[n];  
		double valB_test = B_test[n]; 
	    

		// binary search to find indices of the test data
		int idxA_test = Binsearch(valA_test, A_sorted, N_train);
		int idxB_test = Binsearch(valB_test, B_sorted, N_train);

		
		if (idxA_test < 0 || idxA_test >= N_train || idxB_test < 0 || idxB_test >= N_train) {
			fprintf(stderr, "Invalid index from binary search: idxA_test=%d, idxB_test=%d\n", idxA_test, idxB_test);
			continue; 
		}

		// Map the indices to the uniform distribution [-1, 1]
		UA_test[n] = (idxA_test + 0.5)*inv_N_train;  // uniform [0, 1]
		UB_test[n] = (idxB_test + 0.5)*inv_N_train;  // uniform [0, 1]
    
	}


	printf("Projecting test data is completed.\n");



	printf("----------------------\n");
	printf(" Calculate Spearman's rho\n");
	printf("----------------------\n");
	double mean = 0.5;
	double variance = 1.0/12.0;
	double stdev = sqrt(variance);
	double covar = 0.0;
	for (n=0; n<N_train; n++) {
		covar += (double)(UA[n]-mean)*(double)(UB[n]-mean);
	}
	covar = covar*inv_N_train;
	double rho  = covar / variance;
	
	//printf("Covariance %f Correlation %f\n", covAB, corrAB);
	printf("mean %f vairance %f stdev %f covariance %f rho %f\n", mean, variance, stdev, covar, rho);
	//printf("press enter\n");
	//fgetc(stdin);
	
	printf("----------------------\n");
	printf(" Calculate Kendall's tau\n");
	printf("----------------------\n");
	
	Pair *pairs = (Pair*)malloc(N_train * sizeof(Pair));
	for (n=0; n<N_train; n++) {
		pairs[n].x = UA[n];
		pairs[n].y = UB[n];
	}
	double tau = kendall_tau(pairs, N_train);
	
	printf("tau %f\n", tau);
	
	//printf("press enter\n");
	//fgetc(stdin);
	
	
	printf("----------------------\n");
	printf(" Check if Kendall's tau or Spearman's rho is Negative\n");
	printf("----------------------\n");	
	if (rho < 0.0 || tau < 0.0) {
		printf("Skipping pair featA=%d featB=%d due to negative rho or tau (rho=%f, tau=%f)\n", arg_featA, arg_featB, rho, tau);
		
		// Free memory that has been allocated so far
		free(A); 
		free(B);
		free(A_sorted); 
		free(B_sorted);
		free(UA); 
		free(UB);
		free(A_test); 
		free(B_test);
		free(UA_test); 
		free(UB_test);
		free(pairs);

		UnmapBinfile(datafile);
		UnmapBinfile(testfile);

		return 1;  // Skip the rest of the Run function
	}

	printf("-------------------------------\n");
	printf(" For every copula\n");
	printf("-------------------------------\n");
	
	double cross_entropy_archi[NUM_ARCHI];
	double log_pdf_archi[NUM_ARCHI];

	
	int archi;
	double theta;
	for (archi=0; archi<NUM_ARCHI; archi++)
	{
		printf("------------------------\n");
		printf(" archi %d  fit %s\n", archi, str_archi[archi]);
		printf("------------------------\n");
		
		#define numbin 100
		theta = archi_theta(archi, rho, tau);
		
		//printf(" theta %f\n", theta);
		printf(" archi %d  fit %s\n", archi, str_archi[archi]);
		//printf("press enter\n");
		//fgetc(stdin);
		
		//char outfname[32];
		//sprintf(outfname, "pit_archi/density_%s.csv", str_archi[archi]);
		char outfname[4096];
		//sprintf(outfname, "%s/density_%s.csv", output_dir, str_archi[archi]);
		sprintf(outfname, "%s/archi_%s_%d_%d.csv", output_dir, str_archi[archi], featA, featB);

	
		printf("open %s\n", outfname);
	
		FILE *fcsv = fopen(outfname, "w");
		
		if (fcsv==NULL) {
			printf("ERROR: cannot open %s for writing\n", outfname);
			exit(1);
		}

		printf("write header bins\n");

		fprintf(fcsv, "BinA/BinB");
		for (x=0; x<numbin; x++)
			fprintf(fcsv, ",%d", x);
		fprintf(fcsv, "\n");

		double inv_num_bins = 1.0/numbin;
		
		for (y=0; y<numbin; y++) {
			fprintf(fcsv, "%d", y);
			
			for (x=0; x<numbin; x++) {
				double xA = (y+0.5) * inv_num_bins;
				double xB = (x+0.5) * inv_num_bins;
			
				// Calculate pdf at that test point
				double pdf = archi_copula_density(archi, theta, xA, xB);
				fprintf(fcsv, ",%f", pdf);
			}
			fprintf(fcsv, "\n");
		}
		
		fclose(fcsv);
		
		
		printf("------------------------\n");
		printf(" Plot the archimedian copula \n");
		printf("------------------------\n");
		

		sprintf(cmd, "python3 pit_plot.py %s", outfname);
		printf("system %s\n", cmd);
		system(cmd);

		printf("------------------------\n");
		printf(" Evaluate the archimedian copula \n");
		printf("------------------------\n");
				
		cross_entropy_archi[archi] = 0.0;
		log_pdf_archi[archi]       = 0.0;

		for (int n=0; n<N_test; n++) {

			double UA_test_val = UA_test[n];  
			double UB_test_val = UB_test[n];  

			// Calculate up the PDF value
			double pdf_archi = archi_copula_density(archi, theta, UA_test_val, UB_test_val);
			
			// Convert to density scale of (-1,1)
			pdf_archi *= 0.25;
						
			
			if (pdf_archi<0.0001)
				pdf_archi=0.0001;
	
			////if (strcmp(str_archi[archi], "joe") == 0) {
				////printf("pdf_archi[%d] (%s): %f\n", archi, str_archi[archi], pdf_archi);
				//fgetc(stdin);
			////}
			//if (isnan(pdf_archi) || isinf(pdf_archi) || pdf_archi <= 0.0) {
				//printf("\n[ERROR] Invalid PDF for Gumbel Copula\n");
				//printf("archi = %d (%s)\n", archi, str_archi[archi]);
				//printf("theta = %.16f\n", theta);
				//printf("UA_test_val = %.16f\n", UA_test_val);
				//printf("UB_test_val = %.16f\n", UB_test_val);
				//printf("pdf_archi (before threshold) = %.16f\n", pdf_archi);
				//fgetc(stdin);
				
			//}

			log_pdf_archi[archi] += log(pdf_archi);
			//printf("log_pdf_archi[%d] (%s): %f\n", archi, str_archi[archi], log_pdf_archi[archi]);
			//fgetc(stdin);
			
			if (isnan(log_pdf_archi[archi])) {
				printf("WARNING: log_pdf_archi[%d] is NaN! Press enter to continue...\n", archi);
				//fgetc(stdin);
			}
			cross_entropy_archi[archi] = log_pdf_archi[archi] * (-1.0)/N_test;
		}
		
		//printf("cross_entropy_archi[%d] (%s): %f\n", archi, str_archi[archi], cross_entropy_archi[archi]);

		//printf("press enter\n");
		///fgetc(stdin);		
	}




	

	printf("-------------------------------\n");
	printf(" Calculate ECF\n");
	printf("-------------------------------\n");

	int nEcf = 8*nMoments;
	//int nEcf = 20*nMoments;
	
	double ecf_mean_a = 0.0;
	double ecf_variance_a = 0.0;
	double ecf_stdev_a = 0.0;
	double ecf_mean_b = 0.0;
	double ecf_variance_b = 0.0;
	double ecf_stdev_b = 0.0;

	double  ecf_ka[256];
	double  ecf_kb[256];
	complex ETA[256];
	complex ETB[256];
	complex ET[256][256];
	double ETA_damp[256];
	double ETB_damp[256];
//	double ET_damp[64][64];
	complex emom[256][256];
	for (mA=0; mA<nEcf; mA++) {
		for (mB=0; mB<nEcf; mB++) {
			emom[mA][mB].r = 0.0;
			emom[mA][mB].i = 0.0;
		}
	}

	printf("-------------------------------\n");
	printf(" ECF mean and standard deviation\n");
	printf("-------------------------------\n");

	// ECF mean
	ecf_mean_a = 0.0;
	ecf_mean_b = 0.0;
	for (i=0; i<N_train; i++) {
		ecf_mean_a += A[i];
		ecf_mean_b += B[i];
	}
	ecf_mean_a /= N_train;
	ecf_mean_b /= N_train;


	// ECF variance and stdev
	ecf_variance_a = 0.0;
	ecf_variance_b = 0.0;
	for (i=0; i<N_train; i++) {
		ecf_variance_a += (A[i]-ecf_mean_a)*(A[i]-ecf_mean_a);
		ecf_variance_b += (B[i]-ecf_mean_b)*(B[i]-ecf_mean_b);
	}
	ecf_variance_a /= N_train;
	ecf_variance_b /= N_train;
	ecf_stdev_a = sqrt(ecf_variance_a);
	ecf_stdev_b = sqrt(ecf_variance_b);

	// Assign ka and kb
	
	double ecf_ka0 = ecf_mean_a - pi_multiplier *PI * ecf_stdev_a;
	double ecf_ka1 = ecf_mean_a + pi_multiplier *PI * ecf_stdev_a;
	double ecf_kb0 = ecf_mean_b - pi_multiplier *PI * ecf_stdev_b;
	double ecf_kb1 = ecf_mean_b + pi_multiplier *PI * ecf_stdev_b;
	double ecf_ka_step = (ecf_ka1-ecf_ka0) / (nEcf-1);
	double ecf_kb_step = (ecf_kb1-ecf_kb0) / (nEcf-1);


//	printf("ecf_mean_a %f ecf_stdev_a %f ecf_ka0 %f ecf_ka1 %f ecf_ka_step %f\n", ecf_mean_a, ecf_stdev_a, ecf_ka0, ecf_ka1, ecf_ka_step);
//	printf("ecf_mean_b %f ecf_stdev_b %f ecf_kb0 %f ecf_kb1 %f ecf_kb_step %f\n", ecf_mean_b, ecf_stdev_b, ecf_kb0, ecf_kb1, ecf_kb_step);
//	fgetc(stdin);

	for (mA=0; mA<nEcf; mA++)
		ecf_ka[mA] = ecf_ka0 + ecf_ka_step * mA;
	for (mB=0; mB<nEcf; mB++)
		ecf_kb[mB] = ecf_kb0 + ecf_kb_step * mB;

	printf("-------------------------------\n");
	printf(" ECF moments\n");
	printf("-------------------------------\n");

	for (n=0; n<N_train; n++)
	{
		double xA = A[n] + 1.0;
		double xB = B[n] + 1.0;

		// Calculate Fourier function values
//		ETA[0].r = 0.70710678118;                       // sqrt(2)/2
//		ETA[0].i = 0.0;
		for (mA=0; mA<nEcf; mA++) {
			double ka = ecf_ka[mA];
			ETA[mA].r = cos( ka * xA );
			ETA[mA].i = sin( ka * xA );
		}
		
//		ETB[0].r = 0.70710678118;
//		ETB[0].i = 0.0;
		for (mB=0; mB<nEcf; mB++) {
			double kb = ecf_kb[mB];
			ETB[mB].r = cos( kb * xB );
			ETB[mB].i = sin( kb * xB );
		}
		
		// Calculate the Fourier of A,B and compute MoM
		for (mA=0; mA<nEcf; mA++) {
			for (mB=0; mB<nEcf; mB++) {
				ET[mA][mB]      = complex_mul(ETA[mA], ETB[mB]);   // Calculate Fourier
				emom[mA][mB].r += ET[mA][mB].r;		   // Add to MoM
				emom[mA][mB].i += ET[mA][mB].i;
			}
		}
	}
	for (mA=0; mA<nEcf; mA++) {
		for (mB=0; mB<nEcf; mB++) {
			emom[mA][mB].r /= N_train;		   // Add to MoM
			emom[mA][mB].i /= N_train;
		}
	}
	
	printf(" Save the ECF Moments (real)\n");
	FILE *fecfr = fopen(arg_outecfr, "w");
	if (fecfr == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outecfr);
		exit(1);
	}
	
	fprintf(fecfr, "\t\t");
	for (mB=0; mB<nEcf; mB++)
		fprintf(fecfr, "%f\t", ecf_kb[mB]);
	fprintf(fecfr, "\n");
	for (mB=0; mB<nEcf; mB++)
		fprintf(fecfr, "%d\t", mB);
	fprintf(fecfr, "\n");
	for (mA=0; mA<nEcf; mA++) {
		fprintf(fecfr, "%f\t%d\t", ecf_ka[mA], mA);
		for (mB=0; mB<nEcf; mB++)
			fprintf(fecfr, "%f\t", emom[mA][mB].r);
		fprintf(fecfr, "\n");
	}
	
	fclose(fecfr);
	
	
	printf(" Save the ECF Moments (imaginary)\n");
	FILE *fecfi = fopen(arg_outecfi, "w");
	if (fecfi == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outecfi);
		exit(1);
	}
	
	fprintf(fecfi, "\t\t");
	for (mB=0; mB<nEcf; mB++)
		fprintf(fecfi, "%f\t", ecf_kb[mB]);
	fprintf(fecfi, "\n");
	for (mB=0; mB<nEcf; mB++)
		fprintf(fecfi, "%d\t", mB);
	fprintf(fecfi, "\n");
	for (mA=0; mA<nEcf; mA++) {
		fprintf(fecfi, "%f\t%d\t", ecf_ka[mA], mA);
		for (mB=0; mB<nEcf; mB++)
			fprintf(fecfi, "%f\t", emom[mA][mB].i);
		fprintf(fecfi, "\n");
	}
	
	fclose(fecfi);


	printf("------------------------------\n");
	printf(" Plot the ECF PDF\n");
	printf("------------------------------\n");
	#define num_bin 100
	double ecf_damp_integral = 0.0;
	double ecf_raw_integral = 0.0;
	double damp_buffer[num_bin][num_bin]; 
	double deltaxy = 4.0 / (num_bin * num_bin);
	

	
	//double inv_num_bins = 1.0/num_bin;
	
	// Calculate our dampening factors
//	ETA_damp[0] = 1.0;
	for (mA=0; mA<nEcf; mA++) {
		double k = ecf_ka[mA];
		double b = 2.0*ecf_ka_step;
		ETA_damp[mA] = (2.0*(1.0 - cos(k*b))) / (b*b*k*k);
		//printf("mA %d ETA_damp %f k %f b %f\n", mA, ETA_damp[mA], k, b);
		//fgetc(stdin);
	}
//	ETB_damp[0] = 1.0;
	for (mB=0; mB<nEcf; mB++) {
		double k = ecf_kb[mB];
		double b = 2.0*ecf_kb_step;
		ETB_damp[mB] = (2.0*(1.0 - cos(k*b))) / (b*b*k*k);
		//printf("mB %d ETB_damp %f k %f b %f\n", mB, ETB_damp[mB], k, b);
	}
	
	
	FILE *ecf_file = fopen(arg_outecf, "w");
	if (ecf_file == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outecf);
		exit(1);
	}

	// Write header row
	fprintf(ecf_file, "BinA/BinB");
	//fgetc(stdin);
	for (int x = 0; x < num_bin; x++) {
		fprintf(ecf_file, ",%d", x);
	}
	fprintf(ecf_file, "\n");
	
	
	for (int y = 0; y < num_bin; y++) {
		fprintf(ecf_file, "%d", y);
		for (int x = 0; x < num_bin; x++) {
			
			int iA = (int)((x * (N_train - 1)) / (num_bin - 1));
			int iB = (int)((y * (N_train - 1)) / (num_bin - 1));
			//printf("iA = %d, iB = %d\n", iA, iB);
			
			
			//fgetc(stdin);
			
			//indxA = MINMAX(iA, 0, N_train - 1);
        		//indxB = MINMAX(iB, 0, N_train - 1);

			double xA = A_sorted[iA] + 1.0;
			double xB = B_sorted[iB] + 1.0;
			//printf("x=%d, y=%d | iA=%d, iB=%d | xA=%.6f, xB=%.6f\n", x, y, iA, iB, xA, xB);
			//fgetc(stdin);
			
			// Calculate ECF function values
//			ETA[0].r = 0.70710678118;                       // sqrt(2)/2
//			ETA[0].i = 0.0;
//			printf("mA %d Eta (%f + i %f)\n", 0, ETA[0].r, ETA[0].i);
			
			for (mA=0; mA<nEcf; mA++) {
				double ka = ecf_ka[mA];
				//ETA[mA].r = cos( ka * (xA+1) );
				//ETA[mA].i = sin( ka * (xA+1) );
				ETA[mA].r =  OVER_2PI * cos( ka * (xA)) * ecf_ka_step;
				ETA[mA].i = -OVER_2PI * sin( ka * (xA)) * ecf_ka_step;
	//			printf("mA %d Eta (%f + i %f)\n", mA, ETA[mA].r, ETA[mA].i);
			}
//			ETB[0].r = 0.70710678118;                       // sqrt(2)/2
//			ETB[0].i = 0.0;
//			printf("mB %d Etb (%f + i %f)\n", 0, ETB[0].r, ETB[0].i);
			
			for (mB=0; mB<nEcf; mB++) {
				double kb = ecf_kb[mB];
				//ETB[mB].r = cos( kb * (xB+1) );
				//ETB[mB].i = sin( kb * (xB+1) );
				ETB[mB].r =  OVER_2PI * cos( kb * (xB)) * ecf_kb_step;
				ETB[mB].i = -OVER_2PI * sin( kb * (xB)) * ecf_kb_step;
	//			printf("mB %d Etb (%f + i %f)\n", mB, ETB[mB].r, ETB[mB].i);
			}

			

			// Calculate pdf at that point
			double raw_pdf_ecf = 0.0;
			double damp_pdf_ecf = 0.0;
			//int num = 0;
			for (mA=0; mA<nEcf; mA++){
				for (mB=0; mB<nEcf; mB++){
				
					if (mB >= 0 && mB < nEcf) {
						// Safe to access TB[mB]
					} else {
						printf("Invalid index for mB: %d\n", mB);
						////fgetc(stdin);
						// Handle error or continue to the next iteration
					}
				
					complex et = complex_mul(ETA[mA], ETB[mB]);
					complex raw_dpdf_ri = complex_mul(emom[mA][mB], et);
					double raw_dpdf = raw_dpdf_ri.r;
					double damp_dpdf = raw_dpdf * ETA_damp[mA] * ETA_damp[mB];
					//printf(" mA %d mB %d et (%f + i %f)  raw_dpdf_ri (%f + i %f)  raw_dpdf %f damp_dpdf %f\n", mA, mB, et.r, et.i, raw_dpdf_ri.r, raw_dpdf_ri.i, raw_dpdf, damp_dpdf);
					raw_pdf_ecf += raw_dpdf;
					damp_pdf_ecf += damp_dpdf;
				}
			}
			
			//printf("raw_pdf_ecf %f  damp_pdf_ecf %f\n", raw_pdf_ecf, damp_pdf_ecf);

			// Convert to copula space
			int idx_step = 1000;
			int idxA = Binsearch(xA, A_sorted, N_train);
			int idxB = Binsearch(xB, B_sorted, N_train);
			int idxA0 = idxA-idx_step;
			int idxA1 = idxA+idx_step;
			int idxB0 = idxB-idx_step;
			int idxB1 = idxB+idx_step;
			idxA0 = MINMAX(idxA0,0,N_train-1);
			idxA1 = MINMAX(idxA1,0,N_train-1);
			idxB0 = MINMAX(idxB0,0,N_train-1);
			idxB1 = MINMAX(idxB1,0,N_train-1);
			double yA0 = (double)idxA0 / (double)(N_train-1);
			double yA1 = (double)idxA1 / (double)(N_train-1);
			double yB0 = (double)idxB0 / (double)(N_train-1);
			double yB1 = (double)idxB1 / (double)(N_train-1);
			double xA0 = A_sorted[idxA0];
			double xA1 = A_sorted[idxA1];
			double xB0 = B_sorted[idxB0];
			double xB1 = B_sorted[idxB1];
			double dxA = xA1-xA0;
			double dyA = yA1-yA0;
			double dxB = xB1-xB0;
			double dyB = yB1-yB0; 
			dyA *= 2.0;
			dyB *= 2.0;

			// Copula density
			double raw_dens_ecf  = raw_pdf_ecf  * (dxA*dxB) / (dyA*dyB);
			double damp_dens_ecf = damp_pdf_ecf * (dxA*dxB) / (dyA*dyB);
			//damp_buffer[y][x] = damp_dens_ecf; 


			if (raw_dens_ecf<0.0001)
				raw_dens_ecf=0.0001;
			if (damp_dens_ecf<0.0001)
				damp_dens_ecf=0.0001;

			damp_buffer[y][x]  = damp_dens_ecf;
			ecf_damp_integral += damp_dens_ecf;
			ecf_raw_integral  += raw_dens_ecf;
			
			//damp_buffer[y][x] = damp_dens_ecf;  
			//ecf_damp_integral += damp_dens_ecf * deltaxy;  


			//printf("xA (%.16f %.16f)  yA (%f %f)  xB (%.16f %.16f)  yB (%f %f)\n", xA0, xA1, yA0, yA1, xB0, xB1, yB0, yB1);
			//printf("raw_dens_ecf %f  damp_dens_ecf %f\n", raw_dens_ecf, damp_dens_ecf);
			//printf("ecf_damp_integral %f ", ecf_damp_integral);
			
			//fgetc(stdin);


		fprintf(ecf_file, ",%f", damp_dens_ecf);  
//		fprintf(ecf_file, ",%f", damp_pdf_ecf);  
		}

	fprintf(ecf_file, "\n");
	}
	printf("ecf_damp_integral %f ", ecf_damp_integral);
	fclose(ecf_file);
	printf("ECF Copula Density saved to %s\n", arg_outecf);
	//fgetc(stdin);
	
	
	//printf("damp_dens_ecf sum %f  integral %f  scale %f\n", ecf_damp_integral, ecf_damp_integral*deltaxy, 1.0/(ecf_damp_integral*deltaxy));
	//printf("raw_dens_ecf sum %f  integral %f  scale %f\n",  ecf_raw_integral, ecf_raw_integral*deltaxy, 1.0/(ecf_raw_integral*deltaxy));
	//fgetc(stdin);
	
	double scale = 1.0 / (ecf_damp_integral * deltaxy);
	
	FILE *norm_file = fopen(arg_norm_ecf, "w");
	if (norm_file == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_norm_ecf);
		exit(1);
	}

	// Write header
	fprintf(norm_file, "BinA/BinB");
	for (int x = 0; x < num_bin; x++)
		fprintf(norm_file, ",%d", x);
	fprintf(norm_file, "\n");

	// Write normalized values
	for (int y = 0; y < num_bin; y++) {
		fprintf(norm_file, "%d", y);
		for (int x = 0; x < num_bin; x++) {
			double val = damp_buffer[y][x] * scale;
			fprintf(norm_file, ",%f", 4.0*val);
		}
		fprintf(norm_file, "\n");
	}

	fclose(norm_file);
	printf("Normalized ECF saved to %s\n", arg_norm_ecf);
	
	//fgetc(stdin);
	
	printf("------------------------\n");
	printf(" Plot the ECF \n");
	printf("------------------------\n");
	sprintf(cmd, "python3 pit_plot.py %s", arg_outecf);
	printf("system %s\n", cmd);
	system(cmd);
	printf("------------------------\n");
	printf("ECF PLOTTED !!!");
	printf("------------------------\n");
	//fgetc(stdin);
	
	printf("------------------------\n");
	printf(" Plot the Normalized ECF \n");
	printf("------------------------\n");
	sprintf(cmd, "python3 pit_plot.py %s", arg_norm_ecf);
	printf("system %s\n", cmd);
	system(cmd);
	printf("------------------------\n");
	printf("ECF Norm PLOTTED !!!");
	printf("------------------------\n");
	//fgetc(stdin);


	printf("------------------------------\n");
	printf(" Evaluation of the ECF PDF\n");
	printf("------------------------------\n");


	double raw_cross_entropy_ecf  = 0.0;
	double damp_cross_entropy_ecf = 0.0;
	double raw_log_dens_ecf       = 0.0;
	double damp_log_dens_ecf      = 0.0;

	for (int n = 0; n < N_test; n++) {
		
		// Use indexed values of UA_test and UB_test arrays
		double xA = A_test[n] + 1.0;
		double xB = B_test[n] + 1.0;

		//printf("xA %.16f xB %.16f\n", xA, xB);
		
		for (mA=0; mA<nEcf; mA++) {
			double ka = ecf_ka[mA];
			//ETA[mA].r = cos( ka * (xA+1) );
			//ETA[mA].i = sin( ka * (xA+1) );
			ETA[mA].r =  OVER_2PI * cos( ka * (xA)) * ecf_ka_step;
			ETA[mA].i = -OVER_2PI * sin( ka * (xA)) * ecf_ka_step;
//			printf("mA %d Eta (%f + i %f)\n", mA, ETA[mA].r, ETA[mA].i);
		}
//			ETB[0].r = 0.70710678118;                       // sqrt(2)/2
//			ETB[0].i = 0.0;
//			printf("mB %d Etb (%f + i %f)\n", 0, ETB[0].r, ETB[0].i);
		
		for (mB=0; mB<nEcf; mB++) {
			double kb = ecf_kb[mB];
			//ETB[mB].r = cos( kb * (xB+1) );
			//ETB[mB].i = sin( kb * (xB+1) );
			ETB[mB].r =  OVER_2PI * cos( kb * (xB)) * ecf_kb_step;
			ETB[mB].i = -OVER_2PI * sin( kb * (xB)) * ecf_kb_step;
//			printf("mB %d Etb (%f + i %f)\n", mB, ETB[mB].r, ETB[mB].i);
		}

		

		// Calculate pdf at that point
		double raw_pdf_ecf = 0.0;
		double damp_pdf_ecf = 0.0;
		//int num = 0;
		for (mA=0; mA<nEcf; mA++){
			for (mB=0; mB<nEcf; mB++){
			
				if (mB >= 0 && mB < nEcf) {
					// Safe to access TB[mB]
				} else {
					printf("Invalid index for mB: %d\n", mB);
					////fgetc(stdin);
					// Handle error or continue to the next iteration
				}
			
				complex et = complex_mul(ETA[mA], ETB[mB]);
				complex raw_dpdf_ri = complex_mul(emom[mA][mB], et);
				double raw_dpdf = raw_dpdf_ri.r;
				double damp_dpdf = raw_dpdf * ETA_damp[mA] * ETA_damp[mB];
				//printf(" mA %d mB %d et (%f + i %f)  raw_dpdf_ri (%f + i %f)  raw_dpdf %f damp_dpdf %f\n", mA, mB, et.r, et.i, raw_dpdf_ri.r, raw_dpdf_ri.i, raw_dpdf, damp_dpdf);
				raw_pdf_ecf += raw_dpdf;
				damp_pdf_ecf += damp_dpdf;
			}
		}
	
		//printf("raw_pdf_ecf %f  damp_pdf_ecf %f\n", raw_pdf_ecf, damp_pdf_ecf);

		// Convert to copula space
		int idx_step = 1000;
		int idxA = Binsearch(xA, A_sorted, N_train);
		int idxB = Binsearch(xB, B_sorted, N_train);
		int idxA0 = idxA-idx_step;
		int idxA1 = idxA+idx_step;
		int idxB0 = idxB-idx_step;
		int idxB1 = idxB+idx_step;
		idxA0 = MINMAX(idxA0,0,N_train-1);
		idxA1 = MINMAX(idxA1,0,N_train-1);
		idxB0 = MINMAX(idxB0,0,N_train-1);
		idxB1 = MINMAX(idxB1,0,N_train-1);
		double yA0 = (double)idxA0 / (double)(N_train-1);
		double yA1 = (double)idxA1 / (double)(N_train-1);
		double yB0 = (double)idxB0 / (double)(N_train-1);
		double yB1 = (double)idxB1 / (double)(N_train-1);
		double xA0 = A_sorted[idxA0];
		double xA1 = A_sorted[idxA1];
		double xB0 = B_sorted[idxB0];
		double xB1 = B_sorted[idxB1];
		double dxA = xA1-xA0;
		double dyA = yA1-yA0;
		double dxB = xB1-xB0;
		double dyB = yB1-yB0; 
		dyA *= 2.0;
		dyB *= 2.0;

		// Copula density
		double raw_dens_ecf  = raw_pdf_ecf  * (dxA*dxB) / (dyA*dyB);
		double damp_dens_ecf = damp_pdf_ecf * (dxA*dxB) / (dyA*dyB);

		raw_dens_ecf *= 1.0/(ecf_raw_integral*deltaxy);
		damp_dens_ecf *= 1.0/(ecf_damp_integral*deltaxy);

		//printf("xA (%.16f %.16f)  yA (%f %f)  xB (%.16f %.16f)  yB (%f %f)  idxA %d idxB\n", xA0, xA1, yA0, yA1, xB0, xB1, yB0, yB1, idxA, idxB);
		//printf("raw_dens_ecf %f  damp_dens_ecf %f\n", raw_dens_ecf, damp_dens_ecf);


		if (raw_dens_ecf<0.0001)
			raw_dens_ecf=0.0001;
		if (damp_dens_ecf<0.0001)
			damp_dens_ecf=0.0001;
		
		
		raw_log_dens_ecf = log(raw_dens_ecf);
		damp_log_dens_ecf = log(damp_dens_ecf);
		
		//printf("raw_log_dens_ecf %f  damp_log_dens_ecf %f\n", raw_log_dens_ecf, damp_log_dens_ecf);
		
		if (raw_dens_ecf<=0.0) {
			//printf("WARNING n %d raw_dens_ecf %f\n", n, raw_log_dens_ecf);
		}
		if (damp_dens_ecf<=0.0) {
			//printf("WARNING n %d raw_dens_ecf %f\n", n, damp_log_dens_ecf);
		}

		// Calculate cross-entropy using the log(pdf)
		raw_cross_entropy_ecf  +=  raw_log_dens_ecf * (-1.0)/N_test;
		damp_cross_entropy_ecf += damp_log_dens_ecf * (-1.0)/N_test;

		//printf("raw_cross_entropy_ecf %f  damp_cross_entropy_ecf %f\n", raw_cross_entropy_ecf, damp_cross_entropy_ecf);
		
		//printf("press enter\n");
		//fgetc(stdin);
	}	


	//printf("raw_cross_entropy_ecf %f\n", raw_cross_entropy_ecf);
	//printf("damp_cross_entropy_ecf %f\n", damp_cross_entropy_ecf);

	//printf("press enter\n");
	//fgetc(stdin);



	
	printf("---------------------------\n");
	printf(" Calculate Legendre Moments\n");
	printf("---------------------------\n");
	double  TA[64];
	double  TB[64];
	double   T[64][64];
	double mom[64][64];
	
	
	
	FILE *csvfile_mom = fopen(arg_outmom, "w");
	//FILE *csvfile_mom = fopen("pit_archi/moments_legendre.csv", "w");	
	if (csvfile_mom == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outmom);
		//printf("ERROR: cannot open %s for writing\n", "pit_archi/moments_legendre.csv");
		
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
		double xA = 2.0*UA[n] - 1.0;
	
		// Calculate Legendre of B
		double xB = 2.0*UB[n] - 1.0;

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

			//printf("MoM_A_%d_B_%d_,%f\n", mA, mB, mom[mA][mB]);
		}
		fprintf(csvfile_mom, "\n"); 
	}
	printf("MOMENTS are saved as 2D matrix to %s\n", arg_outmom);
	fclose(csvfile_mom);	
	
	
	//printf("Covariance %f Correlation %f\n", covAB, corrAB);
	
	
	
	printf("--------------------------\n");
	printf(" Calculate Fourier Moments\n");
	printf("---------------------------\n");
	//#define nMoments 11
	double  FTA[64];
	double  FTB[64];
	double   FT[64][64];
	double fmom[64][64];
	
	
	//FILE *csvfile_fmom = fopen("pit_archi/moments_fourier.csv", "w");
	FILE *csvfile_fmom = fopen(arg_outfmom, "w");
	if (csvfile_fmom == NULL) {
		//printf("ERROR: cannot open %s for writing\n", "pit_archi/moments_fourier.csv");
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
		double xA = 2.0 * UA[n] - 1.0;
	
		// Calculate Fourier of B
		double xB = 2.0 * UB[n] - 1.0;


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

			//printf("Fourier MoM_A_%d_B_%d_,%f\n", mA, mB, fmom[mA][mB]);
		}
		fprintf(csvfile_fmom, "\n"); 
	}
	printf("Fourier MOMENTS are saved as 2D matrix to %s\n", arg_outfmom);
	fclose(csvfile_fmom);	
	
	printf("---------------------------------------------\n");
	printf(" Calculate Histogram of data  Range is [-1 1]\n");
	printf("---------------------------------------------\n");
	
	int num_bins = nbin; //arg_nbin;
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

		int binA = (int)(UA[i]*num_bins);
		int binB = (int)(UB[i]*num_bins);

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
		int UA_index = (int)(UA_test_val * num_bins);
		int UB_index = (int)(UB_test_val * num_bins);
		
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
		//printf("pdf_hist = %f, \n", pdf_hist);
		check_empty_bins(histogram, arg_nbin);
		//fgetc(stdin);
		
		//total_pdf_hist += pdf_hist;
		if(pdf_hist == 0){
			printf("Warning: Empty bin at index %d\n", hist_index);
			pdf_hist = 0.0001;
			
		}
		log_pdf_hist += log(pdf_hist);

		cross_entropy_hist =  log_pdf_hist * (-1.0)/N_test;	
	}

	
	printf("------------------------\n");
	printf(" write out the Histogram\n");
	printf("------------------------\n");
		
	// Write the histogram as a 2D matrix

	//FILE *csvfile = fopen("pit_archi/density_histogram.csv", "w");
	FILE *csvfile = fopen(arg_outcsv, "w");
	

	if (csvfile == NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outcsv);
		//printf("ERROR: cannot open %s for writing\n", "pit_archi/density_histogram.csv");
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
	      		
			fprintf(csvfile, ",%f", 4*normalized_histogram[binA * num_bins + binB]);
	

		}
		fprintf(csvfile, "\n");  
	}

	printf("Histogram saved as 2D matrix to %s\n", arg_outcsv);
	fclose(csvfile);
	
		
	printf("------------------------\n");
	printf(" Plot the histogram \n");
	printf("------------------------\n");
	
	sprintf(cmd, "python3 pit_plot.py %s", arg_outcsv);
	//sprintf(cmd, "python3 pit_plot.py %s", "pit_archi/density_histogram.csv");
	printf("system %s\n", cmd);
	system(cmd);
	

	printf("--------------------------------------\n");
	printf(" Calculate Legendre Characteristic PDF\n");
	printf("--------------------------------------\n");
	
	#define num_bins 100
	
	//FILE *gcffile = fopen("pit_archi/density_legendre.csv", "w");
	FILE *gcffile = fopen(arg_outgcf, "w");
	if (gcffile==NULL) {
		//printf("ERROR: cannot open %s for writing\n", "pit_archi/density_legendre.csv");
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
			fprintf(gcffile, ",%f", 4*pdf);
		}
		fprintf(gcffile, "\n");
	}
	
	fclose(gcffile);


	printf("------------------------\n");
	printf(" Plot the Legendre \n");
	printf("------------------------\n");
	
	sprintf(cmd, "python3 pit_plot.py %s", arg_outgcf);
	//sprintf(cmd, "python3 pit_plot.py %s", "pit_archi/density_legendre.csv");
	printf("system %s\n", cmd);
	system(cmd);

	printf("-------------------------------------\n");
	printf(" Calculate Fourier Characteristic PDF \n");
	printf("--------------------------------------\n");
	
	#define num_bins 100
	//FILE *fgcffile = fopen("pit_archi/density_fourier.csv", "w");
	FILE *fgcffile = fopen(arg_outfgcf, "w");
	
	if (gcffile==NULL) {
		printf("ERROR: cannot open %s for writing\n", arg_outfgcf);
		//printf("ERROR: cannot open %s for writing\n", "pit_archi/density_fourier.csv");
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
			fprintf(fgcffile, ",%f", 4*pdf);
		}
		fprintf(fgcffile, "\n");
	}
	
	fclose(fgcffile);


	printf("------------------------\n");
	printf(" Plot the Fourier \n");
	printf("------------------------\n");
	
	sprintf(cmd, "python3 pit_plot.py %s", arg_outfgcf);
	//sprintf(cmd, "python3 pit_plot.py %s", "pit_archi/density_fourier.csv");
	printf("system %s\n", cmd);
	system(cmd);



	printf("-------------------------------\n");
	printf(" Evaluation of the Legendre PDF\n");
	printf("--------------------------------\n");


	double cross_entropy_legendre= 0.0;

	double log_pdf_legendre = 0.0;
	  
	  
	for (int n = 0; n < N_test; n++) {
		
		// Use indexed values of UA_test and UB_test arrays
		double UA_test_val = 2.0 * UA_test[n] - 1.0;  
		double UB_test_val = 2.0 * UB_test[n] - 1.0;  
		
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
				////	fgetc(stdin);
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
			printf("N_test %d\n", N_test);
			
			
			

			////fgetc(stdin);
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
		double UA_test_val = 2.0 * UA_test[n] - 1.0;  
		double UB_test_val = 2.0 * UB_test[n] - 1.0;  
		
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
					////fgetc(stdin);
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
			printf("N_test %d\n", N_test);

			
			

			////fgetc(stdin);
		}

		// Calculate cross-entropy using the log(pdf)
		cross_entropy_fourier =  log_pdf_fourier * (-1.0)/N_test;
	}	

	for (archi=0; archi<NUM_ARCHI; archi++)
		printf("Cross-Entropy of %s Estimation %f\n", str_archi[archi], cross_entropy_archi[archi]);

	printf("Cross-Entropy of Histogram Estimation: %f\n", cross_entropy_hist);
	printf("Cross-Entropy of Legendre Polynomial Estimation: %f\n", cross_entropy_legendre);	
	printf("Cross-Entropy of Fourier Estimation: %f\n", cross_entropy_fourier);	
	
	printf("Cross-Entropy of Raw-ECF Estimation: %f\n", raw_cross_entropy_ecf);
	printf("Cross-Entropy of Damp-ECF Estimation: %f\n", damp_cross_entropy_ecf);
	

	printf("------------------------------\n");
	printf(" Saving Losses to a CSV File\n");
	printf("------------------------------\n");	

	FILE *csvfile_loss = fopen(outloss_file, "a");
	//FILE *csvfile_loss = fopen("pit_archi/losses.csv", "a");
	
	//FILE *csv_file_loss = fopen("cross_entropy_results.csv", "a"); 
	
	//sprintf(loss_path, "%s/losses.csv", arg_outloss);
	
	
	//FILE *csvfile_loss = fopen(loss_path, "a");
	
	
	
	if (csvfile_loss == NULL) {
		printf("Error opening file!\n", outloss_file);
		//printf("Error opening file!\n", "pit_archi/losses.csv");
		//printf("Error opening file!\n", loss_path);
	return 1;
	}

	// Write header only once if the file is empty
	fseek(csvfile_loss, 0, SEEK_END);
	long size = ftell(csvfile_loss);
	if (size == 0) {
		
		
		////fprintf(csvfile_loss, "Feature A_test,Feature B_test,Cross-Entropy Histogram,Cross-Entropy Legendre,Cross-Entropy Fourier, Raw-ECF Cross_Entropy, Damp-ECF Cross_Entropy");
		fprintf(csvfile_loss, "Feature A_test,Feature B_test,legendre,fourier,histogram, raw-ecf, damp-ecf");
		for (int archi = 0; archi < NUM_ARCHI; archi++) {
			////fprintf(csvfile_loss, ",%s Cross_Entropy ", str_archi[archi]);
			fprintf(csvfile_loss, ",%s", str_archi[archi]);
		}
		//fprintf(csvfile_loss, "\n");
		fprintf(csvfile_loss, ",Kendall_Tau,Spearman_Rho\n");  
	}
	

	fprintf(csvfile_loss, "%d,%d,%f,%f,%f,%f,%f", arg_featA,arg_featB,cross_entropy_legendre, cross_entropy_fourier, cross_entropy_hist, raw_cross_entropy_ecf,damp_cross_entropy_ecf);
	
	for (int archi = 0; archi < NUM_ARCHI; archi++) {
		fprintf(csvfile_loss, ",%f", cross_entropy_archi[archi]);
	}
	//fprintf(csvfile_loss, ",Kendall_Tau,Spearman_Rho\n");
	fprintf(csvfile_loss, ",%f,%f", tau, rho);

	//fprintf(csvfile_loss, "\n");

	//printf("Losses are saved as 2D matrix to %s\n", "pit_archi/losses.csv");
	printf("Losses are saved as 2D matrix to %s\n", outloss_file);
	
	//printf("Losses are saved as 2D matrix to %s\n", loss_path);
	fprintf(csvfile_loss, "\n");
	fclose(csvfile_loss);


	

	
	printf("----------------------\n");
	printf(" Unmap the dataset features\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);	
	UnmapBinfile(testfile);	


	printf("----------------------\n");
	printf(" Free everything\n");
	printf("----------------------\n");
	free(A);
	free(B);
	free(A_sorted);
	free(B_sorted);
	free(UA);
	free(UB);
	free(A_test);
	free(B_test);
	free(UA_test);
	free(UB_test);
	free(pairs);

	
	//return 0;
}

int main() {

	char *datasets[1] = {"mnist"};
	char *models[1] = {"resnet18"};
	char *layers[1][1] = {{"A"}};
	//char *layers[1][2] = {{"X", "A"}};
	//char *layers[1][3] = {{"A", "X", "D"}};
	//char *layers[1][4] = {{"X", "A", "B", "C"}};
	//char *layers[1][2] = {{"C","B"}};
	//char *datasets[4] = {"imagenette2","cifar100", "cifar10", "mnist"};
	//char *models[3] = {"resnet18", "resnet50", "vgg19"};
	//char *layers[3][6] = {{"X", "A", "B", "C", "D"}, {"X", "A", "B", "C", "D"}, {"A", "B", "C", "D", "E"}};
	
	int seed = 0;
	int L, M, D;
	char outloss_file[4096];
	const int MAX_RANDOM_PAIRS = 300; //1000
	const int SELECTED_PAIRS = 25;
	//int featA = 19;			    
	//int featB = 41;
	//sprintf(outloss_file, "pit_archi/%s_%s_%d_%s/loss.csv", datasets, models, seed, layers);
	
	double pi_multiplier;
	int max_try = MAX_RANDOM_PAIRS;
	
	//int tried = 0;
	
	
	for (D = 0; D < 1; D++) {
		for (M = 0; M <1; M++) {
			for (L = 0; L < 1; L++) {
				int selected = 0;
				char *dataset = datasets[D];
				char *model = models[M];
				char *layer = layers[M][L];
				if      (strcmp(dataset,"imagenette2")==0){
					if      (strcmp(layer, "X") == 0) pi_multiplier = 5.0;
					else if (strcmp(layer, "A") == 0) pi_multiplier = 5.0;
					else if (strcmp(layer, "B") == 0) pi_multiplier = 7.0;
					else if (strcmp(layer, "C") == 0) pi_multiplier = 7.0;
					else if (strcmp(layer, "D") == 0) pi_multiplier = 1.0;
					else {
						fprintf(stderr, "Unknown layer %s, using default pi_multiplier = 2.0\n", layer);
						pi_multiplier = 2.0;
					}
				}
				else if (strcmp(dataset,"cifar100")==0){
					if      (strcmp(layer, "X") == 0) pi_multiplier = 27.0;
					else if (strcmp(layer, "A") == 0) pi_multiplier = 20.0;
					else if (strcmp(layer, "B") == 0) pi_multiplier = 18.0;
					else if (strcmp(layer, "C") == 0) pi_multiplier = 18.0;
					else if (strcmp(layer, "D") == 0) pi_multiplier = 4.0;
					else {
						fprintf(stderr, "Unknown layer %s, using default pi_multiplier = 2.0\n", layer);
						pi_multiplier = 2.0;
					}
				}
				else if (strcmp(dataset,"cifar10")==0){
					if      (strcmp(layer, "X") == 0) pi_multiplier = 27.0;  // Smaller might bee betther for some pairs and worse for some other pairs !!!
					else if (strcmp(layer, "A") == 0) pi_multiplier = 20.0;  
					else if (strcmp(layer, "B") == 0) pi_multiplier = 18.0;
					else if (strcmp(layer, "C") == 0) pi_multiplier = 22.0;
					else if (strcmp(layer, "D") == 0) pi_multiplier = 12.0;  
					else {
						fprintf(stderr, "Unknown layer %s, using default pi_multiplier = 2.0\n", layer);
						pi_multiplier = 2.0;
					}
				}
				else if (strcmp(dataset,"mnist")==0){
					if      (strcmp(layer, "X") == 0) pi_multiplier = 110.0;
					else if (strcmp(layer, "A") == 0) pi_multiplier = 90.0;
					else if (strcmp(layer, "B") == 0) pi_multiplier = 80.0;
					else if (strcmp(layer, "C") == 0) pi_multiplier = 70.0;
					else if (strcmp(layer, "D") == 0) pi_multiplier = 12.0;
					else {
						fprintf(stderr, "Unknown layer %s, using default pi_multiplier = 2.0\n", layer);
						pi_multiplier = 2.0;
					}
				}
			// Step 1: Load data once to get data_C (channels/features)
				char tmp_file[1024];
				sprintf(tmp_file, "../features/%s_%s_%d/%s_test.bin", dataset, model, seed, layer);
				Binfile tmp_data = MapBinfileR(tmp_file);
				int total_features = tmp_data.shape[1];  // number of channels
				
				// Map data to compute dead channels
				float *tmp_data_ptr = (float *)tmp_data.data;
				int64 tmp_N  = tmp_data.shape[0];
				int64 tmp_C  = tmp_data.shape[1];
				int64 tmp_sY = tmp_data.shape[2];
				int64 tmp_sX = tmp_data.shape[3];

				int *dead_mask = (int *)malloc(tmp_C * sizeof(int));
				double jitter = 0.0000001;  
				analyze_channels(tmp_data_ptr, tmp_N, tmp_C, tmp_sY, tmp_sX, jitter, dead_mask);

				UnmapBinfile(tmp_data);
				
				// Step 2: Generate 1000 unique random feature pairs, skipping dead channels
				int (*pairs)[2] = malloc(MAX_RANDOM_PAIRS * sizeof(int[2]));
				if (pairs == NULL) {
					fprintf(stderr, "Failed to allocate memory for feature pairs\n");
					exit(1);
				}
				
				srand(time(NULL));
				int count = 0;
				while (count < MAX_RANDOM_PAIRS) {
					int a = rand() % total_features;
					int b = rand() % total_features;
					if (a != b && !dead_mask[a] && !dead_mask[b]) {
						if (a > b) { int temp = a; a = b; b = temp; }

						int exists = 0;
						for (int k = 0; k < count; k++) {
							if (pairs[k][0] == a && pairs[k][1] == b) {
								exists = 1;
								break;
							}
				        	}

				        if (!exists) {
						pairs[count][0] = a;
						pairs[count][1] = b;
						count++;
				        }
				    }
				}


				// Step 3: Shuffle the pairs
				for (int i = MAX_RANDOM_PAIRS - 1; i > 0; i--) {
					int j = rand() % (i + 1);
					int tmpA = pairs[i][0], tmpB = pairs[i][1];
					pairs[i][0] = pairs[j][0];
					pairs[i][1] = pairs[j][1];
					pairs[j][0] = tmpA;
					pairs[j][1] = tmpB;
				}
				
				// Step 4: Run analysis for first 35 pairs
				sprintf(outloss_file, "pit_archi/%s_%s_%d_%s/loss.csv", dataset, model, seed, layer);
				//for (int p = 0; p < SELECTED_PAIRS && p < MAX_RANDOM_PAIRS; p++) {
				//	int featA = pairs[p][0];
				//	int featB = pairs[p][1];


				for (int p = 0; p < max_try && selected < SELECTED_PAIRS; p++) {
					int featA = pairs[p][0];
					int featB = pairs[p][1];

					//printf("Trying pair %d: featA=%d featB=%d\n", p, featA, featB);
					printf("Running for dataset=%s model=%s layer=%s featA=%d featB=%d\n",
					   dataset, model, layer, featA, featB);
					int status = Run(dataset, model, seed, layer, featA, featB, 11, 10, outloss_file, pi_multiplier);
					//Run(dataset, model, seed, layer, featA, featB, 11, 10, outloss_file, pi_multiplier);
					
					if (status == 0) {
						selected++;  // accepted
					} else {
						printf("Rejected pair due to negative rho/tau.\n");
					}
				}
				
				///printf("Running for dataset=%s model=%s layer=%s featA=%d featB=%d\n",dataset, model, layer, featA, featB);
					
					//Run(dataset, model, seed, layer, featA, featB, 11, 10, outloss_file,pi_multiplier );
				//}

				free(pairs);
				free(dead_mask);
            		}
		}
	}

	return 0;
}
				









