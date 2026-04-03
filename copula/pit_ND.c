#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "../mmap.h"


#define int64 long long int
#define uint16 unsigned short int
#define MAX_SHORT 0x00010000

#define PI 3.14159265358979323846264
#define RANDF   ( rand() / (float)RAND_MAX )

//#define MIN(a,b)  ( (a)<(b) ) ? (a) : (b)
//#define MAX(a,b)  ( (a)<(b) ) ? (b) : (a)
#define MINMAX(x,min,max)   MIN(max,MAX(x,min))

// Easy string formatting
char path[4096];
char *cat(const char*str1, const char*str2) {
	(sprintf)(path, "%s%s", str1, str2);
	return path;
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


//---------
// Output folder
//---------
char run_outdir[4096];
char run_infile[4096];
char run_infile_test[4096];
char run_outcsv[6144];
char run_outgcf[6144];

FILE *run_floss;
char run_outloss[6144];

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
// CDF for histograms
//--------------------------
#define CDF_BINS 10000
#define MAX_FEAT 2048         // was 512 before
int cdf_histo[MAX_FEAT][CDF_BINS];
int cdf_nzero[MAX_FEAT];
float cdf_minval[MAX_FEAT];
float cdf_maxval[MAX_FEAT];

//--------------------------
// Feature data (Train)
//--------------------------
uint16 *U_train[MAX_FEAT];    // uniform distributed training data
int64 N_train;

//--------------------------
// Feature data (Test)
//--------------------------
uint16 *U_test[MAX_FEAT];    // uniform distributed test data
int64 N_test;

//convert from [0 MAX_SHORT] to [-1 1]
#define TO_INTERVAL(val)   ((float)(val) * 0.000030517578125 - 1.0)

//--------------------------
// List of features that we selected
//--------------------------
int feat_indices[MAX_FEAT];
int feat_N;


//---------
// Histogram
//---------
double *histo;
int histo_nBins, histo_total_bins;

#define MAX_BIN 256
double hist_plot[MAX_BIN][MAX_BIN];
int hist_count[MAX_FEAT];
int hist_idx;

void hist_count_reset()
{
	int c;
	hist_idx = 0;
	for (c=0; c<MAX_FEAT; c++)
		hist_count[c] = 0;
}
void hist_count_next()
{
	int c;

	hist_idx++;
	
	// add one to counter
	hist_count[0]++;
	
	// carry the ones
	for (c=0; c<MAX_FEAT; c++) {
		if (hist_count[c]<histo_nBins)
			break;
		
		hist_count[c]=0;
		if (c<MAX_FEAT-1)
			hist_count[c+1]++;
	}
		
}


//---------
// Counter to keep track of moments
//---------
int mom_total_moments;
int mom_count[MAX_FEAT];
int mom_idx;
int mom_nMoments=11;
#define MAX_MOMENTS 64

double *mom_legendre;
double *mom_fourier;

void mom_count_reset()
{
	int c;
	mom_idx = 0;
	for (c=0; c<MAX_FEAT; c++)
		mom_count[c] = 0;
}

void mom_count_next()
{
	int c;

	mom_idx++;
	
	// add one to counter
	mom_count[0]++;
	
	// carry the ones
	for (c=0; c<MAX_FEAT; c++) {
		if (mom_count[c]<mom_nMoments)
			break;
		
		mom_count[c]=0;
		if (c<MAX_FEAT-1)
			mom_count[c+1]++;
	}
		
}

void Run()
{
	int nFeat = feat_N;
	int n,c,m,y,x,i,j,mA,mB;

	printf("-------------------------\n");
	printf(" Convert features to uniform distribution  (train)\n");
	printf("-------------------------\n");
	
	printf("Convert the training features\n");
	for (n=0; n<data_N; n++) {
		if (n%500==0) printf("n %d/%lld\n", n, data_N);
		
		for (c=0; c<nFeat; c++) {
			int cidx = feat_indices[c];
printf("10\n");		
			float *map = data + n*data_CsYsX + cidx*data_sYsX;
			int nidx = n * data_sYsX;
		
printf("20\n");		
			// How to split up the zero and non-zero values . . .
			float frac_zero    = (float)cdf_nzero[cidx] / (float)N_train;
			float frac_nonzero = 1.0 - frac_zero;
			int n_zero_bins = MAX_SHORT * frac_zero;
			int n_nonzero_bins = MAX_SHORT - n_zero_bins;
			
printf("30\n");		
			float min = cdf_minval[cidx];
			float max = cdf_maxval[cidx];
			float scale = 1.0 / (max-min);
			float histo_scale = 1.0 / (N_train * frac_nonzero);

printf("40 n %d c %d\n", n, c);		
/*
			printf("cidx %d\n", cidx);
			printf("cdf_nzero[cidx] %d\n", cdf_nzero[cidx]);
			printf("N_train %d\n", N_train);
			printf("frac_zero %f\n", frac_zero);
			printf("frac_nonzero %f\n", frac_nonzero);
			printf("n_zero_bins %d\n", n_zero_bins);
			printf("n_nonzero_bins %d\n", n_nonzero_bins);
			printf("min %f\n", min);
			printf("max %f\n", max);
			printf("scale %f\n", scale);
			printf("histo_scale %f\n", histo_scale);
			printf("enter\n");
			fgetc(stdin);
			*/
			for (i=0; i<data_sYsX; i++) {
				float val = map[i];
//if (n==1196) printf("41\n");				
				if (nidx+i>=N_train) {
					printf("ASSERT:  nidx+i>=N_train\n");
					exit(1);
				}
			
//if (n==1196) printf("42  n_zero_bins %d\n", n_zero_bins);
				if (val<=0)
					U_train[c][nidx+i] = rand() % MAX(n_zero_bins,1);
				else
				{
//if (n==1196) printf("43\n");				
					// Which CDF bin ?
					int cdf_bin = 0;
					if (val<min)
						cdf_bin = 0;
					else if (val >= max)
						cdf_bin = CDF_BINS-1;
					else {
						cdf_bin = CDF_BINS*(val-min)*scale;
						cdf_bin = MINMAX(cdf_bin,0,CDF_BINS-1);
					}
					
//if (n==1196) printf("44\n");				
					// Read the CDF and convert to short bins
					float cdf_val = histo_scale * cdf_histo[cidx][cdf_bin];    // CDF value between 0 and 1
					int cdf_short_bin = cdf_val * n_nonzero_bins + n_zero_bins;
					cdf_short_bin = MINMAX(cdf_short_bin, n_zero_bins, MAX_SHORT-1);
					
//					printf("c %d  cidx %d  val %f minmax %f %f  cdf_bin %d  cdf_val %f   cdf_short_bin 0x%x\n", c, cidx, val, min, max, cdf_bin, cdf_val, cdf_short_bin);
					
//if (n==1196) printf("45\n");				
					// Store the short bin into U
					U_train[c][nidx+i] = (uint16)cdf_short_bin;
//if (n==1196) printf("46\n");				
				}
			}
//if (n==1196) printf("50\n");		
//			printf("enter\n");
//			fgetc(stdin);
		}
	}
printf("60\n");		

	

	printf("-------------------------\n");
	printf(" Convert features to uniform distribution  (test)\n");
	printf("-------------------------\n");
	
	printf("Convert the testing features\n");
	for (n=0; n<data_N_test; n++) {
		if (n%500==0) printf("n %d/%lld\n", n, data_N_test);
		
		for (c=0; c<nFeat; c++) {
			int cidx = feat_indices[c];
		
			float *map = data_test + n*data_CsYsX_test + cidx*data_sYsX_test;
			int nidx = n * data_sYsX_test;
		
			// How to split up the zero and non-zero values . . .
			float frac_zero    = (float)cdf_nzero[cidx] / (float)N_train;
			float frac_nonzero = 1.0 - frac_zero;
			int n_zero_bins = MAX_SHORT * frac_zero;
			int n_nonzero_bins = MAX_SHORT - n_zero_bins;
			
			float min = cdf_minval[cidx];
			float max = cdf_maxval[cidx];
			float scale = 1.0 / (max-min);
			float histo_scale = 1.0 / (N_train * frac_nonzero);
			
			for (i=0; i<data_sYsX_test; i++) {
				float val = map[i];
			
				if (nidx+i>=N_test) {
					printf("ASSERT: nidx+i>=N_test\n");
					exit(1);
				}
			
				if (val<=0)
					U_test[c][nidx+i] = rand() % MAX(n_zero_bins,1);
				else
				{
					// Which CDF bin ?
					int cdf_bin = 0;
					if (val<min)
						cdf_bin = 0;
					else if (val >= max)
						cdf_bin = CDF_BINS-1;
					else {
						cdf_bin = CDF_BINS*(val-min)*scale;
						cdf_bin = MINMAX(cdf_bin,0,CDF_BINS-1);
					}
					
					// Read the CDF and convert to short bins
					float cdf_val = histo_scale * cdf_histo[cidx][cdf_bin];    // CDF value between 0 and 1
					int cdf_short_bin = cdf_val * n_nonzero_bins + n_zero_bins;
					cdf_short_bin = MINMAX(cdf_short_bin, n_zero_bins, MAX_SHORT-1);
					
//					printf("cdf_val %f   cdf_short_bin 0x%x\n", cdf_val, cdf_short_bin);

					// Store the short bin into U
					U_test[c][nidx+i] = (uint16)cdf_short_bin;
				}
			}
		}
	}

	
	
	
	
	printf("---------------------------\n");
	printf(" Allocate the moments\n");
	printf("---------------------------\n");
	mom_total_moments=1;
	for (c=0; c<nFeat; c++)
		mom_total_moments *= mom_nMoments;

	mom_legendre = (double*)calloc(mom_total_moments, sizeof(double));
	mom_fourier  = (double*)calloc(mom_total_moments, sizeof(double));
	
	printf("mom_total_moments %d\n", mom_total_moments);
	printf("mom_legendre %p\n", mom_legendre);
	printf("mom_fourier %p\n", mom_fourier);
	//printf("enter\n"); fgetc(stdin);
	


	
	printf("---------------------------\n");
	printf(" Calculate Histogram\n");
	printf("---------------------------\n");
	
	double delta_x = 2.0 / histo_nBins;
	double over_N_train = 1.0 / N_train;
	for (c=0; c<nFeat; c++)
		over_N_train /= delta_x;
	
	for (i=0; i<histo_total_bins; i++)
		histo[i] = 0.0;
	
	for (n=0; n<N_train; n++)
	{
		if (n%100000 == 0) printf("n %d/%lld\n", n, N_train);

		// Find the histogram index
		int hist_idx = 0;
		int step=1;
		for (c=0; c<nFeat; c++) {
			int bin = ((unsigned int)U_train[c][n] * histo_nBins) >> 16;
			hist_idx += bin*step;
			step*=histo_nBins;
		}
		
		// Bin the value
		histo[hist_idx] += over_N_train;
	}



	printf("---------------------------\n");
	printf(" Calculate Legendre Moments\n");
	printf("---------------------------\n");
	double T[64][64];   // 1D moments
	
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
	
	double inv_N_train = 1.0 / N_train;
	for (n=0; n<N_train; n++)
	{
		if (n%100000 == 0) printf("n %d/%lld\n", n, N_train);

		// Calculate the 1D moment contributions
		for (c=0; c<nFeat; c++) {
		
			// Calculate Ledgendre polynomial function values
			double x = TO_INTERVAL(U_train[c][n]);
			T[c][0] = 1.0;
			T[c][1] = x;
			for (m=2; m<mom_nMoments; m++)
				T[c][m] = ((2*m-1.0)/m)*x*T[c][m-1] - ((m-1.0)/m)*T[c][m-2];
				
			// Rescale polynomials to unit length
			for (m=0; m<mom_nMoments; m++)
				T[c][m] *= legendre_scaling[m];
				
			//for (m=0; m<mom_nMoments; m++)
			//	printf("n  %d  T[%d][%d] %f    x %f  U_train %d\n", n, c, m, T[c][m], x, U_train[c][n]);
		}
		
		// Multiply to get ND moments
		for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
		{
			// Multiply the moment contributions
			double contrib = 1.0;
			for (c=0; c<nFeat; c++) {
				m = mom_count[c];
				contrib *= T[c][m];
			}
		
			// Add to the total
			mom_legendre[mom_idx] += contrib * inv_N_train;
		}
	}
	
	for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
	{
		printf("mom_idx %d ", mom_idx);
		for (c=0; c<nFeat; c++)
			printf("%d ", mom_count[c]);
		printf("   legendre %16f\n", mom_legendre[mom_idx]);
	}
	//printf("enter\n");
	//fgetc(stdin);
	
	printf("---------------------------\n");
	printf(" Calculate Fourier Moments\n");
	printf("---------------------------\n");

	for (n=0; n<N_train; n++)
	{
		if (n%100000 == 0) printf("n %d/%lld\n", n, N_train);

		// Calculate the 1D moment contributions
		for (c=0; c<nFeat; c++) {
		
			// Calculate Fourier function values
			double x = TO_INTERVAL(U_train[c][n]);
			T[c][0] = 0.70710678118;               // sqrt(2)/2
			for (m=1; m<mom_nMoments; m++)
				T[c][m] = cos( 0.5*m*PI * (x-1.0) );
		}
		
		// Multiply to get ND moments
		for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
		{
			// Multiply the moment contributions
			double contrib = 1.0;
			for (c=0; c<nFeat; c++) {
				m = mom_count[c];
				contrib *= T[c][m];
			}
		
			// Add to the total
			mom_fourier[mom_idx] += contrib * inv_N_train;
		}
	}
		
	for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
	{
		printf("mom_idx %d ", mom_idx);
		for (c=0; c<nFeat; c++)
			printf("%d ", mom_count[c]);
		printf("   fourier %16f\n", mom_fourier[mom_idx]);
	}
	//printf("enter\n");
	//fgetc(stdin);



	
	printf("---------------------------\n");
	printf(" TODO MAKE PLOTS\n");
	printf("---------------------------\n");

	double TA[MAX_MOMENTS];
	double TB[MAX_MOMENTS];

	int j_step = mom_nMoments;

	// Scaling to convert to 2D plot
	double rescale_2d = 1.0;
	for (c=0; c<nFeat-2; c++)
		rescale_2d /= 0.70710678118;

	for (j=1; j<nFeat; j++)
	{
		int i_step = 1;
		for (i=0; i<j; i++)
		{
			int A = feat_indices[i];
			int B = feat_indices[j];
						
			int num_bins = 100;


			printf("---------------------------\n");
			printf(" Histogram Plot features %d and %d\n", A, B);
			printf("---------------------------\n");
			
			double histo_deltax_scale = 1.0;
			for (c=0; c<nFeat-2; c++)
				histo_deltax_scale *= delta_x;
			
			printf(" reset the hist plot\n");
			for (y=0; y<histo_nBins; y++)
				for (x=0; x<histo_nBins; x++)
					hist_plot[y][x] = 0.0;
			
			printf(" tally up the hist plot\n");
			for (hist_count_reset(); hist_idx<histo_total_bins; hist_count_next()) {
				y = hist_count[i];
				x = hist_count[j];
				hist_plot[y][x] += histo[hist_idx] * histo_deltax_scale;
			}
			
			
			sprintf(run_outgcf, "%s/his_%d_%d.csv", run_outdir, A, B);
			printf("%s\n", run_outgcf);
			
			FILE *gcffile = fopen(run_outgcf, "w");
			
			if (gcffile==NULL) {
				printf("ERROR: cannot open %s for writing\n", run_outgcf);
				exit(1);
			}

			fprintf(gcffile, "BinA/BinB");
			for (x=0; x<histo_nBins; x++)
				fprintf(gcffile, ",%d", x);
			fprintf(gcffile, "\n");

			for (y=0; y<histo_nBins; y++) {
				fprintf(gcffile, "%d", y);
				
				for (x=0; x<histo_nBins; x++) {
					
					// Calculate pdf at that test point
					double pdf = hist_plot[y][x];
					fprintf(gcffile, ",%f", pdf);
				}
				fprintf(gcffile, "\n");
			}
			
			fclose(gcffile);
	

			
			printf("------\n");
			printf(" Make the plot\n");
			printf("------\n");
			sprintf(cmd, "python3 pit_plot.py %s", run_outgcf);
			System(cmd);



			printf("---------------------------\n");
			printf(" Legendre Plot features %d and %d\n", A, B);
			printf("---------------------------\n");
			
			sprintf(run_outgcf, "%s/leg_%d_%d.csv", run_outdir, A, B);
			printf("%s\n", run_outgcf);
			
			gcffile = fopen(run_outgcf, "w");
			
			if (gcffile==NULL) {
				printf("ERROR: cannot open %s for writing\n", run_outgcf);
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
					for (mA=2; mA<mom_nMoments; mA++)
						TA[mA] = ((2*mA-1.0)/mA)*xA*TA[mA-1] - ((mA-1.0)/mA)*TA[mA-2];
					TB[0] = 1;
					TB[1] = xB;
					for (mB=2; mB<mom_nMoments; mB++)
						TB[mB] = ((2*mB-1.0)/mB)*xB*TB[mB-1] - ((mB-1.0)/mB)*TB[mB-2];
					
					// Rescale polynomials to unit length
					for (mA=0; mA<mom_nMoments; mA++)
						TA[mA] *= legendre_scaling[mA];
					for (mB=0; mB<mom_nMoments; mB++)
						TB[mB] *= legendre_scaling[mB];		

					// Calculate pdf at that test point
					double pdf = 0.0;
					for (mA=0; mA<mom_nMoments; mA++){
						for (mB=0; mB<mom_nMoments; mB++){
							int curr_mom_idx = mA*i_step + mB*j_step;
							double mom = mom_legendre[curr_mom_idx] * rescale_2d;
							double cont = mom*TA[mA]*TB[mB];
							pdf += cont;
						}
					}
					fprintf(gcffile, ",%f", pdf);
				}
				fprintf(gcffile, "\n");
			}
			
			fclose(gcffile);
			
			
			
			printf("------\n");
			printf(" Make the plot\n");
			printf("------\n");
			sprintf(cmd, "python3 pit_plot.py %s", run_outgcf);
			System(cmd);
			
			

			printf("---------------------------\n");
			printf(" Fourier Plot features %d and %d\n", A, B);
			printf("---------------------------\n");
			
			sprintf(run_outgcf, "%s/fou_%d_%d.csv", run_outdir, A, B);
			printf("%s\n", run_outgcf);
			
			gcffile = fopen(run_outgcf, "w");
			
			if (gcffile==NULL) {
				printf("ERROR: cannot open %s for writing\n", run_outgcf);
				exit(1);
			}

			fprintf(gcffile, "BinA/BinB");
			for (x=0; x<num_bins; x++)
				fprintf(gcffile, ",%d", x);
			fprintf(gcffile, "\n");

			inv_num_bins = 1.0/num_bins;
	
			for (y=0; y<num_bins; y++) {
				fprintf(gcffile, "%d", y);
				for (x=0; x<num_bins; x++) {
					double xA = 2.0 * (y+0.5) * inv_num_bins - 1.0;
					double xB = 2.0 * (x+0.5) * inv_num_bins - 1.0;
		
		
					// Calculate Fourier function values
					TA[0] = 0.70710678118;                       // sqrt(2)/2
					for (mA=1; mA<mom_nMoments; mA++)
						TA[mA] = cos( 0.5*mA*PI * (xA-1.0) );
			
					TB[0] = 0.70710678118;
					for (mB=1; mB<mom_nMoments; mB++)
						TB[mB] = cos( 0.5*mB*PI * (xB-1.0) );

					// Calculate pdf at that test point
					double pdf = 0.0;
					for (mA=0; mA<mom_nMoments; mA++){
						for (mB=0; mB<mom_nMoments; mB++){
							int curr_mom_idx = mA*i_step + mB*j_step;
							
							//printf("mom_fourier %d/%d  mA %d mB %d  i %d j %d i_step %d j_step %d\n", curr_mom_idx, mom_total_moments, mA, mB, i, j, i_step, j_step);
							double mom = mom_fourier[curr_mom_idx] * rescale_2d;
							double cont = mom*TA[mA]*TB[mB];
							pdf += cont;
						}
					}
					fprintf(gcffile, ",%f", pdf);
				}
				fprintf(gcffile, "\n");
			}
			
			fclose(gcffile);
			
			
			printf("------\n");
			printf(" Make the plot\n");
			printf("------\n");
			sprintf(cmd, "python3 pit_plot.py %s", run_outgcf);
			System(cmd);
			


			
			
			i_step *= mom_nMoments;
		}
		
		j_step *= mom_nMoments;
	}


	printf("Write out the features to the loss file\n");
	for (c=0; c<nFeat; c++)
		fprintf(run_floss, "%d,", feat_indices[c]);
	fflush(run_floss);

	printf("-------------------------------\n");
	printf(" Evaluation of the Legendre PDF\n");
	printf("--------------------------------\n");

	double cross_entropy_legendre= 0.0;
	double log_pdf_legendre = 0.0;	  
	  
	for (int n = 0; n < N_test; n++) {
		
		// Calculate the 1D moment contributions
		for (c=0; c<nFeat; c++) {
		
			// Calculate Ledgendre polynomial function values
			double x = TO_INTERVAL(U_test[c][n]);
			T[c][0] = 1.0;
			T[c][1] = x;
			for (m=2; m<mom_nMoments; m++)
				T[c][m] = ((2*m-1.0)/m)*x*T[c][m-1] - ((m-1.0)/m)*T[c][m-2];
				
			// Rescale polynomials to unit length
			for (m=0; m<mom_nMoments; m++)
				T[c][m] *= legendre_scaling[m];
		}
		
		// Calculate the legendre pdf
		double pdf_legendre = 0.0;
		for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
		{
			// Multiply the moment contributions
			double contrib = mom_legendre[mom_idx];
			for (c=0; c<nFeat; c++) {
				m = mom_count[c];
				contrib *= T[c][m];
			}
			
			// What is the order of this moment
			int order = 0;
			for (c=0; c<nFeat; c++)
				order += mom_count[c];
			if (order < mom_nMoments)
				pdf_legendre += contrib;
		}
		if (pdf_legendre<0.0001)
			pdf_legendre=0.0001;
		
		
		log_pdf_legendre += log(pdf_legendre);
		
		if (pdf_legendre<=0.0) {
			//num +=1;
			printf("WARNING n %d pdf_legendre %f\n", n, pdf_legendre);
			//fgetc(stdin);
		}

		// Calculate cross-entropy using the log(pdf)
		cross_entropy_legendre =  log_pdf_legendre * (-1.0)/N_test;
	}	
	fprintf(run_floss, "%f,", cross_entropy_legendre);


	printf("-------------------------------\n");
	printf(" Evaluation of the Fourier PDF\n");
	printf("--------------------------------\n");

	double cross_entropy_fourier= 0.0;
	double log_pdf_fourier = 0.0;	  
	  
	for (int n = 0; n < N_test; n++) {
		
		// Calculate the 1D moment contributions
		for (c=0; c<nFeat; c++) {
		
			// Calculate Fourier polynomial function values
			double x = TO_INTERVAL(U_test[c][n]);
			T[c][0] = 0.70710678118;               // sqrt(2)/2
			for (m=1; m<mom_nMoments; m++)
				T[c][m] = cos( 0.5*m*PI * (x-1.0) );
		}
		
		// Calculate the fourier pdf
		double pdf_fourier = 0.0;
		for (mom_count_reset(); mom_idx < mom_total_moments; mom_count_next())
		{
			// Multiply the moment contributions
			double contrib = mom_fourier[mom_idx];
			for (c=0; c<nFeat; c++) {
				m = mom_count[c];
				contrib *= T[c][m];
			}
			int order = 0;
			for (c=0; c<nFeat; c++)
				order += mom_count[c];
			if (order < mom_nMoments)
				pdf_fourier += contrib;
		}
		if (pdf_fourier<0.0001)
			pdf_fourier=0.0001;
		
		
		log_pdf_fourier += log(pdf_fourier);
		
		if (pdf_fourier<=0.0) {
			//num +=1;
			printf("WARNING n %d pdf_fourier %f\n", n, pdf_fourier);
			//fgetc(stdin);
		}

		// Calculate cross-entropy using the log(pdf)
		cross_entropy_fourier =  log_pdf_fourier * (-1.0)/N_test;
	}	
	fprintf(run_floss, "%f,", cross_entropy_fourier);


	printf("-------------------------------\n");
	printf(" Evaluation of the Histogram PDF\n");
	printf("--------------------------------\n");

	double cross_entropy_histogram = 0.0;
	double log_pdf_histogram = 0.0;	
	
	for (n=0; n<N_test; n++)
	{
		if (n%100000 == 0) printf("n %d/%lld\n", n, N_test);

		// Find the histogram index
		int hist_idx = 0;
		int step=1;
		for (c=0; c<nFeat; c++) {
			int bin = ((unsigned int)U_test[c][n] * histo_nBins) >> 16;
			hist_idx += bin*step;
			step*=histo_nBins;
		}
		
		// Calculate cross entropy
		double pdf_histogram = histo[hist_idx];
		printf("pdf_histogram %f\n", pdf_histogram);
		if (pdf_histogram<0.0001)
			pdf_histogram=0.0001;

		
		log_pdf_histogram += log(pdf_histogram);

		if (pdf_histogram<=0.0) {
			//num +=1;
			printf("WARNING n %d pdf_histo %f\n", n, pdf_histogram);
			//fgetc(stdin);
		}
		
		// Calculate cross-entropy using the log(pdf)
		cross_entropy_histogram =  log_pdf_histogram * (-1.0)/N_test;
	}

	
	fprintf(run_floss, "%f,\n", cross_entropy_histogram);
	
	
	
	printf("-------------------------------\n");
	printf(" Clean up\n");
	printf("--------------------------------\n");
	
	
	printf("Free the moments\n");
	free(mom_legendre);
	free(mom_fourier);
	
}

void RunAll(
	const char *dataset, const char *model, int seed,
	const char *layer, int nFeat, int nMoments, int nBins, int nRuns)
{
	int b,n,c,y,x,r,i,j;


	printf("-------------------------\n");
	printf(" Create the output folders\n");
	printf("-------------------------\n");
	sprintf(run_outdir,      "pitnd/%s_%s_%d_%s_f_%d_m_%d_b_%d/", dataset, model, seed, layer, nFeat, nMoments, nBins);
	sprintf(run_infile,      "../features/%s_%s_%d/%s_train.bin", dataset, model, seed, layer);
	sprintf(run_infile_test, "../features/%s_%s_%d/%s_test.bin", dataset, model, seed, layer);
	sprintf(run_outloss,     "%s/loss.csv", run_outdir);
	printf("run_outdir %s\n", run_outdir);
	printf("run_infile %s\n", run_outdir);
	printf("run_infile_test %s\n", run_outdir);
	printf("run_outloss %s\n", run_outloss);

	Mkdir("pitnd");
	Mkdir(run_outdir);

	printf("construct the cross entropy file %s\n", run_outloss);
	run_floss = fopen(run_outloss, "w");
	if (run_floss==NULL) {
		printf("ERROR: could not open %s for writing\n", run_outloss);
		exit(1);
	}
	for (i=0; i<nFeat; i++)
		fprintf(run_floss, ",");
	fprintf(run_floss, "legendre,fourier,histogram,\n");
	fflush(run_floss);


	printf("-------------------------\n");
	printf(" Memory map the data file\n");
	printf("-------------------------\n");
	
	datafile = MapBinfileR(run_infile);
	printf("Opening train file: %s\n", run_infile);
	data = (float*)datafile.data;
	data_N = datafile.shape[0];
	data_C = datafile.shape[1];
	data_sY = datafile.shape[2];
	data_sX = datafile.shape[3];
	data_CsYsX = data_C*data_sY*data_sX;
	data_sYsX = data_sY*data_sX;
	N_train = data_N * data_sYsX;
	
	printf("data_N %lld data_C %lld data_sY %lld data_sX %lld data_CsYsX %lld data_sYsX %lld\n", data_N, data_C, data_sY, data_sX, data_CsYsX, data_sYsX);

	
	printf("------------------------------\n");
	printf(" Memory map the test data file\n");
	printf("------------------------------\n");

	testfile = MapBinfileR(run_infile_test);
	printf("Opening test file: %s\n", run_infile_test);
	
	data_test = (float*)testfile.data;
	data_N_test = testfile.shape[0];
	data_C_test = testfile.shape[1];
	data_sY_test = testfile.shape[2];
	data_sX_test = testfile.shape[3];
	data_CsYsX_test = data_C_test * data_sY_test * data_sX_test;
	data_sYsX_test = data_sY_test * data_sX_test;
	N_test = data_N_test * data_sYsX_test;

	printf("data_N_test %lld data_C_test %lld data_sY_test %lld data_sX_test %lld data_CsYsX_test %lld data_sYsX_test %lld\n", 
		data_N_test, data_C_test, data_sY_test, data_sX_test, data_CsYsX_test, data_sYsX_test);


	if (data_C > MAX_FEAT) {
		printf("ERROR: data_C %lld > MAX_FEAT %d\n", data_C, MAX_FEAT);
		exit(1);
	}


	printf("------------------------------\n");
	printf(" Allocate the histogram\n");
	printf("------------------------------\n");

	histo_nBins = nBins;
	histo_total_bins = 1;
	for (c=0; c<nFeat; c++)
		histo_total_bins *= histo_nBins;
	histo = (double*)malloc(histo_total_bins * sizeof(double));	
	
	printf("histo %p nBins %d total_bins %d\n", histo, histo_nBins, histo_total_bins);


	printf("------------------------------\n");
	printf(" Calculate a CDF\n");
	printf("------------------------------\n");
	
	printf("Zero out the histogram\n");
	for (c=0; c<MAX_FEAT; c++) {
		cdf_nzero[c]  = 0;
		cdf_minval[c] = 0.0;         // Hack set minimum always to zero
		cdf_maxval[c] = -99999999;
		for (b=0; b<CDF_BINS; b++)
			cdf_histo[c][b] = 0.0;
	}
	
	printf("Calculate min and max feature values\n");
	int64 idx=0;
	for (n=0; n<data_N; n++) {
		if (n%500==0) printf("n %d/%lld\n", n, data_N);
	for (c=0; c<data_C; c++) {
	for (y=0; y<data_sY; y++) {
	for (x=0; x<data_sX; x++) {
		float val = data[idx++];
	
		if (val>0.0) {	// clipping for zero features
			cdf_minval[c] = MIN(cdf_minval[c], val);
			cdf_maxval[c] = MAX(cdf_maxval[c], val);
		}
	}}}}
	
	printf("Calculate the PDF\n");
	idx=0;
	for (n=0; n<data_N; n++) {
		if (n%500==0) printf("n %d/%lld\n", n, data_N);
	for (c=0; c<data_C; c++) {
	for (y=0; y<data_sY; y++) {
	for (x=0; x<data_sX; x++) {
		float val = data[idx++];
		float min = cdf_minval[c];
		float max = cdf_maxval[c];
	
		if (val>0.0) {	// clipping for zero features
			int bin = CDF_BINS * (val-min) / (max-min);
			bin = MINMAX(bin, 0, CDF_BINS-1);
			cdf_histo[c][bin]++;
		}
		else
			cdf_nzero[c]++;
	}}}}
	
//	for (b=0; b<CDF_BINS; b++)
//		printf("pdf_histo[0][%d] %d\n", b, cdf_histo[0][b]);
//	printf("cdf_minval[0] %f  cdf_maxval[0] %f\n", cdf_minval[0], cdf_maxval[0]);
//	printf("enter\n");
//	fgetc(stdin);
	
	printf("Sum the CDF\n");
	idx=0;
	for (c=0; c<MAX_FEAT; c++) {
	for (b=1; b<CDF_BINS; b++) {
		cdf_histo[c][b] += cdf_histo[c][b-1];
	}}

//	for (b=0; b<CDF_BINS; b++)
//		printf("cdf_histo[0][%d] %d\n", b, cdf_histo[0][b]);
//	printf("enter\n");
//	fgetc(stdin);



	printf("------------------------------\n");
	printf(" Allocate the train/test uniform feature data\n");
	printf("------------------------------\n");
	for (i=0; i<nFeat; i++) {
		U_train[i] = (uint16*)malloc(N_train * sizeof(uint16));
		U_test[i]  = (uint16*)malloc(N_test * sizeof(uint16));
		
		printf("U_train[%d] %p   N_train %lld\n", i, U_train[i], N_train);
		printf("U_test[%d] %p   N_test %lld\n", i, U_test[i], N_test);
	}


	printf("------------------------------\n");
	printf(" Run to calculate feature distributions\n");
	printf("------------------------------\n");

	
	printf("How many non-zero features to choose from?\n");
	int nFeat_nonzero = 0;
	for (i=0; i<data_C; i++) {
		printf("N_train %lld  cdf_nzero[%d] %d\n", N_train, i, cdf_nzero[i]);
		
		if (cdf_nzero[i] < N_train)
			feat_indices[nFeat_nonzero++] = i;
	}
	
	for (i=0; i<nFeat_nonzero; i++)
		printf("feat_indices[%d] %d\n", i, feat_indices[i]);
	//printf("enter");
	//fgetc(stdin);
//	printf("Make an array of indices\n");
//	for (i=0; i<data_C; i++)
//		feat_indices[i] = i;

	printf("----------------------\n");
	for (r=0; r<nRuns; r++) {
		printf("Select features without replacement\n");
		feat_N = nFeat;
		for (i=0; i<nFeat; i++) {
			j = i + rand()%(nFeat_nonzero-i);             // pick random index >= i
			int temp = feat_indices[i];           // swap indices
			feat_indices[i] = feat_indices[j];
			feat_indices[j] = temp;
		}
		
		printf("Sort the features\n");
		for (i=0; i<nFeat-1; i++) {
		for (j=i+1; j<nFeat; j++) {
			if (feat_indices[j] < feat_indices[i]) {
				int temp = feat_indices[j];
				feat_indices[j] = feat_indices[i];
				feat_indices[i] = temp;
		}}}
		
		
		printf("picked");
		for (i=0; i<nFeat; i++)
			printf(" %d", feat_indices[i]);
		printf("\n");
		
		
		printf("Run the analysis\n");
		Run();
	}

	printf("----------------------\n");
	printf(" Unmap the dataset features\n");
	printf("----------------------\n");
	UnmapBinfile(datafile);
	UnmapBinfile(testfile);
	
	printf("Free memory\n");
	for (i=0; i<nFeat; i++) {
		free(U_train[i]);
		free(U_test[i]);
	}
	
	printf("close the cross entropy file %s\n", run_outloss);
	fclose(run_floss);
}

int main()
{
	char *datasets[4] = {"cifar100", "mnist", "cifar10", "imagenette2"};
	char *layers[3][6] = {{"X", "A", "B", "C", "D"}, {"X", "A", "B", "C", "D"}, {"A", "B", "C", "D", "E"}};
	char *models[3] = {"resnet18", "resnet50", "vgg19"};
	//char *datasets[1] = {"cifar10"};
	//char *layers[1][2] = {{"C", "D"}};
	//char *models[3] = {"resnet50"};
	int Fs[] = {4, 3, 2, 5, 6};

	int L, M, D, iF;
	for (iF=0; iF<4; iF++) {
		int F = Fs[iF];
		for (D=0; D<4; D++) {
			for (M=0; M<3; M++) {
				for (L=0; L<5; L++) {
				//for (L=0; L<2; L++) {
					char *dataset = datasets[D];
					char *layer = layers[M][L];
					char *model = models[M];
					int seed = 0;
					printf("dataset %s layer %s model %s\n", dataset, layer, model);
					
					RunAll(dataset, model, seed,
						layer, F, 11, 10, 30);
				}
			}
		}
	}

//	RunAll("imagenette2", "vgg19", 0,
//		"C", 5, 11, 10, 1);

		
	return 0;
}


