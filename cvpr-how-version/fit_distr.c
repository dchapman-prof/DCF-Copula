#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "dc_csv.h"

#define i64 long long int
#define ABS(x)   ( (x)<=0 ? 0-(x) : (x) )
#define RANDF ((double)rand() / (double)RAND_MAX)
#define RANDBALF (2.0f*RANDF - 1.0f)
#define PI    3.14159265358979323846264
#define SQRT2 1.41421356237309504880168

int mydebug=0;

char strbuf[4000];

//----------------
// Command Arguments
//----------------
char *arg_dataset;
char *arg_model;
float arg_fmin;
int   arg_seed;

//----------------
// Directories
//----------------
char hist_dir[1000];
char dist_dir[1000];

//----------------
// Basic Functions
//----------------
int System(char *str) {
	int rt=system(str);
	if (rt/256 != 0) {
		printf("WARNING command returned error: %d\n", rt/256);
		printf("%s\n", str);
	}
	return rt/256;
}

void input() {
	printf("Press enter to continue\n");
	fgetc(stdin);
}

//----------------
// Probability distributions
//----------------
typedef struct Distribution
{
	int type;       // UNIFORM    GAUSSIAN   EXPONENTIAL   GAMMA

	// Parameters   // UNIFORM    GAUSSIAN   EXPONENTIAL   GAMMA
	double param1;  //    mean       mean        lamda      alpha
	double param2;  // halfwidth     stdev        N/A       beta

	// Goodness of fit loss
	double cross_entropy;
	double entropy;
	double kl_diver;
} Distribution;
#define DISTRIB_UNIFORM      0
#define DISTRIB_GAUSSIAN     1
#define DISTRIB_EXPONENTIAL  2
#define DISTRIB_GAMMA        3
#define DISTRIB_WEIBULL      4
#define DISTRIB_TRUNC_NORMAL 5
#define N_DISTRIB_TYPES      5      // Remove truncated normal

const char *distr_name[N_DISTRIB_TYPES]     = {"uniform",   "gaussian", "exponential", "gamma",   "weibull", "trunc_normal"};
const char *distr_p1_name[N_DISTRIB_TYPES]  = {"mu",        "mu",       "lamda",       "alpha",   "lamda",   "mu"};
const char *distr_p2_name[N_DISTRIB_TYPES]  = {"halfwidth", "stdev",    "none",        "beta",    "k",       "stdev"};

void BoundsDistribution(int type, double *p_param1, double *p_param2)
{
	switch (type) {
		case DISTRIB_UNIFORM:
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // halfwidth must be positive
			break;
		case DISTRIB_GAUSSIAN:
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // stdev must be positive
			break;
		case DISTRIB_EXPONENTIAL:
			if (*p_param1 < 0.0000001)
				*p_param1 = ABS(*p_param1) + 0.0000001;  // lamda must be positive
			break;
		case DISTRIB_GAMMA:
			if (*p_param1 < 0.0000001)
				*p_param1 = ABS(*p_param1) + 0.0000001;  // alpha must be positive
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // beta must be positive
			break;
		case DISTRIB_WEIBULL:
			if (*p_param1 < 0.0000001)
				*p_param1 = ABS(*p_param1) + 0.0000001;  // lamda must be positive
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // k must be positive
			break;
		case DISTRIB_TRUNC_NORMAL:
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // stdev must be positive
			break;
	}
}


void PlotDistribution(int type, double param1, double param2, double *X, double *Y, int nBins) {
	int i;

	// premature optimization
	double unif_mu = param1;
	double unif_hw = param2;
	double unif_x0 = unif_mu - unif_hw;
	double unif_x1 = unif_mu + unif_hw;
	double unif_y  = 0.5 / unif_hw;
	double norm_mu = param1;
	double norm_sig = param2;
	double norm_coef = 1.0/sqrt(2.0*PI*norm_sig*norm_sig);
	double norm_scale = -1.0/(2.0*norm_sig*norm_sig);
	double exp_lamda = param1;
	double gam_alpha = param1;
	double gam_beta  = param2;
	double gam_coef  = pow(gam_beta, gam_alpha) / tgamma(gam_alpha);
	double wei_lam   = param1;
	double wei_k     = param2;
	double trunc_norm_F0 = 0.5 * (1.0 + erf(-norm_mu / (norm_sig*SQRT2)));
	double trunc_norm_coef = 1.0 / (1.0 - trunc_norm_F0);

	// Plot the PDF
	double mass=0.0;
	double step=X[1]-X[0];
	for (i=0; i<nBins; i++) {
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (X[i]<unif_x0 || X[i]>unif_x1)
				Y[i] = 0.0;
			else
				Y[i] = unif_y;
			break;

		case DISTRIB_GAUSSIAN:
				Y[i] = norm_coef * exp(norm_scale*(X[i]-norm_mu)*(X[i]-norm_mu));
			break;

		case DISTRIB_EXPONENTIAL:
			if (X[i]<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = exp_lamda * exp(-exp_lamda * X[i]);
			break;

		case DISTRIB_GAMMA:
			if (X[i]<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = gam_coef * pow(X[i], gam_alpha-1.0) * exp(-gam_beta * X[i]);
			break;

		case DISTRIB_WEIBULL:
			if (X[i]<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = (wei_k / wei_lam) * pow((X[i]/wei_lam),(wei_k-1)) * exp(-pow((X[i]/wei_lam),(wei_k)));
			break;

		case DISTRIB_TRUNC_NORMAL:
			if (X[i]<0.0)
				Y[i] = 0.0;
			else
				Y[i] = trunc_norm_coef * norm_coef * exp(norm_scale*(X[i]-norm_mu)*(X[i]-norm_mu));
			break;
		}
		//printf("X %f Y %f\n", X[i], Y[i]);
	}
	//input();
}

//
// Previous version, might cause issues with negative KL divergence
//
/*
double CrossEntropy(double *X, double *Y, double *Yhat, int nBins) {
	int i;
	double step = X[1]-X[0];
	double entropy = 0.0;
	for (i=0; i<nBins; i++) {
		if (Y[i]>0.0000001)
			entropy -= step * Y[i] * log2(Yhat[i]+0.0000001);
	}
	return entropy;
}
*/

//
// Now we normalize to ensure the KL-divergence is non-negative
//
double CrossEntropy(double *X, double *Y, double *Yhat, int nBins)
{
//printf("BEGIN CrossEntropy\n");

	// Get the scaling factors for Y and Yhat
	int i;
	double step = X[1]-X[0];
	double totalY = 0.0;
	double totalYhat = 0.0;
	for (i=0; i<nBins; i++) {
		totalY += Y[i];
		totalYhat += Yhat[i];
	}
	double scaleY = 1.0 / totalY;
	double scaleYhat = 1.0 / totalYhat;

//printf("scaleY %f  scaleYhat %f\n", scaleY, scaleYhat);

	// Calculate the cross entropy
	double entropy = 0.0;
	for (i=0; i<nBins; i++) {
		double y    = (Y[i] * scaleY);
		double yhat = (Yhat[i] * scaleYhat);
//printf("y %.16f yhat %.16f\n", y, yhat);
		if (y>0.0000001)
			entropy -= y * log2(yhat + 0.0000001);
	}

//printf("END CrossEntropy\n");
//fgetc(stdin);
	return entropy;
}

typedef struct InitialGuess {
	double param1;        // Guess for parameters 1 and 2
	double param2;
	double param1_step;   // Step-size for simulated annealing
	double param2_step;
} InitialGuess;

InitialGuess InitialGuessDistribution(int type, double *X, double *Y, int N)
{
	int i;

	// Calculate the step size
	double step = X[1]-X[0];

	// Calculate the mean
	double total = 0.0;
	double count = 0.0;
	for (i=0; i<N; i++) {
		total += X[i]*Y[i];
		count += Y[i];
	}
	double mean = total / count;
if(mydebug)printf("mean %f\n", mean);

	// Calculate the variance
	total = 0.0;
	count = 0.0;
	for (i=0; i<N; i++) {
		total += (X[i]-mean)*(X[i]-mean)*Y[i];
		count += Y[i];
	}
	double variance = total / count;
	
	variance += step*step; // (incorporate contribution of step size to variance
if(mydebug)printf("variance %f\n", variance);

	// Calculate standard deviation
	double stdev = sqrt(variance);
if(mydebug)printf("stdev %f\n", stdev);

	// Estimate parameters
	double param1;  //    mean       mean        lamda      alpha
	double param2;  // halfwidth     stdev        N/A       beta
	InitialGuess guess;
	switch(type) {
	case DISTRIB_UNIFORM:
		guess.param1 = mean;    // mean
		guess.param2 = stdev;   // halfwidth
		guess.param1_step = ABS(mean) + stdev;
		guess.param2_step = stdev;
		break;
	case DISTRIB_GAUSSIAN:
		guess.param1 = mean;    // mean
		guess.param2 = stdev;   // stdev
		guess.param1_step = ABS(mean) + stdev;
		guess.param2_step = stdev;
		break;
	case DISTRIB_EXPONENTIAL:
		guess.param1 = 1.0 / mean;   // lamda
		guess.param2 = 0.0;
		guess.param1_step = guess.param1;
		guess.param2_step = 0.0;
		break;
	case DISTRIB_GAMMA:
		guess.param2 = mean / variance;      // beta
		guess.param1 = mean * guess.param2;  // alpha
		guess.param1_step = guess.param1;
		guess.param2_step = guess.param2;
		break;

	case DISTRIB_WEIBULL:              // uninformed guess
		guess.param1 = 1.0;
		guess.param2 = 1.0;
		guess.param1_step = 1.0;
		guess.param2_step = 1.0;
		break;

	case DISTRIB_TRUNC_NORMAL:              // uninformed guess
		guess.param1 = 0.0;
		guess.param2 = 1.0;
		guess.param1_step = 1.0;
		guess.param2_step = 1.0;
		break;
	}
	return guess;
}


Distribution FitDistribution(int type, double *X, double *Y, int N, double *Y_back)
{
	int i,iter;
	Distribution distr;
	distr.param1 = -9999.0;
	distr.param2 = -9999.0;
	distr.cross_entropy = -9999.0;
	distr.entropy       = -9999.0;
	distr.kl_diver      = -9999.0;

	// Initial guess . . .
	InitialGuess guess = InitialGuessDistribution(type, X, Y, N);
	BoundsDistribution(type, &guess.param1, &guess.param2);
	double guess_param1 = guess.param1;
	double guess_param2 = guess.param2;
	double param1_step  = guess.param1_step;
	double param2_step  = guess.param2_step;


	// Assert that mean is positive   (ToDo: This is only a problem for Gamma and Exponential
//	if (mean<0.00001) {
//		printf("WARNING: Cannot fit distribution, mean %f is not sufficiently positive", mean);
//		return distr;
//	}

	// Calculate Entropy
	double entropy = CrossEntropy(X, Y, Y, N);

	// Fit using simulated annealing
	double param1 = guess_param1;
	double param2 = guess_param2;
	PlotDistribution(type, param1, param2, X, Y_back, N);
	double cross_entropy = CrossEntropy(X, Y, Y_back, N);
	printf(">   guess   %s %s %f %s %f loss %f\n", distr_name[type], distr_p1_name[type], param1, distr_p2_name[type], param2, cross_entropy);
	//input();
	for (iter=0; iter<500; iter++) {
		double new_param1 = param1 + RANDBALF * param1_step;
		double new_param2 = param2 + RANDBALF * param2_step;

		BoundsDistribution(type, &new_param1, &new_param2);

		PlotDistribution(type, new_param1, new_param2, X, Y_back, N);
		double new_cross_entropy = CrossEntropy(X, Y, Y_back, N);
		if (new_cross_entropy < cross_entropy && !isnan(new_cross_entropy) && !isinf(new_cross_entropy)) {
			param1 = new_param1;
			param2 = new_param2;
			cross_entropy = new_cross_entropy;
			if (mydebug)
				printf(">  new %s %s %f %s %f loss %f\n", distr_name[type], distr_p1_name[type], param1,distr_p2_name[type], param2, cross_entropy);
		}

		param1_step *= 0.97;   // falloff
		param2_step *= 0.97;   // falloff
	}


	distr.param1 = param1;
	distr.param2 = param2;
	distr.entropy = entropy;
	distr.cross_entropy = cross_entropy;
	distr.kl_diver = cross_entropy - entropy;
	printf(">  solution  %s %s %f %s %f cross %f entro %f kl %f\n",
		distr_name[type], distr_p1_name[type], param1,
		 distr_p2_name[type], param2,
		 cross_entropy, entropy, distr.kl_diver);
	if(mydebug)input();

	return distr;
}


//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
	int i,y,x,d;

	//-----------------------------
	// Read Command Arguments
	//-----------------------------
	if (argc<4) {
		printf("Usage:\n");
		printf("\n");
		printf("   ./fit_distr dataset model seed\n");
		exit(1);
	}
	arg_dataset =      argv[1];
	arg_model   =      argv[2];
	arg_seed    = atoi(argv[3]);

	printf("arg_dataset %s\n", arg_dataset);
	printf("arg_model %s\n", arg_model);
	printf("arg_seed %d\n", arg_seed);



	printf("--------------------------\n");
	printf(" Create the directories\n");
	printf("--------------------------\n");

	sprintf(hist_dir, "histogram/%s_%s_%d", arg_dataset, arg_model, arg_seed);
	sprintf(dist_dir, "distribution/%s_%s_%d", arg_dataset, arg_model, arg_seed);

	System("mkdir distribution");
	sprintf(strbuf, "mkdir %s", dist_dir);
	System(strbuf);


	printf("------------------------------------------------\n");
	printf(" Create all of the histograms\n");
	printf("------------------------------------------------\n");
	char *prefix[] = {"X", "A", "B", "C", "D", "E", NULL};

	//------------------------------------
	// For every file
	//------------------------------------
	for (i=0; prefix[i]; i++)
	{

		//------------------------------------
		// Read the train/test CSV histograms
		//------------------------------------

		// Where is the CSV file ?
		char train_path[4096];
		char  test_path[4096];
		sprintf(train_path, "%s/%s_train.csv", hist_dir, prefix[i]);
		sprintf( test_path, "%s/%s_test.csv", hist_dir, prefix[i]);

		// Read the CSV file (train)
		int train_rows,train_cols;
		char ***train_csv = CsvReadDefault(train_path, &train_rows, &train_cols);
		if (train_csv==NULL) {
			printf("WARNING could not read %s skipping\n", train_path);
			continue;
		}

		// Read the CSV file (test)
		int test_rows,test_cols;
		char ***test_csv = CsvReadDefault(test_path, &test_rows, &test_cols);
		if (test_csv==NULL) {
			printf("WARNING could not read %s skipping\n", test_path);
			CsvFree(train_csv, train_rows, train_cols);
			continue;
		}

		//printf("train shape (%d %d)\n", train_rows, train_cols);
		//printf("test shape  (%d %d)\n", test_rows, test_cols);
		//input();

		// HACK  remove last row/column, because of \t\n problem (fix later)
		train_rows--;
		test_rows--;
		train_cols--;
		test_cols--;

		// Double check shape (train/test)
		if (train_rows!=test_rows || train_cols!=test_cols) {
			printf("ERROR: train shape (%d %d) differs from test shape (%d %d)\n",
				train_rows, train_cols, test_rows, test_cols);
			exit(1);
		}
		int rows = train_rows;
		int cols = test_cols;

		// How many features
		int nFeat = rows-1;
		int nBins = cols;

		//printf("nFeat %d nBins %d\n", nFeat, nBins);
		//input();

		if (nBins<2) {
			printf("ERROR: number of bins is less than 2\n");
			exit(1);
		}
		// Allocate the feature arrays
		double *ticks       = (double*)malloc(nBins * sizeof(double));   // allocate tick marks
		i64    *count_train =    (i64*)malloc(nBins * sizeof(i64));     // absolute histogram counts
		i64    *count_test  =    (i64*)malloc(nBins * sizeof(i64));     // absolute histogram counts
		double *pdf_train   = (double*)malloc(nBins * sizeof(double));   // probability density function (integral of 1)
		double *pdf_test    = (double*)malloc(nBins * sizeof(double));   // probability density function (integral of 1)
		double *pdf_distr   = (double*)malloc(nBins * sizeof(double));   // backup memory for out of place data 

		// Read the ticks
		for (x=0; x<nBins; x++) {
			ticks[x] = atof(train_csv[0][x]);
			printf("ticks[%d] %f\n", x, ticks[x]);
		}
		//input();


		// Parse the stepsize
		double step = ticks[1] - ticks[0];   // what's the step size ?

		//------------------------------------
		// Open the output empirical histograms
		//------------------------------------

		// Open csv for plotting
		char odist_path[4096];
		sprintf(odist_path, "%s/%s_histogram.csv", dist_dir, prefix[i]);
		FILE *fodist = fopen(odist_path, "w");
		if (fodist==NULL) {
			printf("ERROR: could not open %s for writing\n", odist_path);
			exit(1);
		}

		// Write the ticks
		fprintf(fodist, "feature\tdistribution\t");
		for (x=0; x<nBins; x++)
			fprintf(fodist, "%.4f\t", ticks[x]);
		fprintf(fodist, "\n");

		//------------------------------------
		// Open the output loss table
		//------------------------------------

		// Open the table
		char floss_path[4096];
		sprintf(floss_path, "%s/%s_loss.csv", dist_dir, prefix[i]);
		printf("%s\n", floss_path);
		FILE *floss = fopen(floss_path, "w");
		if (floss==NULL) {
			printf("ERROR: could not open %s for writing\n", floss_path);
			exit(1);
		}

		// Print the loss header
		fprintf(floss, "feature\t");
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fprintf(floss, "test_kl_%s\t", distr_name[d]);
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fprintf(floss, "train_kl_%s\t", distr_name[d]);
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fprintf(floss, "test_loss_%s\t", distr_name[d]);
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fprintf(floss, "train_loss_%s\t", distr_name[d]);
		fprintf(floss, "test_entropy\ttrain_entropy\t\n");
		fflush(floss);


		//------------------------------------
		// Open the output distribution parameters
		//------------------------------------

		// Make the parameter directory
		char param_mkdir[2000];
		sprintf(param_mkdir, "mkdir %s/param", dist_dir);
		System(param_mkdir);

		// Create the output files
		FILE *fparam[N_DISTRIB_TYPES];
		for (d=0; d<N_DISTRIB_TYPES; d++) {

			// Output parameter file
			char param_path[4096];
			sprintf(param_path, "%s/param/%s_%s.csv", dist_dir, prefix[i], distr_name[d]);
			fparam[d] = fopen(param_path, "w");
			if (fparam[d]==NULL) {
				printf("ERROR: could not open %s for writing\n", param_path);
				exit(1);
			}

			// Print header
			fprintf(fparam[d], "%s\t%s\t\n", distr_p1_name[d], distr_p2_name[d]);
			fflush(fparam[d]);
		}


		//------------------------------------
		// For every feature . . .
		//------------------------------------
		for (y=0; y<nFeat; y++) {
			printf("feature %d of %d\n", y, nFeat);

			//------------------------------------
			// Fit the training distribution
			//------------------------------------

			// Read the counts
			for (x=0; x<nBins; x++) {
				count_train[x] = atoll(train_csv[y+1][x]);
				count_test[x]  = atoll(test_csv[y+1][x]);
			}

			// Calculate the total mass and the area under the curve
			i64 mass_train = 0;
			i64 mass_test  = 0;
			for (x=0; x<nBins; x++) {
				mass_train += count_train[x];
				mass_test  += count_test[x];
			}
			double area_train = mass_train * step;       // area under curve is sum of mass times delta step size
			double area_test  = mass_test * step;
			double inv_area_train = 1.0 / area_train;
			double inv_area_test  = 1.0 / area_test;

			// Convert to probability density function (area of 1.0)
			for (x=0; x<nBins; x++) {
				pdf_train[x] = count_train[x] * inv_area_train;
				pdf_test[x]  = count_test[x]  * inv_area_test;
			}

			// Write out the train and test distributions
			fprintf(fodist,"%d\ttrain\t",y);
			for (x=0; x<nBins; x++)
				fprintf(fodist, "%f\t", pdf_train[x]);
			fprintf(fodist,"\n%d\ttest\t",y);
			for (x=0; x<nBins; x++)
				fprintf(fodist, "%f\t", pdf_test[x]);
			fprintf(fodist,"\n");
			fflush(fodist);

			// The train/test entropy and loss
			double train_entropy;
			double test_entropy;
			double train_cross_entropy[N_DISTRIB_TYPES];
			double test_cross_entropy[N_DISTRIB_TYPES];
			double train_kl[N_DISTRIB_TYPES];
			double test_kl[N_DISTRIB_TYPES];

//if (i==1 && y==91) {
//mydebug = 1;
//}
			// Fit the distribution
			for (d=0; d<N_DISTRIB_TYPES; d++) {

				// Record the entropy and loss
				if (d==0) {
					train_entropy          = CrossEntropy(ticks, pdf_train, pdf_train, nBins);
					test_entropy           = CrossEntropy(ticks, pdf_test,  pdf_test, nBins);
					
//printf("Reached entropy (press enter)\n");
//fgetc(stdin);
				}
				// Fit the distribution
				Distribution distr = FitDistribution(d, ticks, pdf_train, nBins, pdf_distr);

				// Plot the distribution
				PlotDistribution(d, distr.param1, distr.param2, ticks, pdf_distr, nBins);

				// Write out the distribution fit
				fprintf(fodist, "%d\t%s\t", y, distr_name[d]);
				for (x=0; x<nBins; x++)
					fprintf(fodist, "%f\t", pdf_distr[x]);
				fprintf(fodist,"\n");

				train_cross_entropy[d] = CrossEntropy(ticks, pdf_train, pdf_distr, nBins);
				test_cross_entropy[d]  = CrossEntropy(ticks, pdf_test,  pdf_distr, nBins);
				train_kl[d] = train_cross_entropy[d] - train_entropy;
				test_kl[d]  = test_cross_entropy[d]  - test_entropy;

				// Record the parameters
				fprintf(fparam[d], "%f\t%f\t\n", distr.param1, distr.param2);
			}

			// Print the losses
			fprintf(floss, "%d\t", y);  // feature number
			for (d=0; d<N_DISTRIB_TYPES; d++)
				fprintf(floss, "%f\t", test_kl[d]);
			for (d=0; d<N_DISTRIB_TYPES; d++)
				fprintf(floss, "%f\t", train_kl[d]);
			for (d=0; d<N_DISTRIB_TYPES; d++)
				fprintf(floss, "%f\t", test_cross_entropy[d]);
			for (d=0; d<N_DISTRIB_TYPES; d++)
				fprintf(floss, "%f\t", train_cross_entropy[d]);
			fprintf(floss, "%f\t%f\t\n", test_entropy, train_entropy);
			fflush(floss);
			
//if (i==1 && y==91) {
//for (d=0; d<N_DISTRIB_TYPES; d++) {
//	printf("d %d %s train_kl %f test_kl %f train_ce %f test_ce %f train_e %f test_e %f\n", d, distr_name[d], train_kl[d], test_kl[d], train_cross_entropy[d], test_cross_entropy[d], train_entropy, test_entropy);
//}
//printf("press enter\n");
//fgetc(stdin);
//}
		}


		// Free CSV memory
		CsvFree(train_csv,train_rows,train_cols);
		CsvFree(test_csv, test_rows, test_cols);
		free(ticks);
		free(count_train);
		free(count_test);
		free(pdf_train);
		free(pdf_test);
		free(pdf_distr);

		// Close output distribution histograms
		fclose(fodist);

		// Close the loss table
		fclose(floss);

		// Close the parameter files
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fclose(fparam[d]);
	}

	printf("Done!\n");

	return 0;
}


