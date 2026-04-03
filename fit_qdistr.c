#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "qdistr.h"




char strbuf[4000];

//----------------
// Command Arguments
//----------------
char *arg_dataset;
char *arg_model;
float arg_fmin;
int   arg_seed;
char *arg_smetric;
int   arg_metric;
int   arg_nquant;
double arg_start_quantile = 0.0;
int   arg_includezero = 0;

//----------------
// Directories
//----------------
char quant_dir[1000];
char qdist_dir[1000];

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

static void input() {
	printf("Press enter to continue\n");
	fgetc(stdin);
}

int IsTrue(char *str) {
	// Is it true ?
	if (strcmp(str,"true")==0 || strcmp(str,"True")==0)
		return 1;
	else if (strcmp(str,"false")==0 || strcmp(str,"False")==0)
		return 0;
	else {
		printf("ERROR str %s should be True or False (or true or false)\n", str);
		exit(1);
	}
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
	if (argc<6) {
		printf("Usage:\n");
		printf("\n");
		printf("   ./fit_qdistr dataset model seed metric nquant [start_quantile includezero]\n");
		printf("\n");
		printf("   metric           must be kl or was  (default kl)\n");
		printf("   start_quantile   (default=0) starting quantile (between 0 and nquant) for high-threshold comparison\n");
		exit(1);
	}
	arg_dataset =      argv[1];
	arg_model   =      argv[2];
	arg_seed    = atoi(argv[3]);
	arg_smetric = argv[4];
	
	if (strcmp(arg_smetric,"kl")==0)
		arg_metric = QD_METRIC_KL;
	else if (strcmp(arg_smetric,"was")==0)
		arg_metric = QD_METRIC_WAS;
	else {
		printf(" error unknown metric %s\n", arg_smetric);
		exit(1);
	}
	
	arg_nquant  = atoi(argv[5]);
	if (argc>6)
		arg_start_quantile = atof(argv[6]);
	
	if (argc>7)
		arg_includezero = IsTrue(argv[7]);
	

	printf("arg_dataset %s\n", arg_dataset);
	printf("arg_model %s\n", arg_model);
	printf("arg_seed %d\n", arg_seed);
	printf("arg_nquant %d\n", arg_nquant);
	printf("arg_metric %d %s\n", arg_metric, arg_smetric);
	printf("arg_start_quantile %f\n", arg_start_quantile);
	printf("arg_includezero %d\n", arg_includezero);


	printf("--------------------------\n");
	printf(" Create the directories\n");
	printf("--------------------------\n");

	sprintf(quant_dir, "quantiles/%s_%s_%d_%d", arg_dataset, arg_model, arg_seed, arg_nquant);
	sprintf(qdist_dir, "qdistribution/%s_%s_%d_%d_%.3f_%d", arg_dataset, arg_model, arg_seed, arg_nquant, arg_start_quantile, arg_includezero);

	System("mkdir qdistribution");
	sprintf(strbuf, "mkdir %s", qdist_dir);
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
		printf("-------------------------------------\n");
		printf(" Prefix %s\n", prefix[i]);
		printf("-------------------------------------\n");
		
		
		printf("------------------\n");
		printf(" Read the train/test CSV histograms\n");
		printf("------------------\n");

		char series_train[256];
		char series_test[256];
		sprintf(series_train, "%s_train", prefix[i]);
		sprintf(series_test, "%s_test", prefix[i]);

		QDistr train = ReadQDistr(arg_dataset, arg_model, arg_seed, arg_nquant, series_train, arg_includezero, arg_start_quantile);
		if (train.X == NULL)
			continue;
		
		QDistr test = ReadQDistr(arg_dataset, arg_model, arg_seed, arg_nquant, series_train, arg_includezero, arg_start_quantile);
		if (test.X == NULL)
			continue;
		
		// Double check shape (train/test)
		if (train.nFeat!=test.nFeat) {
			printf("ERROR: train nFeat %d differs from test nFeat %d\n",
				train.nFeat, test.nFeat);
			exit(1);
		}

		// How many features
		int nFeat = train.nFeat;

		printf("nFeat %d train.nBins %d  test.nBins %d\n", nFeat, train.nBins, test.nBins);

		if (train.nBins<1) {
			printf("ERROR: number of training bins is less than 1\n");
			exit(1);
		}
		if (test.nBins<1) {
			printf("ERROR: number of testing bins is less than 1\n");
			exit(1);
		}
		
		// Allocate the feature histograms (using midpoint rule)
		double *yhat_train = (double*)malloc(train.nBins     * sizeof(double));
		double *yhat_test  = (double*)malloc(test.nBins      * sizeof(double));
		double *cdf_yhat_train = (double*)malloc(train.nBins     * sizeof(double));
		double *cdf_yhat_test  = (double*)malloc(test.nBins      * sizeof(double));		
		double *log_yhat_train = (double*)malloc(train.nBins     * sizeof(double));
		double *log_yhat_test  = (double*)malloc(test.nBins      * sizeof(double));
	

		printf("------------------\n");
		printf(" Open the output loss table\n");
		printf("------------------\n");

		// Open the table
		char floss_path[4096];
		sprintf(floss_path, "%s/%s_loss.csv", qdist_dir, prefix[i]);
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


		printf("------------------\n");
		printf(" Open the output distribution parameters\n");
		printf("------------------\n");

		// Make the parameter directory
		char param_mkdir[2000];
		sprintf(param_mkdir, "mkdir %s/param", qdist_dir);
		System(param_mkdir);

		// Create the output files
		FILE *fparam[N_DISTRIB_TYPES];
		for (d=0; d<N_DISTRIB_TYPES; d++) {

			// Output parameter file
			char param_path[4096];
			sprintf(param_path, "%s/param/%s_%s.csv", qdist_dir, prefix[i], distr_name[d]);
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
			printf("--------\n");
			printf("feature %d of %d\n", y, nFeat);
			printf("--------\n");

		
			//------------------------------------
			// Fit the training distribution

			// The train/test entropy and loss
			double train_entropy;
			double test_entropy;
			double train_cross_entropy[N_DISTRIB_TYPES];
			double test_cross_entropy[N_DISTRIB_TYPES];
			double train_kl[N_DISTRIB_TYPES];
			double test_kl[N_DISTRIB_TYPES];


			// Fit the distribution
			for (d=0; d<N_DISTRIB_TYPES; d++) {
				//printf("--------\n");
				//printf("feature %d of %d   distrib %d\n", y, nFeat, d);
				//printf("--------\n");
				//if (y==321 && d==4) {
				//	printf("set qd_test !!!!\n");
				//	input();
				//	qd_test=1;
				//}

				// Record the entropy and loss
				if (d==0) {
					train_entropy          = CrossEntropy(train.X[y], train.Y[y], train.Y[y], train.istart_quantile[y], train.nBins);
					test_entropy           = CrossEntropy(test.X[y], test.Y[y],  test.Y[y], test.istart_quantile[y], test.nBins);

				}
				// Fit the distribution
				Distribution distr = FitDistribution(d, arg_metric, train.X[y], train.Y[y], train.Y_cdf[y], train.istart_quantile[y], train.nBins, log_yhat_train, cdf_yhat_train);

				// Plot the distribution
				PlotDistribution(d, distr.param1, distr.param2, train.X[y], yhat_train, 0, train.nBins);    // train plot
				PlotDistribution(d, distr.param1, distr.param2, test.X[y],  yhat_test, 0,  test.nBins);    // train plot
				PlotLogDistribution(d, distr.param1, distr.param2, train.X[y], log_yhat_train, 0, train.nBins);    // train plot
				PlotLogDistribution(d, distr.param1, distr.param2, test.X[y],  log_yhat_test, 0,  test.nBins);    // train plot
				PlotCdfDistribution(d, distr.param1, distr.param2, train.X[y], cdf_yhat_train, 0, train.nBins);    // train plot
				PlotCdfDistribution(d, distr.param1, distr.param2, test.X[y],  cdf_yhat_test, 0,  test.nBins);    // train plot

				// Write out the distribution fit
				//fprintf(fodist, "%d\t%s\t", y, distr_name[d]);
				//for (x=0; x<nBins; x++)
				//	fprintf(fodist, "%f\t", yhat_train[x]);
				//fprintf(fodist,"\n");

				train_cross_entropy[d] = LogCrossEntropy(train.X[y], train.Y[y], log_yhat_train, train.istart_quantile[y], train.nBins);
				test_cross_entropy[d]  = LogCrossEntropy(test.X[y], test.Y[y],  log_yhat_test, test.istart_quantile[y], test.nBins);
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

		}


		// Free CSV memory
		QDistrFree(train);
		QDistrFree(test);
		free(yhat_train);
		free(yhat_test);
		free(cdf_yhat_train);
		free(cdf_yhat_test);
		free(log_yhat_train);
		free(log_yhat_test);

		// Close the loss table
		fclose(floss);

		// Close the parameter files
		for (d=0; d<N_DISTRIB_TYPES; d++)
			fclose(fparam[d]);
	}

	printf("Done!\n");

	return 0;
}


