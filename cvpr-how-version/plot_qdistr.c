#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qdistr.h"
#include "dc_csv.h"

char path[4096];
char cmd[4096];

//----------------
// Command Arguments
//----------------
char *arg_dataset;
char *arg_model;
int   arg_seed;
char *arg_smetric;
int   arg_metric;
int   arg_nquant;
double arg_start_quantile = 0.0;
int   arg_includezero = 0;


//----------------
// Basic Functions
//----------------
int System(char *str) {
	
	#ifdef _WIN32
	int i;
	for (i=0; str[i]; i++)
		if (str[i]=='/')
			str[i]='\\';
	#endif
	
	int rt=system(str);
	if (rt/256 != 0) {
		printf("WARNING command returned error: %d\n", rt/256);
		printf("%s\n", str);
	}

	#ifdef _WIN32
	for (i=0; str[i]; i++)
		if (str[i]=='\\')
			str[i]='/';
	#endif

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
	int i,j,y,x,d;

	//-----------------------------
	// Read Command Arguments
	//-----------------------------
	if (argc<6) {
		printf("Usage:\n");
		printf("\n");
		printf("   ./fit_distr dataset model seed metric nquant [start_quantile includezero]\n");
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


	printf("------------------------------------------------\n");
	printf(" Create all of the histograms\n");
	printf("------------------------------------------------\n");
	char *prefix[] = {"X", "A", "B", "C", "D", "E", NULL};
	char *train_test[] = {"train", "test", NULL};
	
	for (i=0; prefix[i]; i++) {
	for (j=0; train_test[j]; j++) {
		
		char series[256];
		sprintf(series, "%s_%s", prefix[i], train_test[j]);
		
		//
		// Read the distribution
		//
		QDistr qd = ReadQDistr(arg_dataset, arg_model, arg_seed, arg_nquant, series, arg_includezero, arg_start_quantile);
		
		//
		// Read the parameters
		//
		int param_csv_rows[N_DISTRIB_TYPES];
		int param_csv_cols[N_DISTRIB_TYPES];
		char ***param_csv[N_DISTRIB_TYPES];
		for (d=0; d<N_DISTRIB_TYPES; d++) {
			sprintf(path, "qdistribution/%s_%s_%d_%d_%.3f_%d/param/%s_%s.csv", arg_dataset, arg_model, arg_seed, arg_nquant, arg_start_quantile, arg_includezero, prefix[i], distr_name[d]);
			param_csv[d] = CsvReadDefault(path, &(param_csv_rows[d]), &(param_csv_cols[d]));
			param_csv_rows[d]--;
			param_csv_cols[d]--;
		}
		
		double *Y_distr = (double*)malloc(qd.nBins*sizeof(double));
		
		sprintf(cmd, "mkdir qdistribution/%s_%s_%d_%d_%.3f_%d/plot", arg_dataset, arg_model, arg_seed, arg_nquant, arg_start_quantile, arg_includezero);
		System(cmd);
		
		//
		// Plot the series
		//
		for (y=0; y<qd.nFeat; y++)
		{
			int x0 = qd.istart_quantile[y];
			
			sprintf(path, "qdistribution/%s_%s_%d_%d_%.3f_%d/plot/%s_%s_%05d.csv", arg_dataset, arg_model, arg_seed, arg_nquant, arg_start_quantile, arg_includezero, prefix[i], train_test[j], y);
			
			FILE *f = fopen(path, "wb");
			if (f==NULL) {
				printf("ERROR, cannot open %s for writing\n",path);
				exit(1);
			}
			
			// Write out the ticks
			fprintf(f, "X\t");
			for (x=x0; x<qd.nBins; x++)
				fprintf(f, "%f\t", 0.5*(qd.X[y][x]+qd.X[y][x+1]));
			fprintf(f, "\n");
			
			// Write out the series data
			fprintf(f, "%s\t", train_test[j]);
			for (x=x0; x<qd.nBins; x++)
				fprintf(f, "%f\t", qd.Y[y][x]);
			fprintf(f, "\n");
						
			// Write out the relevant distributions
			for (d=0; d<N_DISTRIB_TYPES; d++) {
				double param1 = atof(param_csv[d][y+1][0]);
				double param2 = atof(param_csv[d][y+1][1]);
				
				PlotDistribution(d, param1, param2, qd.X[y], Y_distr, x0, qd.nBins);
				
				fprintf(f, "%s\t", distr_name[d]);
				for (x=x0; x<qd.nBins; x++)
					fprintf(f, "%f\t", Y_distr[x]);
				fprintf(f, "\n");
			}
			fclose(f);

			sprintf(cmd, "python3 plot_qdistr.py %s", path);
			System(cmd);
		}
		
	}}


	return 0;
}
