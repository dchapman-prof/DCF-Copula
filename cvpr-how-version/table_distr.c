#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dc_csv.h"

//#define isnan(x)  ( (x)!=(x) )
#define MIN(a,b)  ( (a)<(b) ? (a) : (b) )
#define MAX(a,b)  ( (a)>(b) ? (a) : (b) )



//
// Command arguments
//
char *arg_dataset;
char *arg_model;
int   arg_seed;

char dist_dir[4096];
char table_path[4196];

int main(int argc, char**argv)
{
	int i;

	//------------------------
	// Read command arguments
	//------------------------
	if (argc<4) {
		printf("Usage:\n");
		printf("\n");
		printf("   ./table_distr dataset model seed\n");
		printf("\n");
		printf("dataset:   cifar10 cifar100 mnist or imagenette2\n");
		printf("model:     resnet18 resnet50 vgg19\n");
		printf("seed:      start with 0 and go from there\n");
		exit(1);
	}
	arg_dataset =      argv[1];
	arg_model   =      argv[2];
	arg_seed    = atoi(argv[3]);

	printf("arg_dataset: %s\n", arg_dataset);
	printf("arg_model: %s\n",   arg_model);
	printf("arg_seed: %d\n",    arg_seed);


	sprintf(dist_dir, "distribution/%s_%s_%d", arg_dataset, arg_model, arg_seed);
	sprintf(table_path, "%s/table.csv", dist_dir);

	//---------
	// Read the loss files
	//---------
	//char *prefixes[] = {"X", "A", "B", "C", "D", "E", 0};
	char *prefixes[] = {"A", 0};
	for (i=0; prefixes[i]; i++) {
		char *prefix = prefixes[i];
		
		printf("-----------------\n");
		printf(" Read the loss CSV\n");
		printf("-----------------\n");
		char loss_path[4296];
		sprintf(loss_path, "%s/%s_loss.csv", dist_dir, prefix);

		printf("%s\n", loss_path);		
		printf("press enter\n");
		fgetc(stdin);

		int y,x;
		int rows,cols;
		//char***csv = CsvReadDefault(loss_path, &rows, &cols);
		char***csv = CsvRead(loss_path, &rows, &cols, "\t", "\n", "\r", 0);
		
		printf("rows %d cols %d csv %p\n", rows, cols, csv);
		if (csv==NULL) {
			printf("WARNING: cannot open %s for reading\n", loss_path);
			continue;
		}
		
		int nFeat = rows-2;
		int nField = cols-2;
		
		printf("nFeat: %d\n", nFeat);
		printf("nField: %d\n", nField);
		
		printf("-----------------\n");
		printf(" Copy data to an array\n");
		printf("-----------------\n");
		
		// Allocate the data array
		double **data = (double**)malloc(nFeat * sizeof(double*));
		for (y=0; y<nFeat; y++)
			data[y] = (double*)malloc(nField * sizeof(double));
		
		// Fill in the data
		for (y=0; y<nFeat; y++) {
			for (x=0; x<nField; x++) {
				char *csv_entry = csv[y+1][x+1];
				double val = atof(csv_entry);
				data[y][x] = val;
				printf("y %d x %d csv %s data %f  isnan %d isinf(%d)\n", y, x, csv_entry, val, isnan(val), isinf(val));
				
				
			}
//			printf("press enter\n");
//			fgetc(stdin);
		}
		
		// Fill the field names
		
		
		printf("-----------------\n");
		printf(" Calculate Mean, Mini, Maxi \n");
		printf("-----------------\n");
		
		// Calculate mean, mini, maxi
		double *mean  = (double*)calloc(nField, sizeof(double));
		double *mini  = (double*)calloc(nField, sizeof(double));
		double *maxi  = (double*)calloc(nField, sizeof(double));
		for (x=0; x<nField; x++) {
			mini[x] =  999999.0;
			maxi[x] = -999999.0;
		}

		int count = 0;


		for (y=0; y<nFeat; y++) {
		
			// Is this an all zero row ?
			int isAllZero = 1;
			for (x=0; x<nField; x++) {
				if (data[y][x] != 0) {
					isAllZero = 0;
					break;
				}
			}
			
			// Is this a NaN or Inf row ?
			int isNanInf = 0;
			for (x=0; x<nField; x++) {
				if (isnan(data[y][x]) || isinf(data[y][x]) || data[y][x]<0.0) {
					isNanInf = 1;
					break;
				}
			}
			
			// Is this a bad row ?
			if (isNanInf || isAllZero) {
				printf("WARNING: skipping feature %d\n", y);
				continue;
			}
			
			// Update mean min and max
			for (x=0; x<nField; x++) {
				mean[x] += data[y][x];
				mini[x] = MIN(mini[x], data[y][x]);
				maxi[x] = MAX(maxi[x], data[y][x]);
				count++;
			}
		}
		for (x=0; x<nField; x++)
			mean[x] /= count;
		
		
		
		printf("-----------------\n");
		printf(" Calculate Stdev \n");
		printf("-----------------\n");
		double *stdev = (double*)calloc(nField, sizeof(double));
		count = 0;

		for (y=0; y<nFeat; y++) {
		
			// Is this an all zero row ?
			int isAllZero = 1;
			for (x=0; x<nField; x++) {
				if (data[y][x] != 0) {
					isAllZero = 0;
					break;
				}
			}
			
			// Is this a NaN or Inf row ?
			int isNanInf = 0;
			for (x=0; x<nField; x++) {
				if (isnan(data[y][x]) || isinf(data[y][x]) || data[y][x]<0.0) {
					isNanInf = 1;
					break;
				}
			}
			
			// Is this a bad row ?
			if (isNanInf || isAllZero) {
				printf("WARNING: skipping feature %d\n", y);
				continue;
			}
			
			// Update mean min and max
			for (x=0; x<nField; x++) {
				stdev[x] += (data[y][x] - mean[x]) * (data[y][x] - mean[x]);
				count++;
			}
		}
		for (x=0; x<nField; x++)
			stdev[x] = sqrt(stdev[x] / count);
		
		
		// Debug, print out statistics
		for (x=0; x<nField; x++) {
			printf("field %d mean %f stdev %f mini %f maxi %f count %d\n", x, mean[x], stdev[x], mini[x], maxi[x], count);
		}
		
		
		printf("press enter\n");
		fgetc(stdin);
		
		
		// Free memory
		for (y=0; y<nFeat; y++)
			free(data[y]);
		free(data);
		CsvFree(csv, rows, cols);
		free(mean);
		free(mini);
		free(maxi);
		free(stdev);
	}

	return 0;
}


