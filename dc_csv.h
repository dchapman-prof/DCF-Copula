#ifndef _DC_CSV_H_
#define _DC_CSV_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// Typical Usage
//   int y,x;
//   int rows,cols;                       // returned by reference
//   char***csv = CsvReadDefault("myfile.csv", &rows, &cols);
//
//   for (y=0; y<rows; y++) {
//      for (x=0; x<cols; x++) {
//         printf("%s\t", csv[y][x]);
//      }
//      printf("\n");
//   }
//
//   CsvFree(csv,rows,cols);
//
//

static int debug = 0;

static char ***CsvRead(const char *fname, int *rows, int *cols, const char *col_delim, const char *row_delim, const char *strip_chars, int checkQuote)
{
	int i,j,y,x;
	
	// Open the file
	FILE *f = fopen(fname, "rb");
	if (f==NULL) {
		printf("CsvRead ERROR: could not open %s for reading\n", fname);
		return NULL;
	}

	// How large is it ?
	fseek(f, 0, SEEK_END);
	size_t len = ftell(f);
	fseek(f, 0, SEEK_SET);

	// Read file to a buffer
	char *buf = (char*)malloc(len+1);
	int rt=fread(buf, 1, len, f);
	buf[len] = '\0';
	fclose(f);

	//------------
	// How many rows/cols
	//------------
	int nRows=0;
	int nCols=1;
	int curr_row=0;
	int curr_col=0;
	int isInQuote=0;
//printf("A\n");
	for (i=0; i<len; i++)
	{
		if (checkQuote && buf[i]=='"')
			isInQuote = !isInQuote;
//if (debug) { printf("buf[%d] %c   isInQuote %d\n", i, buf[i], isInQuote); } //fgetc(stdin); }
if (debug && nRows>10500) printf("%c", buf[i]);


		if (!isInQuote)
		{
			// Check for row delim
			for (j=0; row_delim[j]; j++)
				if (buf[i]==row_delim[j])
					goto ROW_DELIM;

			// Check for col delim
			for (j=0; col_delim[j]; j++)
				if (buf[i]==col_delim[j])
					goto COL_DELIM;
		}
		
		// Normal state
		continue;

		// Row delimeter state
ROW_DELIM:
		curr_row++;
		curr_col=0;
		nRows = curr_row+1;
//if (debug) { printf("row\n"); fgetc(stdin); }
if (debug && nRows>10500) { printf("  nRows: %d  i %d len %d\n", (int)nRows, (int)i, (int)len); }
if (debug && nRows>10500) fgetc(stdin);
		continue;

		// Col delimeter state
COL_DELIM:
if (debug && nRows>10500) {printf(",");}
		curr_col++;
		if (curr_col+1 > nCols)
			nCols = curr_col+1;
//if (debug) { printf("col\n"); fgetc(stdin); }
		continue;
	}
printf("B nRows %d nCols %d\n", nRows, nCols);
if (debug) { fgetc(stdin); }


printf(">Allocate the CSV 2D array\n");
	//------
	// Allocate the CSV 2D array
	//------
	char ***csv = (char***)malloc(nRows * sizeof(char**));
	for (i=0; i<nRows; i++)
		csv[i] = (char**)calloc(nCols+1, sizeof(char*));
	
	//------------
	// How many rows/cols
	//------------
	curr_row=0;
	curr_col=0;
	csv[curr_row][curr_col] = buf;
//printf("C\n");
	isInQuote=0;
	for (i=0; i<len; i++)
	{
		if (checkQuote && buf[i]=='"')
			isInQuote = !isInQuote;
//if (debug) { printf("buf[%d] %c   isInQuote %d\n", i, buf[i], isInQuote); } //fgetc(stdin); }
		
		if (!isInQuote)
		{
			// Check for row delim
			for (j=0; row_delim[j]; j++)
				if (buf[i]==row_delim[j])
					goto ROW_DELIM2;

			// Check for col delim
			for (j=0; col_delim[j]; j++)
				if (buf[i]==col_delim[j])
					goto COL_DELIM2;
		}
		
		// Normal state
		continue;

		// Row delimeter state
ROW_DELIM2:
		curr_row++;
		curr_col=0;
		buf[i] = '\0';
		if (curr_row<nRows && curr_col<nCols)
			csv[curr_row][curr_col] = &(buf[i+1]);
//if (debug) { printf("row\n"); fgetc(stdin); }
		continue;

		// Col delimeter state
COL_DELIM2:
		curr_col++;
		buf[i] = '\0';
		if (curr_row<nRows && curr_col<nCols)
			csv[curr_row][curr_col] = &(buf[i+1]);		
//if (debug) { printf("col\n"); fgetc(stdin); }
		continue;
	}
//printf("D\n");
	
	*rows = nRows;
	*cols = nCols;
	
	//------------
	// Clean up the CSV file
	//    case 1  Replace NULL entries with empty string
	//    case 2  remove any "strip characters"
	//------------
	for (y=0; y<nRows; y++) {
		for (x=0; x<nCols; x++) {
			if (csv[y][x] == NULL)
				csv[y][x] = &(buf[len]);    // Point to an empty string
			else
			{
				int out_idx = 0;
				for (i=0; csv[y][x][i]; i++)
				{
					for (j=0; strip_chars[j]; j++)
						if (csv[y][x][i] == strip_chars[j])
							goto STRIP_CHAR;
					csv[y][x][out_idx++] = csv[y][x][i];
STRIP_CHAR:;
				}
				csv[y][x][out_idx] = '\0';
			}
		}
	}
//debug=1;
	return csv;
}

static char ***CsvReadDefault(const char *fname, int *rows, int *cols)
{
	printf("AAA\n");
	return CsvRead(fname, rows, cols, ",\t", "\n", "\"\r", 1);
}




static void CsvFree(char***csv, int rows, int cols)
{
	int i;
	if (csv[0][0])
		free(csv[0][0]);
	for (i=0; i<rows; i++)
		if (csv[i])
			free(csv[i]);
	free(csv);
}

static int StrIdx(char **row, const char *str)
{
	int i;
	for (i=0; row[i]; i++)
		if (strcmp(row[i], str)==0)
			return i;
	return -1;
}




#endif // _DC_CSV_H_

