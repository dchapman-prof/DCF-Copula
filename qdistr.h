#ifndef _QDISTR_H_
#define _QDISTR_H_


#define ABS(x)   ( (x)<=0 ? 0-(x) : (x) )
#define RANDF ((double)rand() / (double)RAND_MAX)
#define RANDBALF (2.0f*RANDF - 1.0f)
#define PI    3.14159265358979323846264
#define SQRT2 1.41421356237309504880168

#define QD_METRIC_KL 0
#define QD_METRIC_WAS 1

// Incomplete gamma using the continued fraction method
double incom_gamma(double a, double x);

// Error function approximation (Abramowitz & Stegun, 1964)
double erf_approx(double x);



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
	double was_dist;
} Distribution;
#define DISTRIB_UNIFORM      0
#define DISTRIB_GAUSSIAN     1
#define DISTRIB_EXPONENTIAL  2
#define DISTRIB_GAMMA        3
#define DISTRIB_WEIBULL      4
//#define DISTRIB_TRUNC_NORMAL 5
#define DISTRIB_GPD          5
#define N_DISTRIB_TYPES      6      // Remove truncated normal


extern const char *distr_name[N_DISTRIB_TYPES];
extern const char *distr_p1_name[N_DISTRIB_TYPES];
extern const char *distr_p2_name[N_DISTRIB_TYPES];

extern int qd_debug;
extern int qd_test;



// Ensure the distribution parameters are within valid ranges
void BoundsDistribution(int type, double *p_param1, double *p_param2);

//
// PlotDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins);

//
// PlotDistribution CDF using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotCdfDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins);


//
// PlotLogDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotLogDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins);

//
// Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
double CrossEntropy(double *X, double *Y, double *Yhat, int startBin, int endBin);

//
// Log of Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
double LogCrossEntropy(double *X, double *Y, double *log2Yhat, int startBin, int endBin);

double WassersteinDistance(double *X, double *cdfY, double *cdfYhat, int startBin, int endBin);







//
// Initial guess using the midpoint rule
//
//    X     length N+1   (for midpoint rule)
//    Y     length N     (at midpoints)
//
typedef struct InitialGuess {
	double param1;        // Guess for parameters 1 and 2
	double param2;
	double param1_step;   // Step-size for simulated annealing
	double param2_step;
} InitialGuess;


InitialGuess InitialGuessDistribution(int type, double *X, double *Y, int N);


Distribution FitDistribution(int type, int metric, double *X, double *Y, double *cdf_Y, int N0, int N, double *log_Yhat, double *cdf_Yhat);



typedef struct QDistr
{
	double **X;           // [nFeat nBins+1]
	double **Y;           // [nFeat nBins]      midpoint rule
	double **Y_cdf;       // [nFeat nBins]
	double *zerofrac;     // [nFeat]
	double *nonzerofrac;  // [nFeat]
	int    *istart_quantile;  // [nFeat]
	
	int nFeat;
	int nBins;
} QDistr;

QDistr ReadQDistr(const char *dataset, const char *model, int seed, int nquant, char *series, int includezero, double start_quantile);

void QDistrFree(QDistr qd);


#endif // _QDISTR_H_
