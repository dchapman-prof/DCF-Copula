#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dc_csv.h"
#include "qdistr.h"


int qd_debug=0;
int qd_test=0;

const char *distr_name[N_DISTRIB_TYPES]     = {"uniform",   "gaussian", "exponential", "gamma",   "weibull", "gpd"};     //, "trunc_normal"};
const char *distr_p1_name[N_DISTRIB_TYPES]  = {"mu",        "mu",       "lamda",       "alpha",   "lamda",   "s"};       //,   "mu"};
const char *distr_p2_name[N_DISTRIB_TYPES]  = {"halfwidth", "stdev",    "none",        "beta",    "k",       "xi"};      //,       "stdev"};



static void input() {
	printf("Press enter to continue\n");
	fgetc(stdin);
	//printf("enter a string to continue (t turns off testing)\n");
	//char str[16];
	//scanf("%s", str);
	//printf("str %s\n", str);
	//if (strcmp(str,"t")==0)
	//	test=0;
}



//----------------
// Incomplete gamma using the continued fraction method
//----------------


double incom_gamma(double a, double x)
{
//if (qd_test) printf("BEGIN incom_gamma   a %f  x %f\n", a, x);

	//
	// The cf expansion is the following
	//
	//   g(a,x) =  (x^a e^-x)/a ( 1 + x/(a+1) ( 1 + x/(a+2) ( 1 + x/(a+3) ( . . .
	//
	//          =        A      ( 1 +    B    ( 1 +    C    ( 1 +   D     ( . . .
	//
	//          =        A     +      AB      +     ABC     +    ABCD     + . . .
	//
	double precision = 0.0000000001;
	
	double sum = 0.0;
	double term = pow(x,a) * exp(-x) / a;
	sum += term;

//if (qd_test) printf("incom_gamma term  %f   sum %f   x %f a %f  n  0\n", term, sum, x, a);
	
	int n = 1;
	while (term>precision) {
		term *= x/(a+n);
		sum += term;
//if (qd_test) printf("incom_gamma term  %f   sum %f   x %f a %f  n  %d\n", term, sum, x, a, n);
		n++;
//if (qd_test) input();
	}
//if (qd_test) printf("END incom_gamma\n");


	return sum;
}


// Error function approximation (Abramowitz & Stegun, 1964)
double erf_approx(double x) {
    // Constants
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;
    const double p = 0.3275911;

    // Save the sign of x
    int sign = (x < 0) ? -1 : 1;
    x = fabs(x);

    // Compute approximation
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}


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
			if (*p_param1 > 20.0)
				*p_param1 = 20.0;                        // alpha cannot be too large (unstable)
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // beta must be positive
			break;
		case DISTRIB_WEIBULL:
			if (*p_param1 < 0.0000001)
				*p_param1 = ABS(*p_param1) + 0.0000001;  // lamda must be positive
			if (*p_param2 < 0.0000001)
				*p_param2 = ABS(*p_param2) + 0.0000001;  // k must be positive
			break;
		case DISTRIB_GPD:
			if (*p_param1 < 0.0000001)
				*p_param1 = ABS(*p_param1) + 0.0000001;  // s must be positive

		//case DISTRIB_TRUNC_NORMAL:
		//	if (*p_param2 < 0.0000001)
		//		*p_param2 = ABS(*p_param2) + 0.0000001;  // stdev must be positive
		//	break;
	}
}


//
// PlotDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins) {
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
	double gpd_s    = param1;
	double gpd_xi   = param2;
	double gpd_s_over_xi = gpd_s / gpd_xi;
	double gpd_over_s = 1.0 / gpd_s;
	double gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		double x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0 || x>unif_x1)
				Y[i] = 0.0;
			else
				Y[i] = unif_y;
			break;

		case DISTRIB_GAUSSIAN:
				Y[i] = norm_coef * exp(norm_scale*(x-norm_mu)*(x-norm_mu));
			break;

		case DISTRIB_EXPONENTIAL:
			if (x<0.00000001)
				Y[i] = 0.0;
			else
				Y[i] = exp_lamda * exp(-exp_lamda * x);
			break;

		case DISTRIB_GAMMA:
			if (x<0.00000001)
				Y[i] = 0.0;
			else
				Y[i] = gam_coef * pow(x, gam_alpha-1.0) * exp(-gam_beta * x);
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00000001)
				Y[i] = 0.0;
			else
				Y[i] = (wei_k / wei_lam) * pow((x/wei_lam),(wei_k-1)) * exp(-pow((x/wei_lam),(wei_k)));
			break;

		case DISTRIB_GPD:
			// Negative x
			if (x<0.00000001)
				Y[i] = 0.0;
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				Y[i] = 0.0;
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				Y[i] = gpd_over_s * exp(-x * gpd_over_s);
			// General case
			else
				Y[i] = gpd_over_s * pow( (1.0 + gpd_xi*x*gpd_over_s),  -(gpd_over_xi + 1.0));
			break;

		//case DISTRIB_TRUNC_NORMAL:
		//	if (x<0.0)
		//		Y[i] = 0.0;
		//	else
		//		Y[i] = trunc_norm_coef * norm_coef * exp(norm_scale*(x-norm_mu)*(x-norm_mu));
		//	break;
		}
	}
}


//
// PlotDistribution CDF using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotCdfDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins) 
{
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
	double gam_coef  = 1.0 / tgamma(gam_alpha);
	double wei_lam   = param1;
	double inv_wei_lam = 1.0 / wei_lam;
	double wei_k     = param2;
	double trunc_norm_F0 = 0.5 * ( 1 + erf_approx((0.0-norm_mu)/(norm_sig*1.41421356237)));
	double gpd_s    = param1;
	double gpd_xi   = param2;
	double gpd_s_over_xi = gpd_s / gpd_xi;
	double gpd_over_s = 1.0 / gpd_s;
	double gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		double x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0)
				Y[i] = 0.0;
			else if (x<unif_x1)
				Y[i] = (x-unif_x0) / (unif_x1-unif_x0);
			else
				Y[i] = 1.0;
			break;

		case DISTRIB_GAUSSIAN:// {
				//double A = x-norm_mu;
				//double B = norm_sig*1.41421356237;
				//double C = A/B;
				//double D = erf_approx(C);
				//double E = 0.5 * ( 1 + D );
				//Y[i] = E;
				//printf("i %d x %f norm_mu %f norm_sig %f x-norm_mu %f norm_sig*1.41421356237 %f (x-norm_mu)/(norm_sig*1.41421356237) %f erf %f Y %f\n", i, x, norm_mu, norm_sig, A,B,C,D,E);
				Y[i] = 0.5 * ( 1 + erf_approx((x-norm_mu)/(norm_sig*1.41421356237)));
			break;
			//}

		case DISTRIB_EXPONENTIAL:
			if (x<0.00000001)
				Y[i] = 0.0;
			else
				Y[i] = 1.0 - exp(-exp_lamda * x);
			break;

		case DISTRIB_GAMMA:
			if (x<0.00000001)
				Y[i] = 0.0;
			else {
				//double A = gam_coef;
				//double B = gam_alpha;
				//double C = x*gam_beta;
				//double D = incom_gamma(B,C);
				//double E = A*D;
				//Y[i] = E;
				//printf("i %d gam_coef %f gam_alpha %f x*gam_beta %f incom_gamma(B,C) %f Y %f\n", i, A,B,C,D,Y[i]);
			
				Y[i] = gam_coef * incom_gamma(gam_alpha, x * gam_beta);
			}
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00000001)
				Y[i] = 0.0;
			else {
				//printf("i %d wei_lam %f  x %f  wei_k %f   -wei_lam*x %f   pow(-wei_lam*x,wei_k) %f exp  %f  Y %f\n",
				// i, wei_lam, x, wei_k, -wei_lam*x, -pow(wei_lam*x,wei_k), exp(-pow(wei_lam*x,wei_k)), 1.0 - exp(-pow(wei_lam * x, wei_k)) );
				Y[i] =  1.0 - exp(-pow(x * inv_wei_lam, wei_k));
			}
			break;

		case DISTRIB_GPD:
			// If out of bounds
			if (x<0.00000001)
				Y[i] = 0.0;
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				Y[i] = 1.0;
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				Y[i] = 1.0 - exp(-x * gpd_over_s);
			// General case
			else
				Y[i] = 1.0 - pow( (1.0 + gpd_xi*x*gpd_over_s),  -gpd_over_xi);
			break;

		//case DISTRIB_TRUNC_NORMAL:
		//	if (x<0.0)
		//		Y[i] = 0.0;
		//	else {
		//		double cdf_norm = 0.5 * ( 1 + erf_approx((x-norm_mu)/(norm_sig*1.41421356237)));
		//		Y[i] = (cdf_norm - trunc_norm_F0) / (1.0 - trunc_norm_F0);
		//	}
		//	break;
		}
	}
	
	//if (type==DISTRIB_GAUSSIAN)
	//	input();
}


//
// PlotLogDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
void PlotLogDistribution(int type, double param1, double param2, double *X, double *Y, int startBin, int nBins) {
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
	double e = 2.718281828459045235360287471352;
	double gpd_s    = param1;
	double gpd_xi   = param2;
	double gpd_s_over_xi = gpd_s / gpd_xi;
	double gpd_over_s = 1.0 / gpd_s;
	double gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		double x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0 || x>unif_x1)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				Y[i] = log2(unif_y);
			break;

		case DISTRIB_GAUSSIAN:
				
				//Y[i] = norm_coef * exp(norm_scale*(x-norm_mu)*(x-norm_mu));
				Y[i] = log2(norm_coef) + log2(e)*(norm_scale*(x-norm_mu)*(x-norm_mu));
			break;

		case DISTRIB_EXPONENTIAL:
			if (x<0.00000001)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				//Y[i] = exp_lamda * exp(-exp_lamda * x);
				Y[i] = log2(exp_lamda) + log2(e)*(-exp_lamda * x);
			break;

		case DISTRIB_GAMMA:
			if (x<0.00000001)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				//Y[i] = gam_coef * pow(x, gam_alpha-1.0) * exp(-gam_beta * x);
				Y[i] = log2(gam_coef) + (gam_alpha-1.0)*log2(x) + (-gam_beta * x)*log2(e);
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00000001)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				//Y[i] = (wei_k / wei_lam) * pow((x/wei_lam),(wei_k-1)) * exp(-pow((x/wei_lam),(wei_k)));
				Y[i] = log2(wei_k) - log2(wei_lam) + (wei_k - 1) * log2( x / wei_lam) - pow(x / wei_lam, wei_k)*log2(e);
				
			break;

		case DISTRIB_GPD:
			// Negative x
			if (x<0.00000001)
				// Y[i] = 0.0
				Y[i] = log2(1e-10);     // to avoid log2(0)
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				//Y[i] = 0.0
				log2(1e-10);     // to avoid log2(0)
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				//Y[i] = gpd_over_s * exp(-x * gpd_over_s)
				Y[i] = log2(gpd_over_s) + log2(e)*(-gpd_over_s * x);
			// General case
			else
				//Y[i] = gpd_over_s * pow( (1.0 + gpd_xi*x*gpd_over_s),  -(gpd_over_xi + 1.0))
				Y[i] = log2(gpd_over_s) + log2( (1.0 + gpd_xi*x*gpd_over_s) ) * (-(gpd_over_xi + 1.0));
			break;

		//case DISTRIB_TRUNC_NORMAL:
		//	if (x<0.0)
		//		Y[i] = log2(1e-10);     // to avoid log2(0)
		//	else
		//		Y[i] = log2(trunc_norm_coef) + log2( norm_coef) + (norm_scale*(x-norm_mu)*(x-norm_mu))*log2(e);
		//	break;
		}
	}
}




//
// Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
double CrossEntropy(double *X, double *Y, double *Yhat, int startBin, int endBin)
{
	int i;

	// Calculate cross entropy
	double entropy = 0.0;
	for (i=startBin; i<endBin; i++) {
		double x0 = X[i];
		double x1 = X[i+1];
		double step = x1-x0;    // What is the step size of the bin ?
		
		double y = Y[i];
		double yhat = Yhat[i];
		entropy -= step * y * log2(yhat);
	}
	return entropy;
}


//
// Log of Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
double LogCrossEntropy(double *X, double *Y, double *log2Yhat, int startBin, int endBin)
{
	int i;

	// Calculate cross entropy
	double entropy = 0.0;
	for (i=startBin; i<endBin; i++) {
		double x0 = X[i];
		double x1 = X[i+1];
		double step = x1-x0;    // What is the step size of the bin ?
		
		double y = Y[i];
		double log2_yhat = log2Yhat[i];
		entropy -= step * y * log2_yhat;
	}
	return entropy;
}


double WassersteinDistance(double *X, double *cdfY, double *cdfYhat, int startBin, int endBin)
{
	int i;
	
	double dist = 0.0;
	for (i=startBin; i<endBin; i++) {
		double x0 = X[i];
		double x1 = X[i+1];
		double step = x1-x0;    // What is the step size of the bin ?
		
		double cdf_y = cdfY[i];
		double cdf_yhat = cdfYhat[i];
		double diff = cdf_y - cdf_yhat;
		dist += step * ABS(diff);
	}
	return dist;
}


//
// Initial guess using the midpoint rule
//
//    X     length N+1   (for midpoint rule)
//    Y     length N     (at midpoints)
//
InitialGuess InitialGuessDistribution(int type, double *X, double *Y, int N)
{
	int i;

	// Calculate the mean
	double total = 0.0;
	double count = 0.0;
	for (i=0; i<N; i++) {
		double x = 0.5*(X[i]+X[i+1]);   // midpoint rule
		double step = X[i+1]-X[i];
		double weight = step*Y[i];
		total += x*weight;
		count += weight;
		//printf("total_mean %f\n", total);
		//printf("count_mean %f\n", count);
		//printf("bin %d - x: %f, total_mean: %f, count_mean: %f, step : %f, weight: %f\n", 
                       // i, x, total, count, step, weight);      
	}
	double mean = total / count;
	//printf("mean: total/count %f\n", mean);
if(qd_debug)printf("mean %f\n", mean);

	// Calculate the variance
	total = 0.0;
	count = 0.0;
	for (i=0; i<N; i++) {
		double x = 0.5*(X[i]+X[i+1]);   // midpoint rule
		double step = X[i+1]-X[i];
		double weight = step*Y[i];
		
		//printf("bin %d - x: %f, mean: %f, step: %f, weight: %f\n", i, x, mean, step, weight);
		
		total += (x-mean)*(x-mean)*weight;
		count += weight;
		//printf("total_var %f\n", total);
		//printf("count_var %f\n", count);
		//printf("bin %d - total_var: %f, count_var: %f, weight: %f\n", 
                     //   i, total, count, weight);
	}
	double variance = total / count;
	//printf("variance %f\n", variance);
	
if(qd_debug)printf("variance %f\n", variance);

	// Calculate standard deviation
	double stdev = sqrt(variance);
if(qd_debug)printf("stdev %f\n", stdev);

	// Estimate parameters
	double param1;  //    mean       mean        lamda      alpha     lamda     s
	double param2;  // halfwidth     stdev        N/A       beta        k       xi
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
		guess.param2 = mean / variance ;      // beta
		guess.param1 = mean * guess.param2;  // alpha
		guess.param1_step = guess.param1;
		guess.param2_step = guess.param2;
		break;
// Y[i] = exp_lamda * exp(-exp_lamda * x);
// Y[i] = (wei_k / wei_lam) * pow((x/wei_lam),(wei_k-1)) * exp(-pow((x/wei_lam),(wei_k)));
// Y[i] = gpd_over_s * exp(-x * gpd_over_s)

	case DISTRIB_WEIBULL:              // exponential guess
		guess.param1 = mean;
		guess.param2 = 1.0;
		guess.param1_step = mean;
		guess.param2_step = 1.0;
		break;

	case DISTRIB_GPD:                  // exponential guess
		guess.param1 = mean;
		guess.param2 = 0.0;
		guess.param1_step = mean;
		guess.param2_step = 0.5;

	//case DISTRIB_TRUNC_NORMAL:              // uninformed guess
	//	guess.param1 = 0.0;
	//	guess.param2 = 1.0;
	//	guess.param1_step = 1.0;
	//	guess.param2_step = 1.0;
	//	break;
	}
	return guess;
}

double temparray[100000];


Distribution FitDistribution(int type, int metric, double *X, double *Y, double *cdf_Y, int N0, int N, double *log_Yhat, double *cdf_Yhat)
{
//qd_debug=1;
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

	// HACK
	if (type == DISTRIB_WEIBULL)
	{
		// Starting bin for initial guess fit
		int iN0 = N0;
		//if (iN0 < N/4)
		//	iN0 = N/4;
			
		// Second bin for initial guess fit
		int iN1 = iN0 + 9*(N-iN0) / 10;
		if (iN1 >= N)
			iN1 = N-1;
		double x0 = 0.5*(X[iN0]+X[iN0+1]);    // first x midpoint rule
		double x1 = 0.5*(X[iN1]+X[iN1+1]);    // second x midpoint rule
		double F0 = cdf_Y[iN0];
		double F1 = cdf_Y[iN1];
		//printf(" HACK x0 %f  x1 %f  F0 %f  F1 %f   N0 %d iN0 %d N %d iN1 %dn", x0, x1, F0, F1, N0, iN0, N, iN1);
		//input();
		
		// Strong initial guess using the two points
		double wei_k   = log( log(1-F0)/log(1-F1) ) / log( x0/x1 );
		double wei_lam = x0 / pow( -log(1-F0), 1.0/wei_k );

		// Store in the guess
		guess.param1 = wei_lam;
		guess.param2 = wei_k;
		guess.param1_step = wei_lam;
		guess.param2_step = wei_k;

		// Avoid copy/paste errors
		guess_param1 = guess.param1;
		guess_param2 = guess.param2;
		param1_step  = guess.param1_step;
		param2_step  = guess.param2_step;
		//printf(" HACK guess_param1 %f  guess_param2 %f  param1_step %f  param2_step %f\n", guess_param1, guess_param2, param1_step, param2_step);
		//input();
	}


	// Assert that mean is positive   (ToDo: This is only a problem for Gamma and Exponential
//	if (mean<0.00001) {
//		printf("WARNING: Cannot fit distribution, mean %f is not sufficiently positive", mean);
//		return distr;
//	}

	// Calculate Entropy
	double entropy = CrossEntropy(X, Y, Y, N0, N);

	// Fit using simulated annealing
	double param1 = guess_param1;
	double param2 = guess_param2;
	//PlotDistribution(type, param1, param2, X, log_Yhat, N0, N);
	PlotLogDistribution(type, param1, param2, X, log_Yhat, N0, N);
	PlotCdfDistribution(type, param1, param2, X, cdf_Yhat, N0, N);
	//double cross_entropy = CrossEntropy(X, Y, log_Yhat, N0, N);
	double cross_entropy = LogCrossEntropy(X, Y, log_Yhat, N0, N);
	double was_dist = WassersteinDistance(X, cdf_Y, cdf_Yhat, N0, N);
	double loss = (metric==QD_METRIC_KL) ? cross_entropy : was_dist;
	printf(">   guess   %s %s %f %s %f cross_entr %f was_dist %f\n", distr_name[type], distr_p1_name[type], param1, distr_p2_name[type], param2, cross_entropy, was_dist);
	if (qd_debug) {
		printf("cdf_Y   ");
		for (i=0; i<10; i++) {
			int idx = N0 + i*(N-N0-1)/9;
			printf("  %f", cdf_Y[idx]);
		}
		printf("\n");
		printf("cdf_Yhat");
		for (i=0; i<10; i++) {
			int idx = N0 + i*(N-N0-1)/9;
			printf("  %f", cdf_Yhat[idx]);
		}
		printf("\n");
		printf("Y       ");
		for (i=0; i<10; i++) {
			int idx = N0 + i*(N-N0-1)/9;
			printf("  %f", Y[idx]);
		}
		printf("\n");
		PlotDistribution(type, param1, param2, X, temparray, N0, N);
		printf("Yhat    ");
		for (i=0; i<10; i++) {
			int idx = N0 + i*(N-N0-1)/9;
			printf("  %f", temparray[idx]);
		}
		printf("\n");
	}

	//input();
//if (type==DISTRIB_WEIBULL)input();
	for (iter=0; iter<500; iter++) {
		double new_param1 = param1 + RANDBALF * param1_step;
		double new_param2 = param2 + RANDBALF * param2_step;

		BoundsDistribution(type, &new_param1, &new_param2);

		//PlotDistribution(type, new_param1, new_param2, X, log_Yhat, N);
		PlotLogDistribution(type, new_param1, new_param2, X, log_Yhat, N0, N);
		PlotCdfDistribution(type, new_param1, new_param2, X, cdf_Yhat, N0, N);
		double new_cross_entropy = LogCrossEntropy(X, Y, log_Yhat, N0, N);
		double new_was_dist      = WassersteinDistance(X, cdf_Y, cdf_Yhat, N0, N);
		double new_loss = (metric==QD_METRIC_KL) ? new_cross_entropy : new_was_dist;
		if (new_loss < loss && !isnan(new_loss) && !isinf(new_loss)) {
			param1 = new_param1;
			param2 = new_param2;
			cross_entropy = new_cross_entropy;
			was_dist = new_was_dist;
			loss = new_loss;
			if (qd_debug) {
				printf(">  new %s %s %f %s %f cross_entr %f was_dist %f\n", distr_name[type], distr_p1_name[type], param1,distr_p2_name[type], param2, cross_entropy, was_dist);
				printf("cdf_Y   ");
				for (i=0; i<10; i++) {
					int idx = N0 + i*(N-N0-1)/9;
					printf("  %f", cdf_Y[idx]);
				}
				printf("\n");
				printf("cdf_Yhat");
				for (i=0; i<10; i++) {
					int idx = N0 + i*(N-N0-1)/9;
					printf("  %f", cdf_Yhat[idx]);
				}
				printf("\n");
				printf("Y       ");
				for (i=0; i<10; i++) {
					int idx = N0 + i*(N-N0-1)/9;
					printf("  %f", Y[idx]);
				}
				printf("\n");
				PlotDistribution(type, new_param1, new_param2, X, temparray, N0, N);
				printf("Yhat    ");
				for (i=0; i<10; i++) {
					int idx = N0 + i*(N-N0-1)/9;
					printf("  %f", temparray[idx]);
				}
				printf("\n");
			}
		}

		param1_step *= 0.97;   // falloff
		param2_step *= 0.97;   // falloff
	}


	distr.param1 = param1;
	distr.param2 = param2;
	distr.entropy = entropy;
	distr.cross_entropy = cross_entropy;
	distr.kl_diver = cross_entropy - entropy;
	distr.was_dist = was_dist;
	printf(">  solution  %s %s %f %s %f cross %f entro %f kl %f was %f\n",
		distr_name[type], distr_p1_name[type], param1,
		 distr_p2_name[type], param2,
		 cross_entropy, entropy, distr.kl_diver, distr.was_dist);
	if(qd_debug)input();


	if (qd_test) {
		printf("qd_test !!!!\n");
		for (i=N0; i<N; i++) {
			printf(" i %d cdf_Y %.16f cdf_Yhat %.16f Y %.16f Yhat %.16f\n", i, cdf_Y[i], cdf_Yhat[i], Y[i], temparray[i]);
		}
		qd_test=0;
	}

	return distr;
}



QDistr ReadQDistr(const char *dataset, const char *model, int seed, int nquant, char *series, int includezero, double start_quantile)
{
	int y,x;
	
	printf("----------\n");
	printf(" ReadQDistr\n");
	printf("  dataset        %s\n", dataset);
	printf("  model          %s\n", model);
	printf("  seed           %d\n", seed);
	printf("  nquant         %d\n", nquant);
	printf("  series         %s\n", series);
	printf("  includezero    %d\n", includezero);
	printf("  start_quantile %f\n", start_quantile);
	printf("----------\n");
	
	QDistr qd;
	memset(&qd, 0, sizeof(QDistr));   // Initilize to all NULL values
	
	printf("-------\n");
	printf(" Read the CSV file (for quantiles)\n");
	printf("-------\n");
	char path[4096];	
	sprintf(path, "quantiles/%s_%s_%d_%d/%s.csv", dataset, model, seed, nquant, series);
	printf("CsvRead %s\n", path);
	int rows,cols;
	char ***csv = CsvReadDefault(path, &rows, &cols);
	if (csv==NULL) {
		printf("WARNING could not read %s skipping\n", path);
		return qd;   // Return NULL values
	}

	// HACK  remove last row/column, because of \t\n problem (fix later)
	rows--;
	cols--;
	
	printf("-------\n");
	printf(" Allocate the data arrays\n");
	printf("-------\n");
	int nFeat = rows;
	int nBins = cols-1;
	printf("nFeat %d  nBins %d\n", nFeat, nBins);
	//input();
	
	double **X          = (double**)malloc(nFeat*sizeof(double*));    // [nFeat nBins+1]
	double **Y          = (double**)malloc(nFeat*sizeof(double*));    // [nFeat nBins]      midpoint rule
	double **Y_cdf      = (double**)malloc(nFeat*sizeof(double*));    // [nFeat nBins]      midpoint rule
	double *zerofrac    = (double*) malloc(nFeat*sizeof(double));     // [nFeat]
	double *nonzerofrac = (double*) malloc(nFeat*sizeof(double));  // [nFeat]
	int    *istart_quantile = (int*)malloc(nFeat*sizeof(int));
	for (y=0; y<nFeat; y++) {
		X[y]     = (double*)malloc((nBins+1)*sizeof(double));
		Y[y]     = (double*)malloc((nBins)*sizeof(double));
		Y_cdf[y] = (double*)malloc((nBins)*sizeof(double));
	}

	printf("-------\n");
	printf(" Read the zerofrac dataset\n");
	printf("-------\n");
	if (!includezero) {
		// don't include zeros, include dummy values
		for (y=0; y<nFeat; y++) {
			zerofrac[y] = 0.0;
			nonzerofrac[y] = 1.0;
		}
	}
	else {
		// Include zeros, let's read the zeros file
		sprintf(path, "quantiles/%s_%s_%d_%d/%s.csv.nonzero.csv", dataset, model, seed, nquant, series);
		printf("CsvRead %s\n", path);
		int nonzero_rows=0,nonzero_cols=0;
		char ***nonzero_csv = CsvReadDefault(path, &nonzero_rows, &nonzero_cols);
		if (nonzero_csv==NULL) {
			printf("WARNING could not read %s exiting\n", path);
			exit(1);
		}

		// HACK Correct for one extra row and column
		nonzero_rows--;
		nonzero_cols--;
	
		// Print the attributes
		printf("nonzero rows %d cols %d\n", nonzero_rows, nonzero_cols);
		
		//
		// Extract percent of zero features to include . . .
		//
		for (y=0; y<nFeat; y++) {
			int num_nonzero = atoi(nonzero_csv[y+1][1]);
			int num_zero    = atoi(nonzero_csv[y+1][2]);
			int num = num_nonzero + num_zero;
			zerofrac[y] = (double)num_zero/(double)num;    // accurate values
			nonzerofrac[y] = 1.0 - zerofrac[y];            // if we include zeros
			printf("y %d zerofrac %f  nonzerofrac %f\n", y, zerofrac[y], nonzerofrac[y]);
		}
		//input();
		
		// Free the nonzero CSV file
		CsvFree(nonzero_csv,nonzero_rows+1,nonzero_cols+1);
	}
	
	printf("-------\n");
	printf(" For every feature . . .\n");
	printf("-------\n");
	for (y=0; y<nFeat; y++)
	{
		//printf("feature %d of %d\n", y, nFeat);

		//printf("-------\n");
		//printf(" Extract quantiles from csv file\n");
		//printf("-------\n");
		for (x=0; x<nBins+1; x++) {           // Read the training x values
			X[y][x] = atof(csv[y][x]);
			//printf("  y %d x %d  X[y][x] %f\n", y, x, X[y][x]);
		}
		//input();
		for (x=0; x<nBins; x++) {             // Compute y values using midpoint rule
			double xstep = X[y][x+1] - X[y][x];
			if (xstep>0.0000000001)
				Y[y][x] = nonzerofrac[y] / (xstep * nBins);
			else
				Y[y][x] = 0.0;
			//printf("  y %d x %d  Y[y][x] %f\n", y, x, Y[y][x]);
		}
		//input();
		// Include the zeros possibly
		double xstep = X[y][1] - X[y][0];
		Y[y][0] += zerofrac[y] / xstep;
		

		//printf("-------\n");
		//printf(" Calculate empirical CDF   (using midpoint rule)\n");
		//printf("-------\n");
		double half_step_curr = 0.5*(X[y][1]-X[y][0]);
		Y_cdf[y][0] = half_step_curr*Y[y][0];
		for (x=1; x<nBins; x++) {
			double half_step_prev = half_step_curr;
			half_step_curr = 0.5*(X[y][x+1] - X[y][x]);
			Y_cdf[y][x] = Y_cdf[y][x-1] + half_step_prev*Y[y][x-1] + half_step_curr*Y[y][x];
			//printf("  y %d x %d  Y_cdf[y][x] %.16f   Y[y][x] %.16f\n", y, x, Y_cdf[y][x], Y[y][x]);
		}
		//input();

		//printf("-------\n");
		//printf(" Calculate the integer start quantile\n");
		//printf("-------\n");
		for (x=0; x<nBins; x++) {
			//printf("x %d  Y_cdf[y] %f  start_quantile %f\n", x, Y_cdf[y][x], start_quantile);
			if (Y_cdf[y][x] >= start_quantile)
				break;
		}
		istart_quantile[y] = x;
		
		//printf("y %d istart_quantile %d\n", y, istart_quantile[y]);
		//input();
	}
	//input();
	
	for (x=0; x<nBins; x++) {
		y=28;
		//printf("  feat %d bin %d  x0 %f x1 %f  y %f  cdf %f\n", y, x, X[y][x], X[y][x+1], Y[y][x], Y_cdf[y][x]);
		//if (x%25==0) input();
	}
	//input();
	
	printf("-------\n");
	printf(" Free the CSV dataset\n");
	printf("-------\n");
	CsvFree(csv,rows+1,cols+1);
	
	printf("-------\n");
	printf(" Pack the output structure and return\n");
	printf("-------\n");
	qd.X = X;
	qd.Y = Y;
	qd.Y_cdf = Y_cdf;
	qd.zerofrac = zerofrac;
	qd.nonzerofrac = nonzerofrac;
	qd.istart_quantile = istart_quantile;
	qd.nFeat = nFeat;
	qd.nBins = nBins;
	return qd;
}

void QDistrFree(QDistr qd)
{
	int y;
	for (y=0; y<qd.nFeat; y++) {
		free(qd.X[y]);
		free(qd.Y[y]);
		free(qd.Y_cdf[y]);
	}
	free(qd.zerofrac);
	free(qd.istart_quantile);
}


