#include "archimedian.h"
#include <math.h>

#define MIN(a,b) ( (a)<(b) ? (a) : (b) )
#define MAX(a,b) ( (a)<(b) ? (b) : (a) )
#define MINMAX(x,min,max)   MIN((max),MAX((x),(min)))


//-------------------------------------------------------------------
//-------------------------------------------------------------------
// High level interface
//-------------------------------------------------------------------
//-------------------------------------------------------------------

const char *str_archi[NUM_ARCHI] = {"amh", "clayton", "frank", "gumbel", "joe"};

double archi_theta(int archi, double rho, double tau)
{
	double theta = -9999.0;

	switch (archi) {
	case ARCHI_AMH:
		theta = amh_theta(rho);
		break;
	case ARCHI_CLAYTON:
		theta = clayton_theta(tau);
		break;
	case ARCHI_FRANK:
		theta = frank_theta(tau);
		break;
	case ARCHI_GUMBEL:
		theta = gumbel_theta(tau);
		break;
	case ARCHI_JOE:
		theta = joe_theta(tau);
		break;
	}
	
	return theta;
}


double archi_copula(int archi, double theta, double x, double y)
{
	double C;
	double psi_x;
	double psi_y;
	

	switch (archi) {
	case ARCHI_AMH:
		psi_x = amh_generator(x,theta);
		psi_y = amh_generator(y,theta);
		C = amh_generator_inv(psi_x + psi_y, theta);
		break;
	case ARCHI_CLAYTON:
		psi_x = clayton_generator(x,theta);
		psi_y = clayton_generator(y,theta);
		C = clayton_generator_inv(psi_x + psi_y, theta);
		break;
	case ARCHI_FRANK:
		psi_x = frank_generator(x,theta);
		psi_y = frank_generator(y,theta);
		C = frank_generator_inv(psi_x + psi_y, theta);
		break;
	case ARCHI_GUMBEL:
		psi_x = gumbel_generator(x,theta);
		psi_y = gumbel_generator(y,theta);
		C = gumbel_generator_inv(psi_x + psi_y, theta);
		break;
	case ARCHI_JOE:
		psi_x = joe_generator(x,theta);
		psi_y = joe_generator(y,theta);
		C = joe_generator_inv(psi_x + psi_y, theta);
		break;
	}
	
	return C;
}

double archi_copula_density(int archi, double theta, double x, double y)
{
	double c;
	double psi_x;
	double psi_y;
	double dpsi_x;
	double dpsi_y;
	
	switch (archi) {
	case ARCHI_AMH:
		psi_x = amh_generator(x,theta);
		psi_y = amh_generator(y,theta);
		dpsi_x = amh_generator_derivative(x,theta);
		dpsi_y = amh_generator_derivative(y,theta);
		c = amh_inv_derivative(psi_x + psi_y, theta) * (dpsi_x + dpsi_y);
		break;
	case ARCHI_CLAYTON:
		psi_x = clayton_generator(x,theta);
		psi_y = clayton_generator(y,theta);
		dpsi_x = clayton_generator_derivative(x,theta);
		dpsi_y = clayton_generator_derivative(y,theta);
		c = clayton_inv_derivative(psi_x + psi_y, theta) * (dpsi_x + dpsi_y);
		break;
	case ARCHI_FRANK:
		psi_x = frank_generator(x,theta);
		psi_y = frank_generator(y,theta);
		dpsi_x = frank_generator_derivative(x,theta);
		dpsi_y = frank_generator_derivative(y,theta);
		c = frank_inv_derivative(psi_x + psi_y, theta) * (dpsi_x + dpsi_y);
		break;
	case ARCHI_GUMBEL:
		psi_x = gumbel_generator(x,theta);
		psi_y = gumbel_generator(y,theta);
		dpsi_x = gumbel_generator_derivative(x,theta);
		dpsi_y = gumbel_generator_derivative(y,theta);
		c = gumbel_inv_derivative(psi_x + psi_y, theta) * (dpsi_x + dpsi_y);
		break;
	case ARCHI_JOE:
		psi_x = joe_generator(x,theta);
		psi_y = joe_generator(y,theta);
		dpsi_x = joe_generator_derivative(x,theta);
		dpsi_y = joe_generator_derivative(y,theta);
		c = joe_inv_derivative(psi_x + psi_y, theta) * (dpsi_x + dpsi_y);
	
		break;
	}
	
	//printf(" x %f y %f theta %f psi_x %f psi_y %f dpsi_x %f dpsi_y %f c %f\n", x, y, theta, psi_x, psi_y, dpsi_x, dpsi_y, c);
	
	return c;
}


//-------------------------------------------------------------------
//-------------------------------------------------------------------
// Copula Generators, Inverses, and Derivatives
//-------------------------------------------------------------------
//-------------------------------------------------------------------


//------------------------------
//------------------------------
// 1. Frank Copula
//------------------------------
//------------------------------

//------------------------------
// Generator
//------------------------------
double frank_generator(double t, double theta) {
	if (theta == 0.0 || t == 0.0){
		return NAN;
	}
	double numerator   = (exp(-theta * t) - 1.0);
	double denominator = (exp(-theta) - 1.0);
	if (numerator == denominator){ return 0;}
	if ( numerator/denominator <=0 ){
		return NAN;
	}
	return -log((exp(-theta * t) - 1.0) / (exp(-theta) - 1.0));
}

//------------------------------
// Generator Inverse
//------------------------------
double frank_generator_inv(double s, double theta) {
	
	return - (1.0 / theta) * log(1.0 + (exp(-s) * (exp(-theta) - 1.0)));
}

//------------------------------
// Generator Derivative
//------------------------------
double frank_generator_derivative(double t, double theta) {
	//double log10_e     = log(10.0);
	double numerator   = theta * exp(-theta * t);
	double denominator = (exp(-theta * t) - 1.0);
	return numerator/denominator;
}

//------------------------------
// Inverse Derivative
//------------------------------
double frank_inv_derivative(double s, double theta) {
	//double log10_e     = log(10.0);
	double A           = exp(-theta) - 1.0;
	double B           = exp(-s);
	double numerator   = B * A;
	double denominator = (1.0 + B * A);
	double jitter      = 0.000000001;
	//double jitter = 1e-9;

	//if (theta == 0.0 || denominator == 0.0) {
	//	return NAN;                     //To avoid division by zero
	//}
	if (theta == 0.0) {
		return NAN;
	}
	
	if (denominator == 0.0){
		denominator += jitter;         //To avoid division by zero
	}
	//if (fabs(denominator) < 1e-12) {
	//	denominator = (denominator < 0 ? -1 : 1) * jitter;
	//}

	return (1.0 / theta) * (numerator / denominator);
}
//------------------------------
//------------------------------
// 2. Gumbel Copula 
//------------------------------
//------------------------------

//------------------------------
// Generator
//-----------------------------
double gumbel_generator(double t, double theta) {
	
	double jitter      = 0.000000001;
	if (t <= 0.0) t = jitter;        // Avoid log(0)
	if (t >= 1.0) t = 1.0 - jitter;  // Avoid log(1) = 0 and pow(0,theta)
	
	
	return pow(-log(t), theta);
}


//------------------------------
// Generator Inverse
//------------------------------
double gumbel_generator_inv(double s, double theta) {
	double gumbel = exp(-pow(s, 1.0 / theta));
	printf(" gumbel %f \n", gumbel);
	return gumbel;
}

//------------------------------
// Generator Derivative
//------------------------------
double gumbel_generator_derivative(double t, double theta) {
	//if (t <= 0.0 || log(t) == 0.0) {
	//	return NAN;                //To avoid log(0) or division by zero
	//}
	double jitter      = 0.000000001;
	if (t <= 0.0) t = jitter;        // Avoid log(0)
	if (t >= 1.0) t = 1.0 - jitter;
	//double log10_e     = log(10.0);    // ln(10)
	double min_ln_t        = -log(t);       // natural log of t

	double numerator   = theta * pow(min_ln_t, theta - 1.0);
	double denominator = -t;

	return numerator / denominator;
}

//------------------------------
// Inverse Derivative
//------------------------------

double gumbel_inv_derivative(double s, double theta) {
	if (theta == 0.0) return NAN;  // avoid division by zero

	//double result = - (1.0 / theta) * exp(-s / theta);
	double inv_theta = 1.0/theta;
	double result = exp(-pow(s,inv_theta))*((-inv_theta) * pow(s, inv_theta - 1.0));
	return result;
}

//------------------------------
//------------------------------
// 3. AMH Copula 
//------------------------------
//------------------------------

//------------------------------
// Generator
//-----------------------------
double amh_generator(double t, double theta) {
	//nom   = 1 - theta*(1-t);
	//denom = t;
	
	//return log(nom/denom);
	//return log((1.0 - theta * t) / (1.0 - theta));
	if (t == 0.0 || (1 - theta*(1-t))/t < 0.0){
		return NAN;
	}
	if ( theta*(1-t) == 1){
		return NAN;
	}
	return log((1 - theta*(1-t))/t);
}
//------------------------------
// Generator Inverse
//------------------------------
double amh_generator_inv(double s, double theta) {
	double denominator = exp(s) - theta;
	double jitter      = 0.000000001;
	
	if (denominator == 0.0){ //return INFINITY;      // avoid devision by zero
		denominator += jitter;
	}
	return (1-theta)/denominator;
}

//------------------------------
// Generator Derivative
//------------------------------
double amh_generator_derivative(double t, double theta) {
	if (t == 0.0 || 1.0 - theta + theta * t == 0.0) {
		return NAN;                     // To avoid division by zero
	}

	double numerator   = theta - 1;
	//double log10_e     = log(10.0);       // natural log of 10
	double denominator = t * (1.0 - theta + theta * t);

	return numerator / denominator;
}

//------------------------------
// Inverse Derivative
//------------------------------
double amh_inv_derivative(double s, double theta) {
	double numerator   = -exp(s)*(1-theta);
	double denominator = (exp(s)-theta)*(exp(s)-theta);
	if (denominator == 0.0){
		return NAN;			// To avoid division by zero
	}

	return numerator / denominator;
}
//------------------------------
//------------------------------
// 4. Joe Copula (θ >= 1)
//------------------------------
//------------------------------



//------------------------------
// Generator
//-----------------------------
double joe_generator(double t, double theta) {
	
	double jitter      = 0.000000001;
	//if (t >= 1.0) t = 1.0 - jitter;  // Clip from above
	//if (t <= 0.0) t = jitter;        // avoid t = 0
	
	double one_minus_t = 1.0 - t;
	if (one_minus_t <= jitter) {  
		one_minus_t = jitter;
	}
	
	return -log(1.0 - pow(1.0 - t, theta));
}

//------------------------------
// Generator Inverse
//------------------------------
double joe_generator_inv(double s, double theta) {
	return 1.0 - pow(1.0 - exp(-s), 1.0 / theta);
}
//------------------------------
// Generator Derivative
//------------------------------
double joe_generator_derivative(double t, double theta) {

	double one_minus_t = 1.0 - t;
	
	double jitter      = 0.000000001;
	//if (t >= 1.0) t = 1.0 - jitter;   // Clip from above
	//if (t <= 0.0) t = jitter; 
	//if (t == 1.0 || one_minus_t <= 0.0) {
	//	return NAN;                //To avoid log(0), pow of negative, or division by zero
	//}

	double numerator   = theta * pow(one_minus_t, theta - 1.0);
	double denominator = (1.0 - pow(one_minus_t, theta));

	if (denominator == 0.0) {
		denominator = jitter;
		//return NAN;
	}

	return - numerator / denominator;   

}

//------------------------------
// Inverse Derivative
//------------------------------
double joe_inv_derivative(double s, double theta) {
	if (theta == 0.0) return NAN;        //To avoid division by zero
	
	double jitter      = 0.000000001;
	//double jitter      = 1e-20;
	double one_minus_exp = 1.0 - exp(-s);
	//printf("one_minus_exp %f \n", one_minus_exp);
	//printf("enter!!!!");
	//fgetc(stdin);

	//if (one_minus_exp <= 0.0) {
	//	return NAN;                 //To avoid invalid pow for negative or zero base
	//}
	
	if (one_minus_exp == 0.0) {  // avoid zero or negative
		//printf("enterrrr!!!!");
		//fgetc(stdin);
		one_minus_exp = jitter;
	}
	
	double exponent = (1.0 / theta) - 1.0;
	double theta_inv = -1/theta;
	//printf("exponent %f \n", exponent);
	double value = pow(one_minus_exp, exponent);
	//printf("value %f \n", value);

	//double res = -exp(-s) / theta * value;
	//printf("res %f \n", res);
	///fgetc(stdin);
	//return -exp(-s) / theta * value;
	return  value * exp(-s) * theta_inv ;
}


//------------------------------
//------------------------------
// 5. Clayton Copula (θ > 0)
//------------------------------
//------------------------------

//------------------------------
// Generator
//-----------------------------
double clayton_generator(double t, double theta) {
	return (pow(t, -theta) - 1.0) / theta;
}
//------------------------------
// Generator Inverse
//------------------------------
double clayton_generator_inv(double s, double theta) {
	double power = -1.0 / theta;
	return pow(1.0 + theta * s, power);
	//return pow(1.0 + theta * s, -1.0 / theta);
} 

//------------------------------
// Generator Derivative
//------------------------------
double clayton_generator_derivative(double t, double theta) {
	if (t <= 0.0) return INFINITY;         // To avoid division by zero or negative powers
	return -pow(t, -(theta + 1.0));
}

//------------------------------
// Inverse Derivative
//------------------------------
double clayton_inv_derivative(double s, double theta) {
	if (theta == 0.0) return NAN;
	double power = (-1.0 / theta) - 1.0;
	return -pow(1.0 + theta * s, power);
}

/*
// ------------------ MAIN ------------------
int main() {
	double t = 0.5;       // input for generator & derivative
	double s = 0.5;       // input for inverse & inverse derivative
	double theta = 2.0;   // example copula parameter

	printf("==== Frank Copula ====\n");
	printf("Generator:           %f\n", frank_generator(t, theta));
	printf("Inverse Generator:   %f\n", frank_generator_inv(s, theta));
	printf("Derivative:          %f\n", frank_generator_derivative(t, theta));
	printf("Inverse Derivative:  %f\n\n", frank_inv_derivative(s, theta));

	printf("==== Gumbel Copula ====\n");
	printf("Generator:           %f\n", gumbel_generator(t, theta));
	printf("Inverse Generator:   %f\n", gumbel_generator_inv(s, theta));
	printf("Derivative:          %f\n", gumbel_generator_derivative(t, theta));
	printf("Inverse Derivative:  %f\n\n", gumbel_inv_derivative(s, theta));

	printf("==== AMH Copula ====\n");
	printf("Generator:           %f\n", amh_generator(t, theta));
	printf("Inverse Generator:   %f\n", amh_generator_inv(s, theta));
	printf("Derivative:          %f\n", amh_generator_derivative(t, theta));
	printf("Inverse Derivative:  %f\n\n", amh_inv_derivative(s, theta));

	printf("==== Joe Copula ====\n");
	printf("Generator:           %f\n", joe_generator(t, theta));
	printf("Inverse Generator:   %f\n", joe_generator_inv(s, theta));
	printf("Derivative:          %f\n", joe_generator_derivative(t, theta));
	printf("Inverse Derivative:  %f\n\n", joe_inv_derivative(s, theta));

	printf("==== Clayton Copula ====\n");
	printf("Generator:           %f\n", clayton_generator(t, theta));
	printf("Inverse Generator:   %f\n", clayton_generator_inv(s, theta));
	printf("Derivative:          %f\n", clayton_generator_derivative(t, theta));
	printf("Inverse Derivative:  %f\n\n", clayton_inv_derivative(s, theta));

	return 0;
}
*/

//-------------------------------------------------------------------
//-------------------------------------------------------------------
// Estimation of Copula theta parameter
//-------------------------------------------------------------------
//-------------------------------------------------------------------

//------------------------------
// Parameter estimation for the Joe Copula
//------------------------------

double joe_tau(double theta, int iter)
{
	int k;
	double sum=0;
	for (k=1; k<=iter; k++) {
		double numer = 4.0;
		double denom = k*(theta*k+2)*(theta*(k-1)+2);
		double term = numer / denom;
		sum+=term;
		//printf("k %d sum %f term %.16f\n", k, sum, term);
	}
	double tau = 1.0 - sum;
	//printf("theta %.16f iter %d tau %.16f\n", theta, iter, tau);
	return tau;
}

double joe_theta(double tau)
{
	int i;
	double thetam, taum;
	
	// An interval to search for
	double theta0 = 1.0;
	double theta1 = 100000.0;
	
	//
	double tau0 = joe_tau(theta0, 1000);
	//fgetc(stdin);
	double tau1 = joe_tau(theta1, 1000);
	//fgetc(stdin);
	
	for (i=0; i<32; i++) {
		thetam = 0.5*(theta0+theta1);
		taum = joe_tau(thetam, 1000);
		
		//printf("theta %f %f %f  tau %f %f %f  goal %f\n", theta0, thetam, theta1, tau0, taum, tau1, tau);

		if (tau<taum) {
			theta1 = thetam;
			tau1 = taum;
		}
		else {
			theta0 = thetam;
			tau0 = taum;
		}
	}
	return thetam;
}


double clayton_theta(double tau)
{
	return 2.0*tau / (1.0-tau);
}

double g_frank_integral[FRANK_NSTEPS+1];  // Trapezoid rule
int    g_frank_integral_initialized = 0;

double frank_tau(double theta)
{	
	int i;

	// If negative theta, negate it
	if (theta<0.0)
		return -frank_tau(-theta);
	
	// Very simple empirical small value approximation
	if (theta<0.1) {
		double tau1 = frank_tau(0.1);
		return tau1*10.0*theta;
	}

	// Initialize the frank integral
	double dt = (FRANK_THETA1-FRANK_THETA0) / FRANK_NSTEPS;
	double inv_dt = FRANK_NSTEPS / (FRANK_THETA1-FRANK_THETA0);
	if (!g_frank_integral_initialized) {
		g_frank_integral[0] = 0.0;
		double t = FRANK_THETA0 + dt;
		double curr_bar = 1.0;
		double prev_bar = 1.0;
		for (i=1; i<=FRANK_NSTEPS; i++) {
			prev_bar = curr_bar;
			curr_bar = t / (exp(t)-1.0);
			double trapezoid = 0.5*(prev_bar+curr_bar)*dt;
			g_frank_integral[i] = trapezoid + g_frank_integral[i-1];
			t += dt;

			//if (i%1000 == 1) {
			//	printf("frank init i %d curr_bar %f t %f sum %f\n", i, curr_bar, t, g_frank_integral[i]);
			//}
		}
		g_frank_integral_initialized = 1;
		
		//printf("dt %f\n", dt);
		//fgetc(stdin);
	}
	
	// Lookup the integral for theta
	i = FRANK_NSTEPS * (theta - FRANK_THETA0) / (FRANK_THETA1 - FRANK_THETA0);
	double igral = 1.64493406685;   // for large theta, it is pi^2/6
	
	if (i<FRANK_NSTEPS) {
		// Trapezoid rule with linear interpolation
		double igral0 = g_frank_integral[i];
		double igral1 = g_frank_integral[i+1];
		double theta0 = FRANK_THETA0 + i*dt;
		double delta = (theta-theta0)*inv_dt;
		igral = igral0 + delta*(igral1-igral0);
	}
	
	
	// Return the frank tau
	double tau = 1.0 - (4.0/theta)*(1.0 - igral/theta);
	return tau;
}


double frank_theta(double tau)
{
	int i;
	double thetam, taum;
	
	// An interval to search for
	double theta0 = -10000.0;
	double theta1 =  10000.0;
	
	//
	double tau0 = frank_tau(theta0);
	double tau1 = frank_tau(theta1);
	
	for (i=0; i<32; i++) {
		thetam = 0.5*(theta0+theta1);
		taum = frank_tau(thetam);
		
		//printf("theta %f %f %f  tau %f %f %f  goal %f\n", theta0, thetam, theta1, tau0, taum, tau1, tau);

		if (tau<taum) {
			theta1 = thetam;
			tau1 = taum;
		}
		else {
			theta0 = thetam;
			tau0 = taum;
		}
	}
	return thetam;
}

double gumbel_theta(double tau) {
	return 1.0 / (1.0-tau);
}

double amh_theta(double rho) {
	rho = MINMAX(rho, -0.2711, 0.4784);   // bounded range for rho
	return (3.0*rho) / (3.0 + rho);
}


//-------------------------------------------------------------------
//-------------------------------------------------------------------
// Estimation of Kendall's tau
//-------------------------------------------------------------------
//-------------------------------------------------------------------


// Count inversions during merge (discordant pairs)
static long long merge_and_count(Pair arr[], Pair temp[], int left, int mid, int right) {
    int i = left, j = mid, k = left;
    long long count = 0;

    while (i <= mid - 1 && j <= right) {
        if (arr[i].y <= arr[j].y)
            temp[k++] = arr[i++];
        else {
            temp[k++] = arr[j++];
            count += (mid - i);  // Count discordant pairs
        }
    }

    while (i <= mid - 1)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (i = left; i <= right; i++)
        arr[i] = temp[i];

    return count;
}


// Modified merge sort for counting discordant y-pairs
static long long merge_sort_and_count(Pair arr[], Pair temp[], int left, int right) {
    long long count = 0;
    if (right > left) {
        int mid = (left + right) / 2;

        count += merge_sort_and_count(arr, temp, left, mid);
        count += merge_sort_and_count(arr, temp, mid + 1, right);

        count += merge_and_count(arr, temp, left, mid + 1, right);
    }
    return count;
}


static void print_pairs(Pair data[], int n) {
	int i;
	for (i=0; i<n; i++)
		printf(" (%f %f) ", data[i].x, data[i].y);
	printf("\n");
}

// Comparison function for qsort on x
static int compare_x(const void *a, const void *b) {
    double diff = ((Pair*)a)->x - ((Pair*)b)->x;
    return (diff > 0) - (diff < 0);
}

// Kendall’s tau computation
double kendall_tau(Pair data[], int n) {

    //printf("kendall_tau  before\n");
    //print_pairs(data,n);

    // Sort by x to prepare for counting discordant y-pairs
    qsort(data, n, sizeof(Pair), compare_x);

    //printf("kendall_tau  after qsort\n");
    //print_pairs(data,n);

    // Count discordant pairs in y using modified merge sort
    Pair *temp = (Pair *)malloc(n * sizeof(Pair));
    long long discordant = merge_sort_and_count(data, temp, 0, n - 1);
    free(temp);
    
    printf("kendall_tau  discordant %lld\n", discordant);

    long long total_pairs = (long long)n * (n - 1) / 2;
    long long concordant = total_pairs - discordant;
    
    printf("kendall_tau  total_pairs %lld\n", total_pairs);
    printf("kendall_tau  concordant %lld\n", concordant);
    
    
    return (double)(concordant - discordant) / total_pairs;
}




