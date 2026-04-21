#ifndef __ARCHIMEDIAN_H__
#define __ARCHIMEDIAN_H__

#include <stdio.h>
#include <stdlib.h>

//-------------------------------------------------------------------
// High level interface
//-------------------------------------------------------------------
#define ARCHI_AMH     0
#define ARCHI_CLAYTON 1
#define ARCHI_FRANK   2
#define ARCHI_GUMBEL  3
#define ARCHI_JOE     4
#define NUM_ARCHI 5

extern const char *str_archi[NUM_ARCHI];

double archi_theta(int archi, double rho, double tau);
double archi_copula(int archi, double theta, double x, double y);
double archi_copula_density(int archi, double theta, double x, double y);


//-------------------------------------------------------------------
// Copula Generators, Inverses, and Derivatives
//-------------------------------------------------------------------

// Declare all functions (you can move this to a header file if modularizing)
double frank_generator(double t, double theta);
double frank_generator_inv(double s, double theta);
double frank_generator_derivative(double t, double theta);
double frank_inv_derivative(double s, double theta);

double gumbel_generator(double t, double theta);
double gumbel_generator_inv(double s, double theta);
double gumbel_generator_derivative(double t, double theta);
double gumbel_inv_derivative(double s, double theta);

double amh_generator(double t, double theta);
double amh_generator_inv(double s, double theta);
double amh_generator_derivative(double t, double theta);
double amh_inv_derivative(double s, double theta);

double joe_generator(double t, double theta);
double joe_generator_inv(double s, double theta);
double joe_generator_derivative(double t, double theta);
double joe_inv_derivative(double s, double theta);

double clayton_generator(double t, double theta);
double clayton_generator_inv(double s, double theta);
double clayton_generator_derivative(double t, double theta);
double clayton_inv_derivative(double s, double theta);


//-------------------------------------------------------------------
// Estimation of Copula theta parameter
//-------------------------------------------------------------------


double joe_tau(double theta, int iter);
double joe_theta(double tau);
double clayton_theta(double tau);
double frank_tau(double theta);
double frank_theta(double tau);
double gumbel_theta(double tau);
double amh_theta(double rho);



#define FRANK_NSTEPS 200000
#define FRANK_THETA0 0.0
#define FRANK_THETA1 20.0
extern double g_frank_integral[FRANK_NSTEPS+1];  // Trapezoid rule
extern int    g_frank_integral_initialized;


//-------------------------------------------------------------------
// Estimation of Kendall's tau
//-------------------------------------------------------------------

typedef struct {
    double x;
    double y;
} Pair;

double kendall_tau(Pair data[], int n);

#endif // _ARCHIMEDIAN_H_


