#include <math.h>
#include <stdio.h>

//------------------------------
// 1. Frank Copula
//------------------------------
double frank_generator(double t, double theta) {
	return -log((exp(-theta * t) - 1.0) / (exp(-theta) - 1.0));
}

double frank_generator_inv(double s, double theta) {
	return - (1.0 / theta) * log(1.0 + (exp(-s) * (exp(-theta) - 1.0)));
}

//------------------------------
// 2. Gumbel Copula (θ >= 1)
//------------------------------
double gumbel_generator(double t, double theta) {
	return pow(-log(t), theta);
}

double gumbel_generator_inv(double s, double theta) {
	return exp(-pow(s, 1.0 / theta));
}
//------------------------------
// 3. AMH Copula (θ in [-1, 1))
//------------------------------
double amh_generator(double t, double theta) {
	//nom   = 1 - theta*(1-t);
	//denom = t;
	
	//return log(nom/denom);
	//return log((1.0 - theta * t) / (1.0 - theta));
	return log((1 - theta*(1-t))/t);
}

double amh_generator_inv(double s, double theta) {
	double denom = exp(s) - theta;
	if (denom ==0) return INFINITY;          // avoid devision by zero
	//return ((1-theta)/(exp(s)-theta));
	return (1-theta)/denom;
}
//------------------------------
// 4. Joe Copula (θ >= 1)
//------------------------------
double joe_generator(double t, double theta) {
	return -log(1.0 - pow(1.0 - t, theta));
}

double joe_generator_inv(double s, double theta) {
	return 1.0 - pow(1.0 - exp(-s), 1.0 / theta);
}

//------------------------------
// 5. Clayton Copula (θ > 0)
//------------------------------
double clayton_generator(double t, double theta) {
	return (pow(t, -theta) - 1.0) / theta;
}

double clayton_generator_inv(double s, double theta) {
	double power = -1.0 / theta;
	return pow(1.0 + theta * s, power);
	//return pow(1.0 + theta * s, -1.0 / theta);
} 

//------------------------------
//------------------------------
// Derivatives
//------------------------------
//------------------------------

double frank_generator_derivative(double t, double theta) {
	double log10_e     = log(10.0);
	double numerator   = theta * exp(-theta * t);
	double denominator = log10_e * (exp(-theta * t) - 1.0);
	return numerator/denominator;
}


double frank_inv_derivative(double s, double theta) {
	double log10_e     = log(10.0);
	double A           = exp(-theta) - 1.0;
	double B           = exp(-s);
	double numerator   = B * A;
	double denominator = log10_e  * (1.0 + B * A);

	if (theta == 0.0 || denominator == 0.0) {
		return NAN;                     //To avoid division by zero
	}

	return (1.0 / theta) * (numerator / denominator);
}


//------------------------------
double clayton_generator_derivative(double t, double theta) {
	if (t <= 0.0) return INFINITY;         // To avoid division by zero or negative powers
	return -pow(t, -(theta + 1.0));
}

double clayton_inv_derivative(double s, double theta) {
	double power = (-1.0 / theta) - 1.0;
	return -pow(1.0 + theta * s, power);
}

//------------------------------
double amh_generator_derivative(double t, double theta) {
	if (t == 0.0 || (1.0 - theta * (1.0 - t) / t) == 0.0) {
		return NAN;                  // To avoid division by zero
	}

	double numerator   = theta;
	double log10_e     = log(10.0);      // natural log of 10
	double denominator = t * t * log10_e * (1.0 - theta * (1.0 - t) / t);

	return numerator / denominator;
}


double amh_inv_derivative(double s, double theta) {
	double numerator   = -exp(s)*(1-theta);
	double denominator = (exp(s)-theta)*(exp(s)-theta);

	return numerator / denominator;
}


//------------------------------

double joe_generator_derivative(double t, double theta) {
	double log10_e     = log(10.0);
	double one_minus_t = 1.0 - t;

	if (t == 1.0 || one_minus_t <= 0.0) {
		return NAN;                //To avoid log(0), pow of negative, or division by zero
	}

	double numerator   = theta * pow(one_minus_t, theta - 1.0);
	double denominator = log10_e * (1.0 - pow(one_minus_t, theta));

	if (denominator == 0.0) {
		return NAN;
	}

	return - numerator / denominator;
}




double joe_inv_derivative(double s, double theta) {
	if (theta == 0.0) return NAN;        //To avoid division by zero

	double one_minus_exp = 1.0 - exp(-s);

	if (one_minus_exp <= 0.0) {
		return NAN;                 //To avoid invalid pow for negative or zero base
	}

	double exponent = (1.0 / theta) - 1.0;
	double value = pow(one_minus_exp, exponent);

	return -exp(-theta) / theta * value;
}








