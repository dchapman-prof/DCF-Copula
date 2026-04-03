#include <math.h>
#include <stdio.h>

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
	double log10_e     = log(10.0);
	double numerator   = theta * exp(-theta * t);
	double denominator = log10_e * (exp(-theta * t) - 1.0);
	return numerator/denominator;
}

//------------------------------
// Inverse Derivative
//------------------------------
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
//------------------------------
// 2. Gumbel Copula 
//------------------------------
//------------------------------

//------------------------------
// Generator
//-----------------------------
double gumbel_generator(double t, double theta) {
	return pow(-log(t), theta);
}


//------------------------------
// Generator Inverse
//------------------------------
double gumbel_generator_inv(double s, double theta) {
	return exp(-pow(s, 1.0 / theta));
}

//------------------------------
// Generator Derivative
//------------------------------
double gumbel_generator_derivative(double t, double theta) {
	if (t <= 0.0 || log(t) == 0.0) {
		return NAN;                //To avoid log(0) or division by zero
	}

	double log10_e     = log(10.0);    // ln(10)
	double ln_t        = log(t);       // natural log of t

	double numerator   = theta * pow(ln_t, theta - 1.0);
	double denominator = log10_e * t;

	return numerator / denominator;
}

//------------------------------
// Inverse Derivative
//------------------------------

double gumbel_inv_derivative(double s, double theta) {
	if (theta == 0.0) return NAN;  // avoid division by zero

	double result = - (1.0 / theta) * exp(-s / theta);
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
	return log((1 - theta*(1-t))/t);
}
//------------------------------
// Generator Inverse
//------------------------------
double amh_generator_inv(double s, double theta) {
	double denominator = exp(s) - theta;
	if (denominator ==0) return INFINITY;          // avoid devision by zero
	//return ((1-theta)/(exp(s)-theta));
	return (1-theta)/denominator;
}

//------------------------------
// Generator Derivative
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

//------------------------------
// Inverse Derivative
//------------------------------
double amh_inv_derivative(double s, double theta) {
	double numerator   = -exp(s)*(1-theta);
	double denominator = (exp(s)-theta)*(exp(s)-theta);

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

//------------------------------
// Inverse Derivative
//------------------------------
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
	double power = (-1.0 / theta) - 1.0;
	return -pow(1.0 + theta * s, power);
}




// Include all your copula functions here or in a separate header file

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











