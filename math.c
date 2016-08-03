#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sann.h"
#include "priv.h"

#define SANN_TINY 1e-37

float sann_sigm(float x, float *deriv)
{
	float y;
	y = 1. / (1. + expf(-x));
	*deriv = y * (1. - y);
	return y;
}

float sann_sigm_cost(float y0, float y)
{
	return - (y0 == 0.? 0. : y0 * logf(y/y0 + SANN_TINY)) - (1 - y0 == 0.? 0. : (1 - y0) * logf((1 - y) / (1 - y0) + SANN_TINY));
}

float sann_tanh(float x, float *deriv) // tanh activation function
{
	float t, y;
	t = expf(-2. * x);
	y = isinf(t)? -1. : (1. - t) / (1. + t);
	*deriv = 1. - y * y;
	return y;
}

float sann_reclin(float x, float *deriv)
{
	*deriv = x < 0.? 0. : 1.;
	return x > 0.? x : 0.;
}

sann_activate_f sann_get_af(int type)
{
	if (type == SANN_AF_SIGM) return sann_sigm;
	if (type == SANN_AF_TANH) return sann_tanh;
	if (type == SANN_AF_RECLIN) return sann_reclin;
	return 0;
}

double sann_normal(int *iset, double *gset)
{ 
	if (*iset == 0) {
		double fac, rsq, v1, v2; 
		do { 
			v1 = 2.0 * drand48() - 1.0;
			v2 = 2.0 * drand48() - 1.0; 
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq); 
		*gset = v1 * fac; 
		*iset = 1;
		return v2 * fac;
	} else {
		*iset = 0;
		return *gset;
	}
}

float sann_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}

void sann_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
