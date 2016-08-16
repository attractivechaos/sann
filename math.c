#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sann.h"
#include "priv.h"

#define SANN_TINY 1e-9

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

#ifdef _NO_SSE
void sann_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
float sann_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
#else
#include <emmintrin.h>
float sann_sdot(int n, const float *x, const float *y)
{
	int i, n4 = n>>2<<2;
	__m128 x_vec, y_vec, s_vec;
	float s, t[4];
	s_vec = _mm_setzero_ps();
	for (i = 0; i < n4; i += 4) {
		x_vec = _mm_loadu_ps(&x[i]);
		y_vec = _mm_loadu_ps(&y[i]);
		s_vec = _mm_add_ps(s_vec, _mm_mul_ps(x_vec, y_vec));
	}
	_mm_storeu_ps(t, s_vec);
	s = t[0] + t[1] + t[2] + t[3];
	for (; i < n; ++i) s += x[i] * y[i];
	return s;
}
void sann_saxpy(int n, float a, const float *x, float *y)
{
	int i, n4 = n>>2<<2;
	__m128 x_vec, y_vec, a_vec, res_vec;
	a_vec = _mm_set1_ps(a);
	for (i = 0; i < n4; i += 4) {
		x_vec = _mm_loadu_ps(&x[i]);
		y_vec = _mm_loadu_ps(&y[i]);
		res_vec = _mm_add_ps(_mm_mul_ps(a_vec, x_vec), y_vec);
		_mm_storeu_ps(&y[i], res_vec);
	}
	for (; i < n; ++i) y[i] += a * x[i];
}
#endif
