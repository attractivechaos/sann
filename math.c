#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sann.h"
#include "priv.h"
#ifndef _NO_SSE
#include <emmintrin.h>
#endif

/************************
 * Activation functions *
 ************************/

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

/****************
 * Gaussian RNG *
 ****************/

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

/*****************
 * BLAS routines *
 *****************/

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

/********************
 * SGD and variants *
 ********************/

void sann_SGD(int n, float h, float *t, float *g, sann_gradient_f func, void *data)
{
	int i;
	func(n, t, g, data);
	for (i = 0; i < n; ++i)
		t[i] -= h * g[i];
}

#ifdef _NO_SSE
void sann_RMSprop(int n, float h0, const float *h, float decay, float *t, float *g, float *r, sann_gradient_f func, void *data)
{
	int i;
	func(n, t, g, data);
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= lr / sqrt(1e-6 + r[i]) * g[i];
	}
}
#else
void sann_RMSprop(int n, float h0, const float *h, float decay, float *t, float *g, float *r, sann_gradient_f func, void *data)
{
	int i, n4 = n>>2<<2;
	__m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
	vh = _mm_set1_ps(h0);
	vd = _mm_set1_ps(decay);
	vd1 = _mm_set1_ps(1.0f - decay);
	vtiny = _mm_set1_ps(1e-6f);
	func(n, t, g, data);
	for (i = 0; i < n4; i += 4) {
		vt = _mm_loadu_ps(&t[i]);
		vr = _mm_loadu_ps(&r[i]);
		vg = _mm_loadu_ps(&g[i]);
		if (h) vh = _mm_loadu_ps(&h[i]);
		vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
		_mm_storeu_ps(&r[i], vr);
		tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
		_mm_storeu_ps(&t[i], tmp);
	}
	for (; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= (h? h[i] : h0) / sqrt(1e-6 + r[i]) * g[i];
	}
}
#endif
