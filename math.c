#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "sann.h"
#include "sann_priv.h"
#ifdef __SSE__
#include <xmmintrin.h>
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

#ifdef __SSE__
float sann_sdot(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}
void sann_saxpy(int n, float a, const float *x, float *y)
{
	int i, n8 = n>>3<<3;
	__m128 va;
	va = _mm_set1_ps(a);
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2, vt1, vt2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vt1 = _mm_add_ps(_mm_mul_ps(va, vx1), vy1);
		vt2 = _mm_add_ps(_mm_mul_ps(va, vx2), vy2);
		_mm_storeu_ps(&y[i], vt1);
		_mm_storeu_ps(&y[i+4], vt2);
	}
	for (; i < n; ++i) y[i] += a * x[i];
}
#else
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

#ifdef __SSE__
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
#else
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
#endif
