#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include "sann_priv.h"

void sae_core_forward(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, float r, const float *x, float *z, float *y, float *deriv1, int scaled)
{
	int j, k;
	float tmp, a01 = 1., a12 = 1.;
	const float *b1, *b2, *w10;
	if (scaled == SAE_SC_SQRT) a01 = 1. / sqrt(n_in), a12 = 1. / sqrt(n_hidden);
	else if (scaled == SAE_SC_FULL) a01 = 1. / n_in, a12 = 1. / n_hidden;
	a01 *= 1.0f / (1.0f - r);
	sae_par2ptr(n_in, n_hidden, t, &b1, &b2, &w10);
	memcpy(y, b2, n_in * sizeof(float));
	for (j = 0; j < n_hidden; ++j) {
		const float *w10j = w10 + j * n_in;
		tmp = a01 * sann_sdot(n_in, x, w10j) + b1[j];
		z[j] = f1(tmp, &deriv1[j]);
		sann_saxpy(n_in, a12 * z[j], w10j, y);
	}
	for (k = 0; k < n_in; ++k)
		y[k] = f2(y[k], &tmp);
}

// buf[] is at least 3*n_in+2*n_hidden in length
void sae_core_backprop(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, float r, const float *x, float *d, float *buf, int scaled)
{
	int i, j, k;
	float *db1, *db2, *dw10, *out0, *out1, *out2, *delta1, *delta2, a01 = 1., a12 = 1.;
	const float *b1, *b2, *w10;
	if (scaled == SAE_SC_SQRT) a01 = 1. / sqrt(n_in), a12 = 1. / sqrt(n_hidden);
	else if (scaled == SAE_SC_FULL) a01 = 1. / n_in, a12 = 1. / n_hidden;
	a01 *= 1.0f / (1.0f - r);
	// set pointers
	out0 = buf, out1 = out0 + n_in, out2 = out1 + n_hidden;
	delta1 = out2 + n_in, delta2 = delta1 + n_hidden;
	sae_par2ptr(n_in, n_hidden, t, &b1, &b2, &w10);
	sae_par2ptr(n_in, n_hidden, d, &db1, &db2, &dw10);
	// add noises to the input
	if (r > 0. && r < 1.) {
		for (i = 0; i < n_in; ++i)
			out0[i] = sann_drand() < r? 0. : x[i];
	} else memcpy(out0, x, n_in * sizeof(float));
	// forward calculation
	sae_core_forward(n_in, n_hidden, t, f1, f2, r, out0, out1, out2, delta1, scaled);
	// backward calculation
	for (k = 0; k < n_in; ++k) // delta at the output layer
		delta2[k] = out2[k] - x[k]; // use x, not out0
	for (j = 0; j < n_hidden; ++j) { // delta at the hidden layer
		const float *w10j = w10 + j * n_in;
		delta1[j] *= a12 * sann_sdot(n_in, w10j, delta2); // now, delta1 is set
	}
	// update differences
	sann_saxpy(n_in, 1., delta2, db2);
	sann_saxpy(n_hidden, 1., delta1, db1);
	for (j = 0; j < n_hidden; ++j) {
		float *dw10j = dw10 + j * n_in;
		sann_saxpy(n_in, a12 * out1[j], delta2, dw10j);
		sann_saxpy(n_in, a01 * delta1[j], out0, dw10j);
	}
}

void sae_core_randpar(int n_in, int n_hidden, float *t, int scaled)
{
	float *b1, *b2, *w10;
	int i, iset = 0;
	double gset;
	sae_par2ptr(n_in, n_hidden, t, &b1, &b2, &w10);
	if (scaled != SAE_SC_NONE) {
		memset(b1, 0, n_hidden * sizeof(float));
		memset(b2, 0, n_in * sizeof(float));
		for (i = 0; i < n_in * n_hidden; ++i)
			w10[i] = sann_normal(&iset, &gset);
	} else {
		int n_par = sae_n_par(n_in, n_hidden);
		for (i = 0; i < n_hidden; ++i)
			b1[i] = n_hidden + 1.;
		for (i = 0; i < n_in; ++i)
			b2[i] = n_in + 1.;
		for (i = 0; i < n_in * n_hidden; ++i)
			w10[i] = n_in + 1.;
		for (i = 0; i < n_par; ++i)
			t[i] = sann_normal(&iset, &gset) / sqrt(t[i]);
	}
}
