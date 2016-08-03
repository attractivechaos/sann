#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include "sann.h"
#include "priv.h"
#include "ksort.h"
KSORT_INIT_GENERIC(float)

void sae_core_forward(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, int k_sparse, const float *x, float *z, float *y, float *deriv1, int scaled)
{
	int j, k;
	float tmp, a1 = 1., a2 = 1.;
	const float *b1, *b2, *w10;
	if (scaled == SAE_SC_SQRT) a1 = 1. / sqrt(n_in), a2 = 1. / sqrt(n_hidden);
	else if (scaled == SAE_SC_FULL) a1 = 1. / n_in, a2 = 1. / n_hidden;
	sae_par2ptr(n_in, n_hidden, t, &b1, &b2, &w10);
	memcpy(y, b2, n_in * sizeof(float));
	for (j = 0; j < n_hidden; ++j) {
		const float *w10j = w10 + j * n_in;
		z[j] = deriv1[j] = a1 * sann_sdot(n_in, x, w10j) + b1[j];
	}
	tmp = k_sparse > 0 && k_sparse < n_hidden? ks_ksmall(float, n_hidden, deriv1, n_hidden - k_sparse) : -FLT_MAX;
	for (j = 0; j < n_hidden; ++j) {
		if (z[j] >= tmp) {
			const float *w10j = w10 + j * n_in;
			z[j] = f1(z[j], &deriv1[j]);
			sann_saxpy(n_in, a2 * z[j], w10j, y);
		} else z[j] = 0., deriv1[j] = 0.;
	}
	for (k = 0; k < n_in; ++k)
		y[k] = f2(y[k], &tmp);
}

// buf[] is at least 3*n_in+2*n_hidden in length
void sae_core_backprop(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, int k_sparse, float r, const float *x, float *d, float *buf, int scaled)
{
	int i, j, k;
	float *db1, *db2, *dw10, *out0, *out1, *out2, *delta1, *delta2, a1 = 1., a2 = 1.;
	const float *b1, *b2, *w10;
	if (scaled == SAE_SC_SQRT) a1 = 1. / sqrt(n_in), a2 = 1. / sqrt(n_hidden);
	else if (scaled == SAE_SC_FULL) a1 = 1. / n_in, a2 = 1. / n_hidden;
	// set pointers
	out0 = buf, out1 = out0 + n_in, out2 = out1 + n_hidden;
	delta1 = out2 + n_in, delta2 = delta1 + n_hidden;
	sae_par2ptr(n_in, n_hidden, t, &b1, &b2, &w10);
	sae_par2ptr(n_in, n_hidden, d, &db1, &db2, &dw10);
	// add noises to the input
	if (r > 0. && r < 1.) {
		for (i = 0; i < n_in; ++i)
			out0[i] = drand48() < r? 0. : x[i];
	} else memcpy(out0, x, n_in * sizeof(float));
	// forward calculation
	sae_core_forward(n_in, n_hidden, t, f1, f2, k_sparse, out0, out1, out2, delta1, scaled);
	// backward calculation
	for (k = 0; k < n_in; ++k)
		delta2[k] = out2[k] - x[k]; // use x, not out0
	for (j = 0; j < n_hidden; ++j) {
		const float *w10j = w10 + j * n_in;
		if (delta1[j] != 0.)
			delta1[j] *= sann_sdot(n_in, w10j, delta2); // now, delta1 is set
	}
	// update differences
	sann_saxpy(n_in, 1., delta2, db2);
	sann_saxpy(n_hidden, 1., delta1, db1);
	for (j = 0; j < n_hidden; ++j) {
		float *dw10j = dw10 + j * n_in;
		if (out1[j] != 0.)   sann_saxpy(n_in, a2 * out1[j], delta2, dw10j);
		if (delta1[j] != 0.) sann_saxpy(n_in, a1 * delta1[j], out0, dw10j); // TODO: should we use x or out0 here???
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

sae_t *sae_init(int n_in, int n_hidden, int scaled)
{
	sae_t *m;
	m = (sae_t*)calloc(1, sizeof(sae_t));
	m->n_in = n_in, m->n_hidden = n_hidden, m->k_sparse = -1;
	m->f1 = SANN_AF_SIGM;
	m->scaled = scaled;
	m->t = (float*)calloc(sae_n_par(n_in, n_hidden), sizeof(float));
	sae_core_randpar(n_in, n_hidden, m->t, scaled);
	return m;
}

void sae_destroy(sae_t *m)
{
	free(m->t); free(m);
}

float sae_run(const sae_t *m, const float *x, float *z, float *y)
{
	int i;
	float *deriv1;
	double cost;
	deriv1 = (float*)calloc(m->n_hidden, sizeof(float));
	sae_core_forward(m->n_in, m->n_hidden, m->t, sann_get_af(m->f1), sann_sigm, m->k_sparse, x, z, y, deriv1, m->scaled);
	for (cost = 0., i = 0; i < m->n_in; ++i)
		cost += sann_sigm_cost(x[i], y[i]);
	free(deriv1);
	return (float)(cost / m->n_in);
}
