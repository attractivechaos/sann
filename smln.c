#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include "sann.h"
#include "priv.h"

int smln_n_par(int n_layers, const int32_t *n_neurons)
{
	int k, n_par = 0;
	for (k = 1; k < n_layers; ++k)
		n_par += n_neurons[k] * (n_neurons[k-1] + 1);
	return n_par;
}

#define smln_par2ptr(type_t, n_layers, n_neurons, _t, _w, _b) do { \
		int _k; \
		type_t _q = (_t); \
		if ((_w) == 0) (_w) = (type_t*)calloc((n_layers), sizeof(type_t)); \
		if ((_b) == 0) (_b) = (type_t*)calloc((n_layers), sizeof(type_t)); \
		for (_k = 1; _k < (n_layers); ++_k) { \
			(_b)[_k] = _q, _q += (n_neurons)[_k]; \
			(_w)[_k] = _q, _q += (n_neurons)[_k] * (n_neurons)[_k-1]; \
		} \
	} while (0)

smln_buf_t *smln_buf_init(int n_layers, const int32_t *n_neurons, cfloat_p t)
{
	int k, sum_neurons = 0, n_par;
	smln_buf_t *b;
	float *p;

	b = (smln_buf_t*)calloc(1, sizeof(smln_buf_t));
	smln_par2ptr(cfloat_p, n_layers, n_neurons, t, b->w, b->b);
	n_par = smln_n_par(n_layers, n_neurons);
	for (k = 1; k < n_layers; ++k)
		sum_neurons += n_neurons[k];
	p = b->buf = (float*)calloc(n_neurons[0] + sum_neurons * 3, sizeof(float));
	b->out = (float**)calloc(n_layers, sizeof(float*));
	b->deriv = (float**)calloc(n_layers, sizeof(float*));
	b->delta = (float**)calloc(n_layers, sizeof(float*));
	b->out[0] = p, p += n_neurons[0]; // ->deriv[0] and ->delta[0] are not allocated
	for (k = 1; k < n_layers; ++k) {
		b->out[k] = p, p += n_neurons[k];
		b->deriv[k] = p, p += n_neurons[k];
		b->delta[k] = p, p += n_neurons[k];
	}
	b->dw = (float**)calloc(n_layers, sizeof(float*));
	b->db = (float**)calloc(n_layers, sizeof(float*));
	return b;
}

void smln_buf_destroy(smln_buf_t *b)
{
	free(b->w); free(b->b); free(b->dw); free(b->db);
	free(b->out); free(b->deriv); free(b->delta);
	free(b->buf); free(b);
}

void smln_core_forward(int n_layers, const int32_t *n_neurons, const int32_t *af, cfloat_p t, cfloat_p x, smln_buf_t *b)
{
	int j, k;
	memcpy(b->out[0], x, n_neurons[0] * sizeof(float));
	for (k = 1; k < n_layers; ++k) {
		sann_activate_f func = sann_get_af(af[k-1]);
		for (j = 0; j < n_neurons[k]; ++j)
			b->out[k][j] = func(sann_sdot(n_neurons[k-1], b->w[k] + j * n_neurons[k-1], b->out[k-1]) + b->b[k][j], &b->deriv[k][j]);
	}
}

void smln_core_backward(int n_layers, const int32_t *n_neurons, cfloat_p y, float *g, smln_buf_t *b)
{
	int i, j, k;
	for (j = 0, k = n_layers - 1; j < n_neurons[k]; ++j) // calculate delta[] at the output layer
		b->delta[k][j] = b->out[k][j] - y[j];
	for (k = n_layers - 1; k > 1; --k) { // calculate delta[k-1]
		memset(b->delta[k-1], 0, n_neurons[k-1] * sizeof(float));
		for (j = 0; j < n_neurons[k]; ++j)
			sann_saxpy(n_neurons[k-1], b->delta[k][j], b->w[k] + j * n_neurons[k-1], b->delta[k-1]);
		for (i = 0; i < n_neurons[k-1]; ++i)
			b->delta[k-1][i] *= b->deriv[k-1][i];
	}
	smln_par2ptr(float_p, n_layers, n_neurons, g, b->dw, b->db);
	for (k = 1; k < n_layers; ++k) { // update gradiant
		sann_saxpy(n_neurons[k], 1., b->delta[k], b->db[k]);
		for (j = 0; j < n_neurons[k]; ++j)
			sann_saxpy(n_neurons[k-1], b->delta[k][j], b->out[k-1], b->dw[k] + j * n_neurons[k-1]);
	}
}

void smln_core_backprop(int n_layers, const int32_t *n_neurons, const int32_t *af, cfloat_p t, cfloat_p x, cfloat_p y, float *g, smln_buf_t *b)
{
	assert(af[n_layers-2] == SANN_AF_SIGM);
	smln_core_forward(n_layers, n_neurons, af, t, x, b);
	smln_core_backward(n_layers, n_neurons, y, g, b);
}

void smln_core_randpar(int n_layers, const int32_t *n_neurons, float *t)
{
	float **b = 0, **w = 0;
	int k, j, iset = 0;
	double gset;
	smln_par2ptr(float_p, n_layers, n_neurons, t, w, b);
	for (k = 1; k < n_layers; ++k) {
		float t;
		int tmp = n_neurons[k-1] * n_neurons[k];
		t = sqrt(n_neurons[k-1] + 1);
		for (j = 0; j < n_neurons[k]; ++j)
			b[k][j] = sann_normal(&iset, &gset) / t;
		for (j = 0; j < tmp; ++j)
			w[k][j] = sann_normal(&iset, &gset) / t;
	}
	free(b); free(w);
}
