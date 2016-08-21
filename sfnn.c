#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include "sann_priv.h"

int sfnn_n_par(int n_layers, const int32_t *n_neurons)
{
	int k, n_par = 0;
	for (k = 1; k < n_layers; ++k)
		n_par += n_neurons[k] * (n_neurons[k-1] + 1);
	return n_par;
}

#define sfnn_par2ptr(type_t, n_layers, n_neurons, _t, _w, _b) do { \
		int _k; \
		type_t _q = (_t); \
		if ((_w) == 0) (_w) = (type_t*)calloc((n_layers), sizeof(type_t)); \
		if ((_b) == 0) (_b) = (type_t*)calloc((n_layers), sizeof(type_t)); \
		for (_k = 1; _k < (n_layers); ++_k) { \
			(_b)[_k] = _q, _q += (n_neurons)[_k]; \
			(_w)[_k] = _q, _q += (n_neurons)[_k] * (n_neurons)[_k-1]; \
		} \
	} while (0)

sfnn_buf_t *sfnn_buf_init(int n_layers, const int32_t *n_neurons, cfloat_p t)
{
	int k, sum_neurons = 0;
	sfnn_buf_t *b;
	float *p;

	b = (sfnn_buf_t*)calloc(1, sizeof(sfnn_buf_t));
	b->out = (float**)calloc(n_layers * 5, sizeof(float*));
	b->deriv = b->out + n_layers;
	b->delta = b->deriv + n_layers;
	b->dw = b->delta + n_layers;
	b->db = b->dw + n_layers;

	sfnn_par2ptr(cfloat_p, n_layers, n_neurons, t, b->w, b->b);
	for (k = 1; k < n_layers; ++k)
		sum_neurons += n_neurons[k];
	p = b->buf = (float*)calloc(n_neurons[0] + sum_neurons * 3, sizeof(float));
	b->out[0] = p, p += n_neurons[0]; // ->deriv[0] and ->delta[0] are not allocated
	for (k = 1; k < n_layers; ++k) {
		b->out[k] = p, p += n_neurons[k];
		b->deriv[k] = p, p += n_neurons[k];
		b->delta[k] = p, p += n_neurons[k];
	}
	return b;
}

void sfnn_buf_destroy(sfnn_buf_t *b)
{
	free(b->w); free(b->b); free(b->out); free(b->buf); free(b);
}

void sfnn_core_forward(int n_layers, const int32_t *n_neurons, const int32_t *af, float r_in, float r_hidden, cfloat_p t, cfloat_p x, sfnn_buf_t *b)
{
	int i, j, k;
	float q[2] = { 1.0f / (1.0f - r_in), 1.0f / (1.0f - r_hidden) };
	if (r_in > 0.0f && r_in < 1.0f) {
		for (i = 0; i < n_neurons[0]; ++i)
			b->out[0][i] = drand48() < r_in? 0.0f : x[i];
	} else memcpy(b->out[0], x, n_neurons[0] * sizeof(float));
	for (k = 1; k < n_layers; ++k) {
		sann_activate_f func = sann_get_af(af[k-1]);
		for (j = 0; j < n_neurons[k]; ++j)
			if (k < n_layers - 1 && r_hidden > 0.0f && drand48() < r_hidden)
				b->out[k][j] = b->deriv[k][j] = 0.0f;
			else b->out[k][j] = func(q[k>1] * sann_sdot(n_neurons[k-1], b->w[k] + j * n_neurons[k-1], b->out[k-1]) + b->b[k][j], &b->deriv[k][j]);
	}
}

void sfnn_core_backward(int n_layers, const int32_t *n_neurons, float r_in, float r_hidden, cfloat_p y, float *g, sfnn_buf_t *b)
{
	int i, j, k;
	float q[2] = { 1.0f / (1.0f - r_in), 1.0f / (1.0f - r_hidden) };
	for (j = 0, k = n_layers - 1; j < n_neurons[k]; ++j) // calculate delta[] at the output layer
		b->delta[k][j] = b->out[k][j] - y[j];
	for (k = n_layers - 1; k > 1; --k) { // calculate delta[k-1]
		memset(b->delta[k-1], 0, n_neurons[k-1] * sizeof(float));
		for (j = 0; j < n_neurons[k]; ++j)
			sann_saxpy(n_neurons[k-1], q[1] * b->delta[k][j], b->w[k] + j * n_neurons[k-1], b->delta[k-1]);
		for (i = 0; i < n_neurons[k-1]; ++i)
			b->delta[k-1][i] *= b->deriv[k-1][i];
	}
	sfnn_par2ptr(float_p, n_layers, n_neurons, g, b->dw, b->db);
	for (k = 1; k < n_layers; ++k) { // update gradiant
		sann_saxpy(n_neurons[k], 1., b->delta[k], b->db[k]);
		for (j = 0; j < n_neurons[k]; ++j)
			sann_saxpy(n_neurons[k-1], q[k>1] * b->delta[k][j], b->out[k-1], b->dw[k] + j * n_neurons[k-1]);
	}
}

void sfnn_core_backprop(int n_layers, const int32_t *n_neurons, const int32_t *af, float r_in, float r_hidden, cfloat_p t, cfloat_p x, cfloat_p y, float *g, sfnn_buf_t *b)
{
	assert(af[n_layers-2] == SANN_AF_SIGM);
	sfnn_core_forward(n_layers, n_neurons, af, r_in, r_hidden, t, x, b);
	sfnn_core_backward(n_layers, n_neurons, r_in, r_hidden, y, g, b);
}

void sfnn_core_randpar(int n_layers, const int32_t *n_neurons, float *t)
{
	float **b = 0, **w = 0;
	int k, j, iset = 0;
	double gset;
	sfnn_par2ptr(float_p, n_layers, n_neurons, t, w, b);
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

void sfnn_core_jacobian(int n_layers, const int32_t *n_neurons, const int32_t *af, cfloat_p t, cfloat_p x, int w, float *d, sfnn_buf_t *b)
{
	int i, j, k;
	sfnn_core_forward(n_layers, n_neurons, af, 0.0f, 0.0f, t, x, b);
	memset(b->delta[n_layers-1], 0, n_neurons[n_layers-1] * sizeof(float));
	b->delta[n_layers-1][w] = 1.0f;
	for (k = n_layers - 1; k > 1; --k) { // calculate delta[k-1]
		memset(b->delta[k-1], 0, n_neurons[k-1] * sizeof(float));
		for (j = 0; j < n_neurons[k]; ++j)
			sann_saxpy(n_neurons[k-1], b->delta[k][j], b->w[k] + j * n_neurons[k-1], b->delta[k-1]);
		for (i = 0; i < n_neurons[k-1]; ++i)
			b->delta[k-1][i] *= b->deriv[k-1][i];
	}
	for (j = 0; j < n_neurons[1]; ++j)
		sann_saxpy(n_neurons[0], b->delta[1][j], b->w[1] + j * n_neurons[0], d);
}
