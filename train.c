#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "sann.h"
#include "priv.h"

void sann_SGD(int n, float h, float *t, float *g, sann_gradient_f func, void *data)
{
	int i;
	func(n, t, g, data);
	for (i = 0; i < n; ++i)
		t[i] -= h * g[i];
}

void sann_RMSprop(int n, float h, float decay, float *t, float *g, float *r, sann_gradient_f func, void *data)
{
	int i;
	func(n, t, g, data);
	for (i = 0; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= h / sqrt(1e-6 + r[i]) * g[i];
	}
}

void sann_tconf_init(sann_tconf_t *tc, int malgo)
{
	memset(tc, 0, sizeof(sann_tconf_t));
	if (malgo <= 0) malgo = SANN_MIN_RMSPROP;
	tc->malgo = malgo;
	tc->r = .3;
	tc->L2_par = .001;
	if (malgo == SANN_MIN_SGD) {
		tc->mini_batch = 10;
		tc->h = .1;
	} else if (malgo == SANN_MIN_RMSPROP) {
		tc->mini_batch = 50;
		tc->h = .01;
		tc->decay = .9;
	}
}

typedef struct {
	sae_t *m;
	smln_t *l;
	const sann_tconf_t *tc;
	double running_cost;
	int n;
	cfloat_p *x, *y;
	float *buf_ae;
	smln_buf_t *buf_mln;
} minibatch_t;

static void mb_gradient(int n, const float *p, float *g, void *data)
{
	minibatch_t *mb = (minibatch_t*)data;
	sae_t *m = mb->m;
	smln_t *l = mb->l;
	const sann_tconf_t *tc = mb->tc;
	int i, k;
	float t = 1. / mb->n;
	memset(g, 0, n * sizeof(float));
	assert(m == 0 || l == 0);
	for (i = 0; i < mb->n; ++i) {
		if (m) {
			sae_core_backprop(m->n_in, m->n_hidden, p, sann_get_af(m->f1), sann_sigm, m->k_sparse, tc->r, mb->x[i], g, mb->buf_ae, m->scaled);
			for (k = 0; k < m->n_in; ++k)
				mb->running_cost += sann_sigm_cost(mb->x[i][k], mb->buf_ae[m->n_in + m->n_hidden + k]);
		} else {
			smln_core_backprop(l->n_layers, l->n_neurons, l->af, p, mb->x[i], mb->y[i], g, mb->buf_mln);
			for (k = 0; k < l->n_neurons[l->n_layers-1]; ++k)
				mb->running_cost += sann_sigm_cost(mb->y[i][k], mb->buf_mln->out[l->n_layers-1][k]);
		}
	}
	if (l) {
		for (i = 0; i < n; ++i)
			g[i] = (g[i] + mb->tc->L2_par * p[i]) * t;
	} else for (i = 0; i < n; ++i) g[i] *= t;
}

float sann_train(void *model, const sann_tconf_t *tc, int n, float *const* x, float *const* y)
{
	minibatch_t mb;
	sae_t *ma = y? 0 : (sae_t*)model;
	smln_t *ml = y? (smln_t*)model : 0;
	float **aux = 0, *t;
	cfloat_p *sx = 0, *sy = 0;
	int *si, i, j, m = 0, n_aux, n_par, n_out;

	si = (int*)calloc(n, sizeof(int));
	for (i = 0; i < n; ++i) si[i] = i;
	for (i = n; i > 1; --i) { // shuffle
		int tmp;
		j = (int)(drand48() * i);
		tmp = si[j]; si[j] = si[i-1]; si[i-1] = tmp;
	}
	sx = (cfloat_p*)calloc(n, sizeof(cfloat_p));
	if (y) sy = (cfloat_p*)calloc(n, sizeof(cfloat_p));
	for (i = 0; i < n; ++i) {
		sx[i] = x[si[i]];
		if (sy) sy[i] = y[si[i]];
	}
	free(si);

	n_out = y? ml->n_neurons[ml->n_layers-1] : ma->n_in;
	n_par = y? smln_n_par(ml->n_layers, ml->n_neurons) : sae_n_par(ma->n_in, ma->n_hidden);
	t = y? ml->t : ma->t;

	n_aux = tc->malgo == SANN_MIN_RMSPROP? 2 : 1;
	aux = (float**)alloca(n_aux * sizeof(float*));
	for (i = 0; i < n_aux; ++i) aux[i] = (float*)calloc(n_par, sizeof(float));

	mb.m = ma, mb.l = ml;
	mb.tc = tc, mb.running_cost = 0.;
	mb.buf_mln = ml? smln_buf_init(ml->n_layers, ml->n_neurons, ml->t) : 0;
	mb.buf_ae = ma? (float*)malloc(sae_buf_size(ma->n_in, ma->n_hidden) * sizeof(float)) : 0;
	while (m < n) {
		mb.n = tc->mini_batch < n - m? tc->mini_batch : n - m;
		mb.x = &sx[m];
		mb.y = sy? &sy[m] : 0;
		if (tc->malgo == SANN_MIN_SGD) {
			sann_SGD(n_par, tc->h, t, aux[0], mb_gradient, &mb);
		} else if (tc->malgo == SANN_MIN_RMSPROP) {
			float h = tc->h;
			if (!y && ma->scaled != SAE_SC_NONE) h *= .5 * sqrt(ma->n_hidden < ma->n_in? ma->n_hidden : ma->n_in);
			sann_RMSprop(n_par, h, tc->decay, t, aux[0], aux[1], mb_gradient, &mb);
		}
		m += mb.n;
	}
	if (mb.buf_ae) free(mb.buf_ae);
	if (mb.buf_mln) smln_buf_destroy(mb.buf_mln);

	for (i = 0; i < n_aux; ++i) free(aux[i]);
	free(sx); free(sy);
	return mb.running_cost / n_out / n;
}
