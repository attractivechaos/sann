#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "priv.h"
#include "sann.h"

int sann_verbose = 3;

#define sae_buf_size(n_in, n_hidden) (3 * (n_in) + 2 * (n_hidden))

sann_t *sann_init_ae(int n_in, int n_hidden, int scaled)
{
	sann_t *m;
	m = (sann_t*)calloc(1, sizeof(sann_t));
	m->is_mln = 0;
	m->k_sparse = -1, m->scaled = scaled; // AE-specific parameters
	m->n_layers = 3;
	m->n_neurons = (int32_t*)calloc(m->n_layers, 4);
	m->n_neurons[0] = m->n_neurons[2] = n_in;
	m->n_neurons[1] = n_hidden;
	m->af = (int32_t*)calloc(2, 4);
	m->af[0] = m->af[1] = SANN_AF_SIGM;
	m->t = (float*)calloc(sae_n_par(n_in, n_hidden), sizeof(float));
	sae_core_randpar(n_in, n_hidden, m->t, scaled);
	return m;
}

sann_t *sann_init_mln(int n_layers, const int *n_neurons)
{
	int i;
	sann_t *m;
	m = (sann_t*)calloc(1, sizeof(sann_t));
	m->is_mln = 1;
	m->n_layers = n_layers;
	m->n_neurons = (int32_t*)calloc(n_layers, 4);
	for (i = 0; i < n_layers; ++i) m->n_neurons[i] = n_neurons[i];
	m->af = (int32_t*)calloc(n_layers - 1, 4);
	for (i = 0; i < n_layers - 2; ++i) m->af[i] = SANN_AF_RECLIN;
	m->af[i] = SANN_AF_SIGM;
	m->t = (float*)calloc(smln_n_par(m->n_layers, m->n_neurons), sizeof(float));
	smln_core_randpar(m->n_layers, m->n_neurons, m->t);
	return m;
}

int sann_n_par(const sann_t *m)
{
	return m->is_mln? smln_n_par(m->n_layers, m->n_neurons) : sae_n_par(m->n_neurons[0], m->n_neurons[1]);
}

void sann_cpy(sann_t *d, const sann_t *m)
{
	d->is_mln = m->is_mln, d->k_sparse = m->k_sparse, d->scaled = m->scaled, d->n_layers = m->n_layers;
	d->n_neurons = (int32_t*)realloc(d->n_neurons, m->n_layers * 4);
	memcpy(d->n_neurons, m->n_neurons, m->n_layers * 4);
	d->af = (int32_t*)realloc(d->af, (m->n_layers - 1) * 4);
	memcpy(d->af, m->af, (m->n_layers - 1) * 4);
	d->t = (float*)realloc(d->t, sann_n_par(m) * sizeof(float));
	memcpy(d->t, m->t, sann_n_par(m) * sizeof(float));
}

sann_t *sann_dup(const sann_t *m)
{
	sann_t *d;
	d = (sann_t*)calloc(1, sizeof(sann_t));
	sann_cpy(d, m);
	return d;
}

void sann_destroy(sann_t *m)
{
	if (m == 0) return;
	free(m->n_neurons); free(m->af); free(m->t); free(m);
}

float sann_apply(const sann_t *m, const float *x, float *y, float *z)
{
	if (m->is_mln) {
		smln_buf_t *b;
		b = smln_buf_init(m->n_layers, m->n_neurons, m->t);
		smln_core_forward(m->n_layers, m->n_neurons, m->af, m->t, x, b);
		memcpy(y, b->out[m->n_layers-1], m->n_neurons[m->n_layers-1] * sizeof(float));
		smln_buf_destroy(b);
		return 0;
	} else {
		int i;
		float *deriv1;
		double cost;
		deriv1 = (float*)calloc(m->n_neurons[1], sizeof(float));
		sae_core_forward(sae_n_in(m), sae_n_hidden(m), m->t, sann_get_af(m->af[0]), sann_sigm, m->k_sparse, x, z, y, deriv1, m->scaled);
		for (cost = 0., i = 0; i < sann_n_out(m); ++i)
			cost += sann_sigm_cost(x[i], y[i]);
		free(deriv1);
		return (float)(cost / sann_n_out(m));
	}
}

/************
 * Training *
 ************/

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
		tc->h = .01;
	} else if (malgo == SANN_MIN_RMSPROP) {
		tc->mini_batch = 50;
		tc->h = .001;
		tc->decay = .9;
	}
}

typedef struct {
	sann_t *m;
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
	sann_t *m = mb->m;
	const sann_tconf_t *tc = mb->tc;
	int i, k;
	float t = 1. / mb->n;
	memset(g, 0, n * sizeof(float));
	for (i = 0; i < mb->n; ++i) {
		if (!m->is_mln) {
			sae_core_backprop(m->n_neurons[0], m->n_neurons[1], p, sann_get_af(m->af[0]), sann_sigm, m->k_sparse, tc->r, mb->x[i], g, mb->buf_ae, m->scaled);
			for (k = 0; k < m->n_neurons[0]; ++k)
				mb->running_cost += sann_sigm_cost(mb->x[i][k], mb->buf_ae[sae_n_in(m) + sae_n_hidden(m) + k]);
		} else {
			smln_core_backprop(m->n_layers, m->n_neurons, m->af, p, mb->x[i], mb->y[i], g, mb->buf_mln);
			for (k = 0; k < m->n_neurons[m->n_layers-1]; ++k)
				mb->running_cost += sann_sigm_cost(mb->y[i][k], mb->buf_mln->out[m->n_layers-1][k]);
		}
	}
	if (m->is_mln) {
		for (i = 0; i < n; ++i)
			g[i] = (g[i] + mb->tc->L2_par * p[i]) * t;
	} else for (i = 0; i < n; ++i) g[i] *= t;
}

float sann_train1(sann_t *m, const sann_tconf_t *tc, int n, float *const* x, float *const* y)
{
	minibatch_t mb;
	float **aux = 0;
	cfloat_p *sx = 0, *sy = 0;
	int *si, i, j, mn = 0, n_aux, n_par, n_out;

	si = (int*)calloc(n, sizeof(int));
	for (i = 0; i < n; ++i) si[i] = i;
	for (i = n; i > 1; --i) { // shuffle
		int tmp;
		j = (int)(drand48() * i);
		tmp = si[j]; si[j] = si[i-1]; si[i-1] = tmp;
	}
	sx = (cfloat_p*)calloc(n, sizeof(cfloat_p));
	if (m->is_mln) sy = (cfloat_p*)calloc(n, sizeof(cfloat_p));
	for (i = 0; i < n; ++i) {
		sx[i] = x[si[i]];
		if (sy) sy[i] = y[si[i]];
	}
	free(si);

	n_out = m->n_neurons[m->n_layers - 1];
	n_par = sann_n_par(m);

	n_aux = tc->malgo == SANN_MIN_RMSPROP? 2 : 1;
	aux = (float**)alloca(n_aux * sizeof(float*));
	for (i = 0; i < n_aux; ++i) aux[i] = (float*)calloc(n_par, sizeof(float));

	mb.m = m, mb.tc = tc, mb.running_cost = 0.;
	mb.buf_mln = m->is_mln? smln_buf_init(m->n_layers, m->n_neurons, m->t) : 0;
	mb.buf_ae = !m->is_mln? (float*)malloc(sae_buf_size(sae_n_in(m), sae_n_hidden(m)) * sizeof(float)) : 0;
	while (mn < n) {
		mb.n = tc->mini_batch < n - mn? tc->mini_batch : n - mn;
		mb.x = &sx[mn];
		mb.y = sy? &sy[mn] : 0;
		if (tc->malgo == SANN_MIN_SGD) {
			sann_SGD(n_par, tc->h, m->t, aux[0], mb_gradient, &mb);
		} else if (tc->malgo == SANN_MIN_RMSPROP) {
			sann_RMSprop(n_par, tc->h, tc->decay, m->t, aux[0], aux[1], mb_gradient, &mb);
		}
		mn += mb.n;
	}
	if (mb.buf_ae) free(mb.buf_ae);
	if (mb.buf_mln) smln_buf_destroy(mb.buf_mln);

	for (i = 0; i < n_aux; ++i) free(aux[i]);
	free(sx); free(sy);
	return mb.running_cost / n_out / n;
}

/*************************
 * Train for many epochs *
 *************************/

#define SANN_TRAIN_FUZZY .005

int sann_train(sann_t *m, const sann_tconf_t *_tc, float min_h, float max_h, int n_epochs, int n, float *const* x, float *const* y)
{
	sann_tconf_t tc = *_tc;
	int k, best_epoch = -1, best_past = -1;
	sann_t *best_m;
	float last_rc = -1, best_rc = -1, best_next_h = -1;
	best_m = sann_dup(m);
	tc.h = sqrt(min_h * max_h);
	for (k = 0; k < n_epochs; ++k) {
		float rc, old_h = tc.h;
		rc = sann_train1(m, &tc, n, x, y);
		if (sann_verbose >= 3)
			fprintf(stderr, "[M::%s] epoch = %d learning_rate = %g running_cost = %g\n", __func__, k+1, old_h, rc);
		if (k == 0) {
			best_rc = rc, best_next_h = tc.h, best_epoch = 0, best_past = 0;
			sann_cpy(best_m, m);
		} else {
			if (rc > best_rc * 1.1 || best_past >= 5) { // revert to the previous best
				sann_cpy(m, best_m);
				tc.h = best_next_h, best_next_h *= .5, best_past = 0;
				rc = best_rc;
				if (sann_verbose >= 3)
					fprintf(stderr, "[M::%s] revert to the model at epoch = %d\n", __func__, best_epoch);
			} else {
				float r = rc / last_rc, f = 1.;
				if (rc < best_rc) {
					sann_cpy(best_m, m);
					best_rc = rc, best_next_h = tc.h, best_epoch = k, best_past = 0;
				} else ++best_past;
				if (r > 1 + SANN_TRAIN_FUZZY) f = 1. / (1. + r);
				else if (r > 1 - SANN_TRAIN_FUZZY) f = 1. / (1. + (r - (1 - SANN_TRAIN_FUZZY)) / (SANN_TRAIN_FUZZY*20));
				else if (r < 1 - SANN_TRAIN_FUZZY) f = 1. + ((1 - SANN_TRAIN_FUZZY) - r) / (SANN_TRAIN_FUZZY*20);
				tc.h *= f;
			}
		}
		if (tc.h > max_h) tc.h = max_h;
		if (tc.h < min_h) tc.h = min_h;
		last_rc = rc;
	}
	sann_cpy(m, best_m);
	sann_destroy(best_m);
	return 0;
}

/**********************
 * Dump/restore model *
 **********************/

#define SANN_MAGIC "SAN\1"

int sann_dump(const char *fn, const sann_t *m, char *const* col_names)
{
	FILE *fp;
	int i, n_par;

	n_par = sann_n_par(m);
	fp = fn && strcmp(fn, "-")? fopen(fn, "w") : stdout;
	if (fp == 0) return -1;
	fwrite(SANN_MAGIC, 1, 4, fp);
	fwrite(&m->is_mln, 4, 1, fp);
	fwrite(&m->k_sparse, 4, 1, fp);
	fwrite(&m->scaled, 4, 1, fp);
	fwrite(&m->n_layers, 4, 1, fp);
	fwrite(m->n_neurons, 4, m->n_layers, fp);
	fwrite(m->af, 4, m->n_layers - 1, fp);
	fwrite(m->t, sizeof(float), n_par, fp);
	if (col_names) {
		uint64_t tot_len = 0;
		for (i = 0; i < sann_n_in(m); ++i)
			tot_len += strlen(col_names[i]) + 1;
		fwrite(&tot_len, 8, 1, fp);
		for (i = 0; i < sann_n_in(m); ++i)
			fwrite(col_names[i], 1, strlen(col_names[i]) + 1, fp);
	}
	if (fp != stdout) fclose(fp);
	return 0;
}

sann_t *sann_restore(const char *fn, char ***col_names)
{
	FILE *fp;
	char magic[4];
	sann_t *m;
	int i, n_par;
	uint64_t tot_len;

	fp = fn && strcmp(fn, "-")? fopen(fn, "r") : stdin;
	if (fp == 0) return 0;
	fread(magic, 1, 4, fp);
	if (strncmp(magic, SANN_MAGIC, 4) != 0) return 0;
	m = (sann_t*)calloc(1, sizeof(sann_t));
	fread(&m->is_mln, 4, 1, fp);
	fread(&m->k_sparse, 4, 1, fp);
	fread(&m->scaled, 4, 1, fp);
	fread(&m->n_layers, 4, 1, fp);
	m->n_neurons = (int32_t*)calloc(m->n_layers, 4);
	m->af = (int32_t*)calloc(m->n_layers - 1, 4);
	fread(m->n_neurons, 4, m->n_layers, fp);
	fread(m->af, 4, m->n_layers - 1, fp);
	n_par = sann_n_par(m);
	m->t = (float*)malloc(n_par * sizeof(float));
	fread(m->t, sizeof(float), n_par, fp);
	if (col_names && fread(&tot_len, 8, 1, fp) == 1) {
		char *p, *q;
		p = (char*)malloc(tot_len);
		fread(p, 1, tot_len, fp);
		*col_names = (char**)calloc(sann_n_in(m), sizeof(char*));
		for (i = 0, q = p; i < sann_n_in(m); ++i) {
			int l;
			l = strlen(q);
			// possible to avoid strdup() but will leave a trap to endusers
			(*col_names)[i] = strdup(q);
			q += l + 1;
			assert(q - p <= tot_len);
		}
		free(p);
		assert(q - p == tot_len);
	}
	if (fp != stdin) fclose(fp);
	return m;
}

void sann_print(const sann_t *m)
{
/*
	const float *b1, *b2, *w10;
	int i, j;
	sae_model_par2ptr(m, &b1, &b2, &w10);
	printf("NI\t%d\n", m->n_in);
	printf("NH\t%d\n", m->n_hidden);
	printf("KS\t%d\n", m->k_sparse);
	printf("F1\t%d\n", m->f1);
	printf("SC\t%d\n", m->scaled);
	printf("B1");
	for (i = 0; i < m->n_hidden; ++i)
		printf("\t%g", b1[i]);
	printf("\nB2");
	for (i = 0; i < m->n_in; ++i)
		printf("\t%g", b2[i]);
	putchar('\n');
	for (j = 0; j < m->n_hidden; ++j) {
		const float *w10j = w10 + j * m->n_in;
		printf("WW\t%d", j);
		for (i = 0; i < m->n_in; ++i)
			printf("\t%g", w10j[i]);
		putchar('\n');
	}
*/
}
