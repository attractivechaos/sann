#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include "priv.h"

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

void sann_apply(const sann_t *m, const float *x, float *y, float *_z)
{
	if (m->is_mln) {
		smln_buf_t *b;
		b = smln_buf_init(m->n_layers, m->n_neurons, m->t);
		smln_core_forward(m->n_layers, m->n_neurons, m->af, m->t, x, b);
		memcpy(y, b->out[m->n_layers-1], m->n_neurons[m->n_layers-1] * sizeof(float));
		smln_buf_destroy(b);
	} else {
		float *deriv1, *z;
		deriv1 = (float*)calloc(m->n_neurons[1], sizeof(float));
		z = _z? _z : (float*)calloc(sae_n_hidden(m), sizeof(float));
		sae_core_forward(sae_n_in(m), sae_n_hidden(m), m->t, sann_get_af(m->af[0]), sann_sigm, m->k_sparse, x, z, y, deriv1, m->scaled);
		if (_z == 0) free(z);
		free(deriv1);
	}
}

float sann_cost(int n, const float *y0, const float *y)
{
	int i;
	double cost = 0.;
	if (n == 0) return 0.;
	for (i = 0; i < n; ++i)
		cost += sann_sigm_cost(y0[i], y[i]);
	return (float)cost / n;
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

void sann_RMSprop2(int n, const float *h, float decay, float *t, float *g, float *r, sann_gradient_f func, void *data)
{
	int i;
	func(n, t, g, data);
	for (i = 0; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= h[i] / sqrt(1e-6 + r[i]) * g[i];
	}
}

void sann_tconf_init(sann_tconf_t *tc, int malgo, int balgo)
{
	memset(tc, 0, sizeof(sann_tconf_t));
	tc->malgo = malgo > 0? malgo : SANN_MIN_MINI_RMSPROP;
	tc->balgo = balgo > 0? balgo : SANN_MIN_BATCH_RPROP;
	tc->r = .3f;
	tc->L2_par = .001f;
	tc->h = .01f;
	tc->h_min = 0.0f, tc->h_max = .1f;
	tc->rprop_dec = .5f, tc->rprop_inc = 1.2f;
	tc->max_inc = 10;

	if (tc->malgo == SANN_MIN_MINI_SGD) {
		tc->mini_batch = 10;
		tc->h = .01f;
	} else if (tc->malgo == SANN_MIN_MINI_RMSPROP) {
		tc->mini_batch = 50;
		tc->h = .001f;
		tc->decay = .9f;
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

float sann_train_epoch(sann_t *m, const sann_tconf_t *tc, const float *h, int n, float *const* x, float *const* y, float **_buf)
{
	minibatch_t mb;
	float *buf, *g, *r;
	cfloat_p *sx = 0, *sy = 0;
	int mn = 0, n_par, n_out, buf_size;

	sx = (cfloat_p*)malloc(n * sizeof(cfloat_p));
	memcpy(sx, x, n * sizeof(cfloat_p));
	if (m->is_mln) {
		sy = (cfloat_p*)malloc(n * sizeof(cfloat_p));
		memcpy(sy, y, n * sizeof(cfloat_p));
	}
	sann_data_shuffle(n, sx, sy, 0);

	n_out = sann_n_out(m);
	n_par = sann_n_par(m);
	buf_size = 2 * n_par;

	buf = _buf? *_buf : 0;
	if (buf == 0) buf = (float*)calloc(buf_size, sizeof(float));
	if (_buf) *_buf = buf;
	g = buf, r = g + n_par;

	mb.m = m, mb.tc = tc, mb.running_cost = 0.;
	mb.buf_mln = m->is_mln? smln_buf_init(m->n_layers, m->n_neurons, m->t) : 0;
	mb.buf_ae = !m->is_mln? (float*)malloc(sae_buf_size(sae_n_in(m), sae_n_hidden(m)) * sizeof(float)) : 0;
	while (mn < n) {
		mb.n = tc->mini_batch < n - mn? tc->mini_batch : n - mn;
		mb.x = &sx[mn];
		mb.y = sy? &sy[mn] : 0;
		if (tc->malgo == SANN_MIN_MINI_SGD) {
			sann_SGD(n_par, tc->h, m->t, g, mb_gradient, &mb);
		} else if (tc->malgo == SANN_MIN_MINI_RMSPROP) {
			if (h) sann_RMSprop2(n_par, h, tc->decay, m->t, g, r, mb_gradient, &mb);
			else sann_RMSprop(n_par, tc->h, tc->decay, m->t, g, r, mb_gradient, &mb);
		}
		mn += mb.n;
	}
	if (mb.buf_ae) free(mb.buf_ae);
	if (mb.buf_mln) smln_buf_destroy(mb.buf_mln);

	if (_buf == 0) free(buf);
	free(sx); free(sy);
	return mb.running_cost / n_out / n;
}

float sann_test(const sann_t *m, int n, float *const* x, float *const* y0)
{
	int i, j;
	float *y;
	double sum = 0.;
	y = (float*)malloc(sann_n_out(m) * sizeof(float));
	for (i = 0; i < n; ++i) {
		double cost = 0.;
		sann_apply(m, x[i], y, 0);
		for (j = 0; j < sann_n_out(m); ++j)
			cost += sann_sigm_cost(m->is_mln? y0[i][j] : x[i][j], y[j]);
		sum += cost;
	}
	free(y);
	return (float)(sum / n / sann_n_out(m));
}

/*************************
 * Train for many epochs *
 *************************/

int sann_train(sann_t *m, const sann_tconf_t *tc0, int n_epochs, int n_train, int n_test, float *const* x, float *const* y)
{
	int i, k, n_par, n_cost_inc = 0, n_cost_inc2 = 0;
	float *g_prev, *g_curr, *t_prev, *h = 0, cost_best = FLT_MAX, cost_prev = FLT_MAX;
	sann_t *best;

	best = sann_dup(m);
	n_par = sann_n_par(m);
	t_prev = (float*)calloc(n_par * 3, sizeof(float));
	g_prev = t_prev + n_par;
	g_curr = g_prev + n_par;
	if (tc0->balgo == SANN_MIN_BATCH_RPROP) {
		h = (float*)calloc(n_par, sizeof(float));
		for (i = 0; i < n_par; ++i) h[i] = tc0->h;
	}
	for (k = 0; k < n_epochs; ++k) {
		float rc, cost;

		if (h) memcpy(t_prev, m->t, n_par * sizeof(float));
		rc = sann_train_epoch(m, tc0, h, n_train, x, y, 0);
		cost = n_test? sann_test(m, n_test, x + n_train, y? y + n_train : 0) : 0.;
		if (sann_verbose >= 3)
			fprintf(stderr, "[M::%s] epoch:%d running_cost:%g validation_cost:%g\n", __func__, k+1, rc, cost);

		if (cost < cost_best) {
			cost_best = cost;
			sann_cpy(best, m);
			n_cost_inc = 0;
		} else if (cost > cost_best) {
			if (++n_cost_inc > tc0->max_inc)
				break;
		}
		if (cost < cost_prev) {
			n_cost_inc2 = 0;
		} else if (cost > cost_prev) {
			if (++n_cost_inc2 > tc0->max_inc/2)
				break;
		}
		cost_prev = cost;

		if (h) { // iRprop-
			for (i = 0; i < n_par; ++i)
				g_curr[i] = m->t[i] - t_prev[i];
			if (k >= 1) { // iRprop-
				for (i = 0; i < n_par; ++i) {
					float tmp = g_prev[i] * g_curr[i];
					if (tmp > 0.) {
						h[i] *= tc0->rprop_inc;
						if (h[i] > tc0->h_max) h[i] = tc0->h_max;
					} else if (tmp < 0.) {
						h[i] *= tc0->rprop_dec;
						if (h[i] < tc0->h_min) h[i] = tc0->h_min;
						g_curr[i] = 0.;
					}
				}
			}
			memcpy(g_prev, g_curr, n_par * sizeof(float));
		}
	}
	free(t_prev); free(h);
	sann_cpy(m, best);
	sann_destroy(best);
	return k;
}

/**********************
 * Dump/restore model *
 **********************/

#define SANN_MAGIC "SAN\1"

static void sann_dump_names(FILE *fp, int n, char *const* names)
{
	uint64_t tot_len = 0;
	int i;
	if (names == 0) return;
	for (i = 0; i < n; ++i)
		tot_len += strlen(names[i]) + 1;
	fwrite(&tot_len, 8, 1, fp);
	for (i = 0; i < n; ++i)
		fwrite(names[i], 1, strlen(names[i]) + 1, fp);
}

int sann_dump(const char *fn, const sann_t *m, char *const* col_names_in, char *const* col_names_out)
{
	FILE *fp;
	int n_par;
	uint8_t name_flag = 0;

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
	if (col_names_in) name_flag |= 1;
	if (col_names_out) name_flag |= 2;
	fwrite(&name_flag, 1, 1, fp);
	sann_dump_names(fp, sann_n_in(m), col_names_in);
	sann_dump_names(fp, sann_n_out(m), col_names_out);
	if (fp != stdout) fclose(fp);
	return 0;
}

static char **sann_restore_names(FILE *fp, int n)
{
	uint64_t tot_len;
	if (fread(&tot_len, 8, 1, fp) == 1) {
		char *p, *q, **names;
		int i;
		p = (char*)malloc(tot_len);
		fread(p, 1, tot_len, fp);
		names = (char**)calloc(n, sizeof(char*));
		for (i = 0, q = p; i < n; ++i) {
			int l;
			l = strlen(q);
			// possible to avoid strdup() but will leave a trap to endusers
			names[i] = strdup(q);
			q += l + 1;
			assert(q - p <= tot_len);
		}
		free(p);
		assert(q - p == tot_len);
		return names;
	} else return 0;
}

sann_t *sann_restore(const char *fn, char ***col_names_in, char ***col_names_out)
{
	FILE *fp;
	char magic[4];
	sann_t *m;
	int n_par;
	uint8_t name_flag;

	if (col_names_in)  *col_names_in  = 0;
	if (col_names_out) *col_names_out = 0;
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
	if (fread(&name_flag, 1, 1, fp) == 1) {
		char **p;
		if (name_flag&1) {
			p = sann_restore_names(fp, sann_n_in(m));
			if (col_names_in) *col_names_in = p;
			else sann_free_names(sann_n_in(m), p);
		}
		if (name_flag&2) {
			p = sann_restore_names(fp, sann_n_out(m));
			if (col_names_out) *col_names_out = p;
			else sann_free_names(sann_n_out(m), p);
		}
	}
	if (fp != stdin) fclose(fp);
	return m;
}

void sann_free_names(int n, char **s)
{
	int i;
	if (s == 0) return;
	for (i = 0; i < n; ++i) free(s[i]);
	free(s);
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
