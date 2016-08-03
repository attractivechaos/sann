#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <zlib.h>
#include "sann.h"
#include "kseq.h"
KSTREAM_INIT(gzFile, gzread, 16384)

#define SAE_MAGIC "SAE\1"
#define SLN_MAGIC "SLN\1"

float **sann_data_read(const char *fn, int *n_, int *n_col_, char ***row_names, char ***col_names)
{
	gzFile fp;
	kstream_t *ks;
	float **x = 0;
	int n = 0, m = 0, dret, n_col = 0;
	kstring_t str = {0,0,0};

	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
	ks = ks_init(fp);
	if (row_names) *row_names = 0;
	if (col_names) *col_names = 0;
	while (ks_getuntil(ks, KS_SEP_LINE, &str, &dret) >= 0) {
		int st, i, k;
		if (str.s[0] == '#' && col_names) {
			for (i = k = 0; i < str.l; ++i)
				if (str.s[i] == '\t') ++k;
			if (k > 0) {
				n_col = k;
				*col_names = (char**)malloc(n_col * sizeof(char*));
				for (i = k = st = 0; i <= str.l; ++i) {
					if (i == str.l || str.s[i] == '\t') {
						if (k > 0) str.s[i] = 0, (*col_names)[k-1] = strdup(&str.s[st]);
						++k, st = i + 1;
					}
				}
			}
		}
		if (str.s[0] == '#') continue;
		for (i = k = 0; i < str.l; ++i)
			if (str.s[i] == '\t') ++k;
		if (n_col == 0) n_col = k;
		if (k != n_col) continue; // TODO: throw a warning/error
		if (n == m) {
			m = m? m<<1 : 8;
			x = (float**)realloc(x, m * sizeof(float*));
			if (row_names)
				*row_names = (char**)realloc(*row_names, m * sizeof(char*));
		}
		x[n] = (float*)malloc(n_col * sizeof(float));
		for (i = k = st = 0; i <= str.l; ++i) {
			if (i == str.l || str.s[i] == '\t') {
				char *p;
				if (k == 0) {
					str.s[i] = 0;
					if (row_names) (*row_names)[n] = strdup(&str.s[st]);
				} else x[n][k-1] = strtod(&str.s[st], &p);
				++k, st = i + 1;
			}
		}
		++n;
	}
	free(str.s);
	ks_destroy(ks);
	gzclose(fp);
	x = (float**)realloc(x, n * sizeof(float*));
	if (row_names) *row_names = (char**)realloc(*row_names, n * sizeof(char*));
	*n_ = n, *n_col_ = n_col;
	return x;
}

int sae_dump(const char *fn, const sae_t *m, char *const* col_names)
{
	FILE *fp;
	int i, n_par = sae_n_par(m->n_in, m->n_hidden);
	fp = fn && strcmp(fn, "-")? fopen(fn, "w") : stdout;
	if (fp == 0) return -1;
	fwrite(SAE_MAGIC, 1, 4, fp);
	fwrite(&m->n_in, 4, 1, fp);
	fwrite(&m->n_hidden, 4, 1, fp);
	fwrite(&m->k_sparse, 4, 1, fp);
	fwrite(&m->f1, 4, 1, fp);
	fwrite(&m->scaled, 4, 1, fp);
	fwrite(m->t, sizeof(float), n_par, fp);
	if (col_names) {
		uint64_t tot_len = 0;
		for (i = 0; i < m->n_in; ++i)
			tot_len += strlen(col_names[i]) + 1;
		fwrite(&tot_len, 8, 1, fp);
		for (i = 0; i < m->n_in; ++i)
			fwrite(col_names[i], 1, strlen(col_names[i]) + 1, fp);
	}
	if (fp != stdout) fclose(fp);
	return 0;
}

sae_t *sae_restore(const char *fn, char ***col_names)
{
	FILE *fp;
	char magic[4];
	sae_t *m;
	int n_par, i;
	uint64_t tot_len;
	if (col_names) *col_names = 0;
	fp = fn && strcmp(fn, "-")? fopen(fn, "r") : stdin;
	if (fp == 0) return 0;
	fread(magic, 1, 4, fp);
	if (strncmp(magic, SAE_MAGIC, 4) != 0) return 0;
	m = (sae_t*)calloc(1, sizeof(sae_t));
	fread(&m->n_in, 4, 1, fp);
	fread(&m->n_hidden, 4, 1, fp);
	fread(&m->k_sparse, 4, 1, fp);
	fread(&m->f1, 4, 1, fp);
	fread(&m->scaled, 4, 1, fp);
	n_par = sae_n_par(m->n_in, m->n_hidden);
	m->t = (float*)malloc(n_par * sizeof(float));
	fread(m->t, sizeof(float), n_par, fp);
	if (col_names && fread(&tot_len, 8, 1, fp) == 1) {
		char *p, *q;
		p = (char*)malloc(tot_len);
		fread(p, 1, tot_len, fp);
		*col_names = (char**)calloc(m->n_in, sizeof(char*));
		for (i = 0, q = p; i < m->n_in; ++i) {
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

int smln_dump(const char *fn, const smln_t *m)
{
	FILE *fp;
	int n_par = smln_n_par(m->n_layers, m->n_neurons);
	fp = fn && strcmp(fn, "-")? fopen(fn, "w") : stdout;
	if (fp == 0) return -1;
	fwrite(SLN_MAGIC, 1, 4, fp);
	fwrite(&m->n_layers, 4, 1, fp);
	fwrite(&m->n_neurons, 4, m->n_layers, fp);
	fwrite(&m->af, 4, m->n_layers - 1, fp);
	fwrite(m->t, sizeof(float), n_par, fp);
	if (fp != stdout) fclose(fp);
	return 0;
}

smln_t *smln_restore(const char *fn)
{
	FILE *fp;
	char magic[4];
	smln_t *m;
	int n_par;
	fp = fn && strcmp(fn, "-")? fopen(fn, "r") : stdin;
	if (fp == 0) return 0;
	fread(magic, 1, 4, fp);
	if (strncmp(magic, SLN_MAGIC, 4) != 0) return 0;
	m = (smln_t*)calloc(1, sizeof(smln_t));
	fread(&m->n_layers, 4, 1, fp);
	m->n_neurons = (int32_t*)calloc(m->n_layers, 4);
	m->af = (int32_t*)calloc(m->n_layers - 1, 4);
	fread(&m->n_neurons, 4, m->n_layers, fp);
	fread(&m->af, 4, m->n_layers - 1, fp);
	n_par = smln_n_par(m->n_layers, m->n_neurons);
	m->t = (float*)malloc(n_par * sizeof(float));
	fread(m->t, sizeof(float), n_par, fp);
	if (fp != stdin) fclose(fp);
	return m;
}
