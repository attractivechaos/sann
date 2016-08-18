#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "sann.h"

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
