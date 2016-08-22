#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "sann.h"
#include "kseq.h"
#ifdef HAVE_ZLIB
#include <zlib.h>
KSTREAM_INIT(gzFile, gzread, 16384)
#else
#include <unistd.h>
#include <fcntl.h>
KSTREAM_INIT(int, read, 16384)
#endif

#define SANN_MAGIC "SAN\1"

float **sann_data_read(const char *fn, int *n_, int *n_col_, char ***row_names, char ***col_names)
{
	kstream_t *ks;
	float **x = 0;
	int n = 0, m = 0, dret, n_col = 0;
	kstring_t str = {0,0,0};

#ifdef HAVE_ZLIB
	gzFile fp;
	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
#else
	int fp;
	fp = fn && strcmp(fn, "-")? open(fn, O_RDONLY) : fileno(stdin);
#endif
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
	x = (float**)realloc(x, n * sizeof(float*));
	if (row_names) *row_names = (char**)realloc(*row_names, n * sizeof(char*));
	*n_ = n, *n_col_ = n_col;
#ifdef HAVE_ZLIB
	gzclose(fp);
#else
	close(fp);
#endif
	return x;
}

void sann_data_shuffle(int n, float **x, float **y, char **names)
{
	int i, *s;
	s = (int*)malloc(n * sizeof(int));
	for (i = n - 1; i >= 0; --i)
		s[i] = (int)(sann_drand() * (i+1));
	for (i = n - 1; i >= 0; --i) {
		float *tf;
		char *ts;
		int j = s[i];
		if (x) tf = x[i], x[i] = x[j], x[j] = tf;
		if (y) tf = y[i], y[i] = y[j], y[j] = tf;
		if (names) ts = names[i], names[i] = names[j], names[j] = ts;
	}
	free(s);
}

void sann_free_names(int n, char **s)
{
	int i;
	if (s == 0) return;
	for (i = 0; i < n; ++i) free(s[i]);
	free(s);
}

void sann_free_vectors(int n, float **x)
{
	int i;
	if (x == 0) return;
	for (i = 0; i < n; ++i) free(x[i]);
	free(x);
}
