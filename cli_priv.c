#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "sann_priv.h"

int main_jacob(int argc, char *argv[])
{
	int c, N, n_in, n_out, i, j, k, trans = 1;
	float **x = 0;
	char **cn_in, **cn_out;
	sann_t *m;
	sfnn_buf_t *b = 0;

	while ((c = getopt(argc, argv, "T")) >= 0) {
		if (c == 'T') trans = 0;
	}
	if (argc - optind < 1) {
		fprintf(stderr, "Usage: sann jacob [-T] <model.snm> [input.snd]\n");
		return 1;
	}

	m = sann_restore(argv[optind], &cn_in, &cn_out);
	n_out = sann_n_out(m), n_in = sann_n_in(m);
	if (argc - optind >= 2) {
		x = sann_data_read(argv[optind+1], &N, &n_in, 0, 0);
		assert(n_in == sann_n_in(m));
	}

	if (m->is_fnn) b = sfnn_buf_init(m->n_layers, m->n_neurons, m->t);
	if (x == 0) {
		const float *w;
		if (!m->is_fnn) {
			const float *b1, *b2;
			sae_par2ptr(n_in, sae_n_hidden(m), m->t, &b1, &b2, &w);
		} else w = b->w[1];
		for (i = 0; i < n_in; ++i) {
			double s = 0.;
			const float *wj;
			for (j = 0, wj = w; j < m->n_neurons[1]; ++j, wj += n_in) s += wj[i] * wj[i];
			if (cn_in) printf("%s", cn_in[i]);
			else printf("i%d", i+1);
			printf("\t%g\n", sqrt(s / m->n_neurons[1]));
		}
	} else if (!trans) {
		float *d;
		if (cn_in) {
			printf("#NA");
			for (i = 0; i < n_in; ++i) printf("\t%s", cn_in[i]);
			putchar('\n');
		}
		d = (float*)malloc(n_in * sizeof(float));
		for (k = 0; k < n_out; ++k) {
			if (cn_out) printf("%s", cn_out[k]);
			else printf("o%d", k+1);
			memset(d, 0, sann_n_in(m) * sizeof(float));
			for (j = 0; j < N; ++j)
				sfnn_core_jacobian(m->n_layers, m->n_neurons, m->af, m->t, x[j], k, d, b);
			for (i = 0; i < n_in; ++i)
				printf("\t%g", d[i] / N);
			putchar('\n');
		}
		free(d);
	} else {
		float **d;
		if (cn_out) {
			printf("#NA");
			for (i = 0; i < n_out; ++i) printf("\t%s", cn_out[i]);
			putchar('\n');
		}
		d = (float**)malloc(n_out * sizeof(float*));
		for (k = 0; k < n_out; ++k) {
			d[k] = (float*)calloc(n_in, sizeof(float));
			for (j = 0; j < N; ++j)
				sfnn_core_jacobian(m->n_layers, m->n_neurons, m->af, m->t, x[j], k, d[k], b);
		}
		for (i = 0; i < n_in; ++i) {
			if (cn_in) printf("%s", cn_in[i]);
			else printf("i%d", i+1);
			for (k = 0; k < n_out; ++k)
				printf("\t%g", d[k][i] / N);
			putchar('\n');
		}
		sann_free_vectors(n_out, d);
	}
	if (b) sfnn_buf_destroy(b);

	if (x) sann_free_vectors(N, x);
	sann_free_names(sann_n_in(m), cn_in);
	sann_free_names(sann_n_out(m), cn_out);
	sann_destroy(m);
	return 0;
}
