#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include "sann_priv.h"

int main_jacob(int argc, char *argv[])
{
	int c, N, N_out, n_in, n_out, i, j, k, trans = 0;
	float **x, **y;
	char **rn, **cn_in, **cn_out;
	sann_t *m;
	sfnn_buf_t *b;

	while ((c = getopt(argc, argv, "T")) >= 0) {
		if (c == 'T') trans = 1;
	}
	if (argc - optind < 3) {
		fprintf(stderr, "Usage: sann jacob [-T] <model.snm> <input.snd> <output.snd>\n");
		return 1;
	}

	m = sann_restore(argv[optind], &cn_in, &cn_out);
	x = sann_data_read(argv[optind+1], &N, &n_in, &rn, 0);
	y = sann_data_read(argv[optind+2], &N_out, &n_out, 0, 0);
	assert(n_in == sann_n_in(m) && n_out == sann_n_out(m) && N == N_out);

	b = sfnn_buf_init(m->n_layers, m->n_neurons, m->t);
	if (!trans) {
		float *d;
		if (cn_in) {
			printf("#NA");
			for (i = 0; i < n_in; ++i) printf("\t%s", cn_in[i]);
			putchar('\n');
		}
		d = (float*)malloc(sann_n_in(m) * sizeof(float));
		for (k = 0; k < sann_n_out(m); ++k) {
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
	sfnn_buf_destroy(b);

	sann_free_names(N, rn);
	sann_free_vectors(N, x);
	sann_free_vectors(N, y);
	sann_free_names(sann_n_in(m), cn_in);
	sann_free_names(sann_n_out(m), cn_out);
	sann_destroy(m);
	return 0;
}
