#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include "priv.h"

int main_jacob(int argc, char *argv[])
{
	int c, N, N_out, n_in, n_out, i, j, k;
	float **x, **y, *d;
	char **rn, **cn_in, **cn_out;
	sann_t *m;
	smln_buf_t *b;

	while ((c = getopt(argc, argv, "")) >= 0) {
	}
	if (argc - optind < 3) {
		fprintf(stderr, "Usage: sann jacob <model.snm> <input.snd> <output.snd>\n");
		return 1;
	}

	m = sann_restore(argv[optind], &cn_in, &cn_out);
	x = sann_data_read(argv[optind+1], &N, &n_in, &rn, 0);
	y = sann_data_read(argv[optind+2], &N_out, &n_out, 0, 0);
	assert(n_in == sann_n_in(m) && n_out == sann_n_out(m) && N == N_out);

	if (cn_in) {
		printf("#out");
		for (i = 0; i < n_in; ++i) printf("\t%s", cn_in[i]);
		putchar('\n');
	}

	b = smln_buf_init(m->n_layers, m->n_neurons, m->t);
	d = (float*)malloc(sann_n_in(m) * sizeof(float));
	for (k = 0; k < sann_n_out(m); ++k) {
		int n;
		if (cn_out) printf("%s", cn_out[k]);
		else printf("o%d", k+1);
		memset(d, 0, sann_n_in(m) * sizeof(float));
		for (j = n = 0; j < N; ++j) {
			if (y[j][k] > 0.) {
				++n;
				smln_core_jacobian(m->n_layers, m->n_neurons, m->af, m->t, x[j], k, d, b);
			}
		}
		for (i = 0; i < n_in; ++i)
			printf("\t%g", d[i] / n);
		putchar('\n');
	}
	smln_buf_destroy(b);

	sann_free_names(N, rn);
	sann_free_vectors(N, x);
	sann_free_vectors(N, y);
	sann_free_names(sann_n_in(m), cn_in);
	sann_free_names(sann_n_out(m), cn_out);
	sann_destroy(m);
	return 0;
}
