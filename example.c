// to compile: gcc -g -O2 -Wall example.c -L. -lsann -lz

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "sann.h"

int main(int argc, char *argv[])
{
	sann_t *m;
	sann_tconf_t conf;
	int i, j, N, tmp, n_neurons[3];
	float **x, **y, *out;
	char **row_names;
	if (argc < 4) {
		fprintf(stderr, "Usage: sann-demo <train-in.snd> <train-out.snd> <test-in.snd>\n");
		return 1;
	}
	// read training data
	x = sann_data_read(argv[1], &N, &n_neurons[0], 0, 0);
	y = sann_data_read(argv[2], &tmp, &n_neurons[2], 0, 0);
	assert(N == tmp);
	sann_data_shuffle(N, x, y, 0);
	// initialize the network
	n_neurons[1] = 50;
	m = sann_init_mln(3, n_neurons);
	// training
	sann_tconf_init(&conf, 0, 0);
	sann_train(m, &conf, N, x, y);
	// free training data
	sann_free_vectors(N, x);
	sann_free_vectors(N, y);
	// read test data
	x = sann_data_read(argv[3], &N, &tmp, &row_names, 0);
	out = (float*)malloc(n_neurons[2] * sizeof(float));
	for (i = 0; i < N; ++i) {
		sann_apply(m, x[i], out, 0);
		printf("%s", row_names[i]);
		for (j = 0; j < n_neurons[2]; ++j)
			printf("\t%g", out[j]);
		putchar('\n');
	}
	free(out);
	sann_free_vectors(N, x);
	sann_free_names(N, row_names);
	// deallocate the model
	sann_destroy(m);
	return 0;
}
