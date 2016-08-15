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
	assert(N == tmp);                // make sure number inputs equal to number outputs
	sann_data_shuffle(N, x, y, 0);   // to avoid biased validation set
	// initialize the network
	n_neurons[1] = 50;               // number of hidden neurons
	m = sann_init_mln(3, n_neurons); // initialize a multi-layer network
	// training
	sann_tconf_init(&conf, 0, 0);    // set training parameters
	sann_train(m, &conf, N, x, y);   // actual training
	// free training data
	sann_free_vectors(N, x);
	sann_free_vectors(N, y);
	// apply the model to test data
	x = sann_data_read(argv[3], &N, &tmp, &row_names, 0);
	out = (float*)malloc(sann_n_out(m) * sizeof(float));
	for (i = 0; i < N; ++i) {        // iterate through test data sets
		sann_apply(m, x[i], out, 0);
		printf("%s", row_names[i]);
		for (j = 0; j < sann_n_out(m); ++j)
			printf("\t%g", out[j] + 1. - 1.);
		putchar('\n');
	}
	free(out);
	sann_free_vectors(N, x);
	sann_free_names(N, row_names);
	// deallocate the model
	sann_destroy(m);
	return 0;
}
