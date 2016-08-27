#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "sann.h"

int main(int argc, char *argv[])
{
	sann_t *m;
	sann_tconf_t conf;
	int i, j, n_neurons[3] = { 2, 2, 1 }, n = 1000, n_succ = 0;
	float x0[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} }, y0[4][1] = { {0}, {1}, {1}, {0} };
	float t, *x[4], *y[4];

	if (argc > 1) n_neurons[1] = atoi(argv[1]); // the optional CLI argument set the number of hidden neurons
	for (i = 0; i < 4; ++i) x[i] = x0[i], y[i] = y0[i];
	sann_tconf_init(&conf, 0, 0);          // initialize training parameters
	conf.vfrac = 0.0f;                     // no validation samples
	conf.n_epochs = conf.max_inc = 200;    // always perform 200 iterations
	sann_verbose = 1;
	for (j = 0; j < n; ++j) {              // try different random seeds
		sann_srand(j * 13 + 1);
		m = sann_init_fnn(3, n_neurons);   // initialize the NN
		sann_train(m, &conf, 4, x, y);
		for (i = 0; i < 4; ++i) {  // test if we can fit the XOR function
			int s = (int)(x[i][0] + .5) ^ (int)(x[i][1] + .5);
			sann_apply(m, x[i], &t, 0);
			if (fabs(s - t) > 0.05) break; // confirm a fit if the output error is within 5%
		}
		if (i == 4) ++n_succ;
		sann_destroy(m);                   // deallocate the model
	}
	printf("chance to fit the XOR function with %d hidden neurons: %.1f%%\n", n_neurons[1], 100. * n_succ / n);
	return 0;
}
