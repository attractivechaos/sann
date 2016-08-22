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

	if (argc > 1) n_neurons[1] = atoi(argv[1]);
	for (i = 0; i < 4; ++i) x[i] = x0[i], y[i] = y0[i];
	sann_tconf_init(&conf, 0, 0);
	conf.r_in = conf.vfrac = 0.0f;
	conf.n_epochs = conf.max_inc = 200;
	sann_verbose = 1;
	for (j = 0; j < n; ++j) {
		int n_cor;
		sann_srand(j * 13 + 1);
		m = sann_init_fnn(3, n_neurons);
		sann_train(m, &conf, 4, x, y);
		for (i = n_cor = 0; i < 4; ++i) {
			int s = (int)(x[i][0] + .5) ^ (int)(x[i][1] + .5);
			sann_apply(m, x[i], &t, 0);
			if (fabs(s - t) < 0.05) ++n_cor;
		}
		if (n_cor == 4) ++n_succ;
		sann_destroy(m);
	}
	printf("chance to fit the XOR function with %d hidden neurons: %.1f%%\n", n_neurons[1], 100. * n_succ / n);
	return 0;
}
