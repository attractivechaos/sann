#include <sys/resource.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <xmmintrin.h>
#include "sann.h"

#define SANN_TRAIN_FUZZY .005

int main_train(int argc, char *argv[])
{
	int c, i, N, n_in, n_out = 0, n_hidden = 50, af = -1, k_sparse = -1, scaled = SAE_SC_SQRT, malgo = 0, balgo = 0;
	int n_layers = 3, n_neurons[3];
	float **x, **y;
	sann_t *m = 0;
	sann_tconf_t tc, tc1;
	char **row_names, **col_names_in = 0, **col_names_out = 0;

	srand48(11);
	memset(&tc1, 0, sizeof(sann_tconf_t));
	tc1.r = tc1.vfrac = -1.0f;
	while ((c = getopt(argc, argv, "h:n:r:e:i:s:f:k:S:T:m:b:")) >= 0) {
		if (c == 'h') n_hidden = atoi(optarg);
		else if (c == 'n') tc1.n_epochs = atoi(optarg);
		else if (c == 'r') tc1.r = atof(optarg);
		else if (c == 'T') tc1.vfrac = atof(optarg);
		else if (c == 'e') tc1.h = atof(optarg);
		else if (c == 'i') m = sann_restore(optarg, &col_names_in, &col_names_out);
		else if (c == 's') srand48(atol(optarg));
		else if (c == 'f') af = atoi(optarg);
		else if (c == 'k') k_sparse = atoi(optarg);
		else if (c == 'S') scaled = atoi(optarg);
		else if (c == 'm') malgo = atoi(optarg);
		else if (c == 'b') balgo = atoi(optarg);
	}
	sann_tconf_init(&tc, malgo, balgo);
	if (tc1.h > 0.0f) tc.h = tc1.h;
	if (tc1.r >= 0.0f) tc.r = tc1.r; 
	if (tc1.vfrac >= 0.0f) tc.vfrac = tc1.vfrac; 
	if (tc1.n_epochs > 0) tc.n_epochs = tc1.n_epochs; 
	if (argc == optind) {
		fprintf(stderr, "Usage: sann train [options] <input.snd> [output.snd]\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  Model construction:\n");
		fprintf(stderr, "    -i FILE     read model from FILE []\n");
		fprintf(stderr, "    -h INT      number of hidden neurons [%d]\n", n_hidden);
		fprintf(stderr, "    -f INT      hidden activation (1:sigm; 2:tanh; 3:ReLU) [1 for AE; 3 for MLN]\n");
		fprintf(stderr, "    -s INT      random seed [11]\n");
		fprintf(stderr, "  Autoencoder specific:\n");
		fprintf(stderr, "    -r FLOAT    fraction of noises [%g]\n", tc.r);
		fprintf(stderr, "    -k INT      k-sparse (<=0 or >={-h} to disable) [-1]\n");
		fprintf(stderr, "    -S INT      weight scaling (0:none; 1:sqrt; 2:full) [%d]\n", scaled);
		fprintf(stderr, "  Model training:\n");
		fprintf(stderr, "    -m INT      minibatch optimization (1:SGD; 2:RMSprop) [%d]\n", SANN_MIN_MINI_RMSPROP);
		fprintf(stderr, "    -b INT      batch optimization (1:fixed rate; 2:iRprop- adaptive) [%d]\n", SANN_MIN_BATCH_RPROP);
		fprintf(stderr, "    -e FLOAT    learning rate [.01 for SGD; .001 for RMSprop]\n");
		fprintf(stderr, "    -T FLOAT    fraction of data used for testing [%g]\n", tc.vfrac);
		fprintf(stderr, "    -n INT      number of epochs [%d]\n", tc.n_epochs);
		return 1;
	}

	if (m) {
		if ((m->is_mln && optind+1 == argc) || (!m->is_mln && optind+1 < argc))
			fprintf(stderr, "[M::%s] mismatch between the input model and the command line\n", __func__);
		n_hidden = m->n_neurons[1];
	}

	x = sann_data_read(argv[optind], &N, &n_in, &row_names, col_names_in? 0 : &col_names_in);
	fprintf(stderr, "[M::%s] read %d vectors, each of size %d\n", __func__, N, n_in);
	if (optind + 1 < argc) {
		y = sann_data_read(argv[optind+1], &N, &n_out, 0, col_names_out? 0 : &col_names_out);
		fprintf(stderr, "[M::%s] read %d vectors, each of size %d\n", __func__, N, n_out);
	} else y = 0;


	if (m) {
		if (sann_n_in(m) != n_in) {
			fprintf(stderr, "[E::%s] the model does not match the input: %d != %d\n", __func__, sann_n_in(m), n_in);
			return 1; // FIXME: memory leak
		}
	} else {
		if (optind + 1 == argc) { // AE
			m = sann_init_ae(n_in, n_hidden, scaled);
			if (af > 0) m->af[0] = af;
			if (k_sparse > 0) m->k_sparse = k_sparse;
		} else {
			n_neurons[0] = n_in, n_neurons[1] = n_hidden, n_neurons[2] = n_out;
			m = sann_init_mln(n_layers, n_neurons);
			if (af > 0)
				for (i = 0; i < m->n_layers - 2; ++i) m->af[i] = af;
		}
	}

	sann_data_shuffle(N, x, y, row_names);
	sann_train(m, &tc, N, x, y);
	sann_dump(0, m, col_names_in, col_names_out);

	sann_free_names(n_in, col_names_in);
	sann_free_names(sann_n_out(m), col_names_out);
	sann_free_names(N, row_names);
	sann_free_vectors(N, x);
	sann_free_vectors(N, y);
	sann_destroy(m);
	return 0;
}

int main_apply(int argc, char *argv[])
{
	int i, j, c, n_samples, n_in, show_hidden = 0;
	sann_t *m;
	float **x, *y, *z;
	double cost;
	char **row_names, **col_names_in = 0, **col_names_out = 0;

	while ((c = getopt(argc, argv, "h")) >= 0) {
		if (c == 'h') show_hidden = 1;
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: sann apply [options] <model> <data>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -h        show the activation of hidden neurons\n");
		return 1;
	}

	m = sann_restore(argv[optind], &col_names_in, &col_names_out);
	x = sann_data_read(argv[optind+1], &n_samples, &n_in, &row_names, col_names_in? 0 : &col_names_in);
	if (sann_n_in(m) != n_in) {
		fprintf(stderr, "[M::%s] mismatch between the input model and the input data\n", __func__);
		return 1;
	}

	if (m->is_mln && col_names_out) {
		printf("#sample");
		for (i = 0; i < sann_n_out(m); ++i)
			printf("\t%s", col_names_out[i]);
		putchar('\n');
	} else if (!m->is_mln && !show_hidden && col_names_in) {
		printf("#sample");
		for (i = 0; i < sann_n_in(m); ++i)
			printf("\t%s", col_names_in[i]);
		putchar('\n');
	}

	y = (float*)malloc((sann_n_out(m) + sae_n_hidden(m)) * sizeof(float));
	z = y + sann_n_out(m);
	for (i = 0, cost = 0.; i < n_samples; ++i) {
		sann_apply(m, x[i], y, z);
		cost += sann_cost(sann_n_out(m), x[i], y);
		printf("%s", row_names[i]);
		if (show_hidden && !m->is_mln) {
			for (j = 0; j < sae_n_hidden(m); ++j)
				printf("\t%g", z[j] + 1.0f - 1.0f);
		} else {
			for (j = 0; j < sann_n_out(m); ++j)
				printf("\t%g", y[j] + 1.0f - 1.0f);
		}
		putchar('\n');
		free(x[i]);
	}
	free(x); free(y);
	sann_free_names(n_samples, row_names);
	sann_free_names(sann_n_in(m), col_names_in);
	sann_free_names(sann_n_out(m), col_names_out);

	if (!m->is_mln) fprintf(stderr, "[M::%s] cost = %g\n", __func__, cost / n_samples);

	sann_destroy(m);
	return 0;
}

/*****************
 * Main function *
 *****************/

int main_jacob(int argc, char *argv[]);

void liftrlimit()
{
#ifdef __linux__
	struct rlimit r;
	getrlimit(RLIMIT_AS, &r);
	r.rlim_cur = r.rlim_max;
	setrlimit(RLIMIT_AS, &r);
#endif
}

double cputime()
{
	struct rusage r;
	getrusage(RUSAGE_SELF, &r);
	return r.ru_utime.tv_sec + r.ru_stime.tv_sec + 1e-6 * (r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}

double realtime()
{
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return tp.tv_sec + tp.tv_usec * 1e-6;
}

int main(int argc, char *argv[])
{
	int ret = 0, i;
	double t_start;
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
	liftrlimit();
	if (argc == 1) {
		fprintf(stderr, "Usage: sann <command> <arguments>\n");
		fprintf(stderr, "Commands:\n");
		fprintf(stderr, "  train      train the model\n");
		fprintf(stderr, "  apply      apply the model\n");
		fprintf(stderr, "  jacob      compute jacobian d{output}/d{input}\n");
		return 1;
	}
	t_start = realtime();
	if (strcmp(argv[1], "train") == 0) ret = main_train(argc-1, argv+1);
	else if (strcmp(argv[1], "apply") == 0) ret = main_apply(argc-1, argv+1);
	else if (strcmp(argv[1], "jacob") == 0) ret = main_jacob(argc-1, argv+1);
	else {
		fprintf(stderr, "[E::%s] unknown command\n", __func__);
		return 1;
	}
	if (ret == 0) {
		fprintf(stderr, "[M::%s] Version: %s\n", __func__, SANN_VERSION);
		fprintf(stderr, "[M::%s] CMD:", __func__);
		for (i = 0; i < argc; ++i)
			fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n[M::%s] Real time: %.3f sec; CPU: %.3f sec\n", __func__, realtime() - t_start, cputime());
	}
	return ret;
}
