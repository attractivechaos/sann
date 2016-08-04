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
	int c, i, N, n_in, n_out = 0, n_hidden = 50, n_rounds = 20, af = -1, k_sparse = -1, scaled = SAE_SC_SQRT;
	int n_layers = 3, n_neurons[3];
	float **x, **y, min_h = .001, max_h = .1;
	sann_t *m = 0;
	sann_tconf_t tc;
	char **row_names, **col_names = 0;

	srand48(11);
	sann_tconf_init(&tc, SANN_MIN_RMSPROP);
	while ((c = getopt(argc, argv, "h:n:r:e:Gi:s:f:k:S:")) >= 0) {
		if (c == 'h') n_hidden = atoi(optarg);
		else if (c == 'n') n_rounds = atoi(optarg);
		else if (c == 'G') sann_tconf_init(&tc, SANN_MIN_SGD);
		else if (c == 'r') tc.r = atof(optarg);
		else if (c == 'i') m = sann_restore(optarg, &col_names);
		else if (c == 's') srand48(atol(optarg));
		else if (c == 'f') af = atoi(optarg);
		else if (c == 'k') k_sparse = atoi(optarg);
		else if (c == 'S') scaled = atoi(optarg);
		else if (c == 'e') {
			char *p;
			min_h = strtod(optarg, &p);
			if (*p == ',') max_h = strtod(p+1, &p);
		}
	}
	if (argc == optind) {
		fprintf(stderr, "Usage: sann train [options] <input.txt> [output.txt]\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -i FILE         read model from FILE []\n");
		fprintf(stderr, "  -h INT          number of hidden neurons [%d]\n", n_hidden);
		fprintf(stderr, "  -r FLOAT        fraction of noises [%g]\n", tc.r);
		fprintf(stderr, "  -k INT          k-sparse (<=0 or >={-h} to disable) [-1]\n");
		fprintf(stderr, "  -n INT          number of rounds of training [%d]\n", n_rounds);
		fprintf(stderr, "  -e FLOAT1[,F2]  min and max learning rate [%g,%g]\n", min_h, max_h);
		fprintf(stderr, "  -s INT          random seed [11]\n");
		fprintf(stderr, "  -f INT          hidden activation function (1:sigm; 2:tanh; 3:ReLU) [1]\n");
		fprintf(stderr, "  -G              use SGD [use RMSprop]\n");
		fprintf(stderr, "  -S INT          scaled model (0:none; 1:sqrt; 2:full) [%d]\n", scaled);
		return 1;
	}

	if (m) {
		if ((m->is_mln && optind+1 == argc) || (!m->is_mln && optind+1 < argc))
			fprintf(stderr, "[M::%s] mismatch between the input model and the command line\n", __func__);
		n_hidden = m->n_neurons[1];
	}

	x = sann_data_read(argv[optind], &N, &n_in, &row_names, col_names? 0 : &col_names);
	fprintf(stderr, "[M::%s] read %d vectors, each of size %d\n", __func__, N, n_in);
	if (optind + 1 < argc) {
		y = sann_data_read(argv[optind+1], &N, &n_out, 0, 0);
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
		}
	}

	sann_train(m, &tc, min_h, max_h, n_rounds, N, x, y);
	sann_dump(0, m, col_names);

	if (col_names) {
		for (i = 0; i < n_in; ++i) free(col_names[i]);
		free(col_names);
	}
	for (i = 0; i < N; ++i) {
		free(x[i]);
		if (y) free(y[i]);
		free(row_names[i]);
	}
	free(x); free(y); free(row_names);
	sann_destroy(m);
	return 0;
}

int main_apply(int argc, char *argv[])
{
	int i, j, c, n_samples, n_in, show_hidden = 0;
	sann_t *m;
	float **x, *y, *z;
	double cost;
	char **row_names, **col_names = 0;

	while ((c = getopt(argc, argv, "h")) >= 0) {
		if (c == 'h') show_hidden = 1;
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: sann apply [options] <model> <data>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -h        show the activation of hidden neurons\n");
		return 1;
	}

	m = sann_restore(argv[optind], &col_names);
	x = sann_data_read(argv[optind+1], &n_samples, &n_in, &row_names, col_names? 0 : &col_names);
	if (sann_n_in(m) != n_in) {
		fprintf(stderr, "[M::%s] mismatch between the input model and the input data\n", __func__);
		return 1;
	}
	y = (float*)malloc((sann_n_out(m) + sae_n_hidden(m)) * sizeof(float));
	z = y + sann_n_out(m);

	if (col_names) {
		printf("#sample");
		for (i = 0; i < sann_n_in(m); ++i)
			printf("\t%s", col_names[i]);
		putchar('\n');
	}
	for (i = 0, cost = 0.; i < n_samples; ++i) {
		cost += sann_apply(m, x[i], y, z);
		printf("%s", row_names[i]);
		if (show_hidden) {
			for (j = 0; j < sae_n_hidden(m); ++j)
				printf("\t%g", z[j] + 1.0f - 1.0f);
		} else {
			for (j = 0; j < sann_n_out(m); ++j)
				printf("\t%g", y[j] + 1.0f - 1.0f);
		}
		putchar('\n');
		free(row_names[i]); free(x[i]);
	}
	free(row_names); free(x);
	free(y);
	if (col_names) {
		for (i = 0; i < sann_n_in(m); ++i) free(col_names[i]);
		free(col_names);
	}

	fprintf(stderr, "[M::%s] cost = %g\n", __func__, cost / n_samples);

	sann_destroy(m);
	return 0;
}

int main_view(int argc, char *argv[])
{
	int i, c;
	sann_t *m;
	char **col_names = 0;
	while ((c = getopt(argc, argv, "")) >= 0) {
	}
	if (argc == optind) {
		fprintf(stderr, "Usage: sann view [options] <model.san>\n");
		return 1;
	}
	m = sann_restore(argv[optind], &col_names);
	sann_print(m);
	sann_destroy(m);
	if (col_names) {
		for (i = 0; i < sann_n_in(m); ++i) free(col_names[i]);
		free(col_names);
	}
	return 0;
}

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
		fprintf(stderr, "  train      train the autoencoder\n");
		fprintf(stderr, "  apply      apply the model\n");
		fprintf(stderr, "  view       print the model parameters\n");
		return 1;
	}
	t_start = realtime();
	if (strcmp(argv[1], "train") == 0) ret = main_train(argc-1, argv+1);
	else if (strcmp(argv[1], "apply") == 0) ret = main_apply(argc-1, argv+1);
	else if (strcmp(argv[1], "view") == 0) ret = main_view(argc-1, argv+1);
	else {
		fprintf(stderr, "[E::%s] unknown command\n", __func__);
		return 1;
	}
	if (ret == 0) {
		fprintf(stderr, "[M::%s] Version: %s\n", __func__, SAE_VERSION);
		fprintf(stderr, "[M::%s] CMD:", __func__);
		for (i = 0; i < argc; ++i)
			fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n[M::%s] Real time: %.3f sec; CPU: %.3f sec\n", __func__, realtime() - t_start, cputime());
	}
	return ret;
}
