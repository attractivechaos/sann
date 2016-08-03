#include <sys/resource.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <xmmintrin.h>
#include "sann.h"

static int train_to_term = 0;

static void train_term_cb(int sig)
{
	if (train_to_term) exit(0);
	fprintf(stderr, "[W::%s] signal SIGINT received. Stopping training...\n", __func__);
	train_to_term = 1;
}

int main_train(int argc, char *argv[])
{
	int c, i, N, k, n_in, n_out = 0, n_hidden = 50, n_rounds = 20, af = -1, k_sparse = -1, scaled = SAE_SC_SQRT, is_mln = 0;
	int n_layers = 3, n_neurons[3];
	float **x, **y;
	sae_t *ma = 0;
	smln_t *ml = 0;
	sann_tconf_t tc;
	char *fn_model = 0, **row_names, **col_names = 0;

	signal(SIGINT, train_term_cb);
	srand48(11);
	sann_tconf_init(&tc, SANN_MIN_RMSPROP);
	while ((c = getopt(argc, argv, "h:n:r:e:Gi:s:f:k:S:")) >= 0) {
		if (c == 'h') n_hidden = atoi(optarg);
		else if (c == 'n') n_rounds = atoi(optarg);
		else if (c == 'G') sann_tconf_init(&tc, SANN_MIN_SGD);
		else if (c == 'e') tc.h = atof(optarg);
		else if (c == 'r') tc.r = atof(optarg);
		else if (c == 'i') fn_model = optarg;
		else if (c == 's') srand48(atol(optarg));
		else if (c == 'f') af = atoi(optarg);
		else if (c == 'k') k_sparse = atoi(optarg);
		else if (c == 'S') scaled = atoi(optarg);
	}
	if (argc == optind) {
		fprintf(stderr, "Usage: sae train [options] <input.txt> [output.txt]\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -i FILE      read model from FILE []\n");
		fprintf(stderr, "  -h INT       number of hidden neurons [%d]\n", n_hidden);
		fprintf(stderr, "  -r FLOAT     fraction of noises [%g]\n", tc.r);
		fprintf(stderr, "  -k INT       k-sparse (<=0 or >={-h} to disable) [-1]\n");
		fprintf(stderr, "  -n INT       number of rounds of training [%d]\n", n_rounds);
		fprintf(stderr, "  -e FLOAT     learning rate [%g]\n", tc.h);
		fprintf(stderr, "  -s INT       random seed [11]\n");
		fprintf(stderr, "  -f INT       hidden activation function (1:sigm; 2:tanh; 3:ReLU) [1]\n");
		fprintf(stderr, "  -G           use SGD [use RMSprop]\n");
		fprintf(stderr, "  -S INT       scaled model (0:none; 1:sqrt; 2:full) [%d]\n", scaled);
		return 1;
	}
	is_mln = optind + 1 < argc? 1 : 0;
	if (fn_model) {
		if (!is_mln) ma = sae_restore(fn_model, &col_names);
		else ml = smln_restore(fn_model);
	}

	if (ma) n_hidden = ma->n_hidden;
	x = sann_data_read(argv[optind], &N, &n_in, &row_names, col_names? 0 : &col_names);
	fprintf(stderr, "[M::%s] read %d vectors, each of size %d\n", __func__, N, n_in);
	if (is_mln) {
		y = sann_data_read(argv[optind+1], &N, &n_out, 0, 0);
		fprintf(stderr, "[M::%s] read %d vectors, each of size %d\n", __func__, N, n_out);
	} else y = 0;

	if (!is_mln) {
		if (ma == 0) {
			ma = sae_init(n_in, n_hidden, scaled);
			if (af > 0) ma->f1 = af;
			if (k_sparse > 0) ma->k_sparse = k_sparse;
		} else if (ma->n_in != n_in) {
			fprintf(stderr, "[E::%s] the model does not match the input: %d != %d\n", __func__, ma->n_in, n_in);
			return 1; // FIXME: memory leak
		}
	} else {
		if (ml == 0) {
			n_neurons[0] = n_in, n_neurons[1] = n_hidden, n_neurons[2] = n_out;
			ml = smln_init(n_layers, n_neurons);
		} else if (ml->n_neurons[0] != n_in) {
			fprintf(stderr, "[E::%s] the model does not match the input: %d != %d\n", __func__, ml->n_neurons[0], n_in);
			return 1; // FIXME: memory leak
		}
	}

	for (k = 0; k < n_rounds; ++k) {
		float rc;
		rc = sann_train(is_mln? (void*)ml : (void*)ma, &tc, N, x, y);
		fprintf(stderr, "[M::%s] running_cost[%d] = %g\n", __func__, k+1, rc);
		if (train_to_term) break;
	}
	if (!is_mln) sae_dump(0, ma, col_names);
	else smln_dump(0, ml);

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

	if (ma) sae_destroy(ma);
	if (ml) smln_destroy(ml);
	return 0;
}

int main_apply(int argc, char *argv[])
{
	int i, j, c, n_samples, n_in, show_hidden = 0;
	sae_t *m;
	float **x, *y, *z;
	double cost;
	char **row_names, **col_names = 0;

	while ((c = getopt(argc, argv, "h")) >= 0) {
		if (c == 'h') show_hidden = 1;
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: sae apply [options] <model> <data>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -h        show the activation of hidden neurons\n");
		return 1;
	}

	m = sae_restore(argv[optind], &col_names);
	x = sann_data_read(argv[optind+1], &n_samples, &n_in, &row_names, col_names? 0 : &col_names);
	y = (float*)malloc((m->n_in + m->n_hidden) * sizeof(float));
	z = y + m->n_in;

	if (col_names) {
		printf("#sample");
		for (i = 0; i < m->n_in; ++i)
			printf("\t%s", col_names[i]);
		putchar('\n');
	}
	for (i = 0, cost = 0.; i < n_samples; ++i) {
		cost += sae_run(m, x[i], z, y);
		printf("%s", row_names[i]);
		if (show_hidden) {
			for (j = 0; j < m->n_hidden; ++j)
				printf("\t%g", z[j] + 1.0f - 1.0f);
		} else {
			for (j = 0; j < m->n_in; ++j)
				printf("\t%g", y[j] + 1.0f - 1.0f);
		}
		putchar('\n');
		free(row_names[i]); free(x[i]);
	}
	free(row_names); free(x);
	free(y);
	if (col_names) {
		for (i = 0; i < m->n_in; ++i) free(col_names[i]);
		free(col_names);
	}

	fprintf(stderr, "[M::%s] cost = %g\n", __func__, cost / n_samples);

	sae_destroy(m);
	return 0;
}

int main_view(int argc, char *argv[])
{
	int i, c;
	sae_t *m;
	char **col_names = 0;
	while ((c = getopt(argc, argv, "")) >= 0) {
	}
	if (argc == optind) {
		fprintf(stderr, "Usage: sae view [options] <model.sae>\n");
		return 1;
	}
	m = sae_restore(argv[optind], &col_names);
	sae_model_print(m);
	sae_destroy(m);
	if (col_names) {
		for (i = 0; i < m->n_in; ++i) free(col_names[i]);
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
		fprintf(stderr, "Usage: sae <command> <arguments>\n");
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
