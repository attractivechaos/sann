#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include "fann.h"
#include "sann.h"

int main_train(int argc, char *argv[])
{
	int c, n_rows, n_in, n_out, n_rows_out, i, n_epochs = 50;
	int n_hidden = 50, seed = 11;
	float **x, **y, eps = .0001f, h = .01;
	struct fann_train_data *data;
	char *out = 0;
	struct fann *ann;

	fann_disable_seed_rand();
	while ((c = getopt(argc, argv, "h:n:o:e:s:")) >= 0) {
		if (c == 'h') n_hidden = atoi(optarg);
		else if (c == 'n') n_epochs = atoi(optarg);
		else if (c == 'o') out = optarg;
		else if (c == 'e') h = atof(optarg);
		else if (c == 's') seed = atoi(optarg);
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: sann-fann [options] <input.snd> <output.snd>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -h INT     number of hidden neurons [%d]\n", n_hidden);
		fprintf(stderr, "  -n INT     number of epochs [%d]\n", n_epochs);
		fprintf(stderr, "  -o FILE    output the model to FILE []\n");
		fprintf(stderr, "  -e FLOAT   learning rate [%g]\n", h);
		fprintf(stderr, "  -s INT     seed (<=0 to random) [%d]\n", seed);
		return 1;
	}
	if (seed > 0) srand(seed);

	x = sann_data_read(argv[optind+0], &n_rows, &n_in, 0, 0);
	y = sann_data_read(argv[optind+1], &n_rows_out, &n_out, 0, 0);
	assert(n_rows == n_rows_out);
	fprintf(stderr, "[M::%s] read %d samples; (%d,%d)\n", __func__, n_rows, n_in, n_out);
	data = fann_create_train_pointer_array(n_rows, n_in, x, n_out, y);
	for (i = 0; i < n_in; ++i) free(x[i]);
	for (i = 0; i < n_out; ++i) free(y[i]);
	free(x); free(y);

	ann = fann_create_standard(3, n_in, n_hidden, n_out);
	fann_set_learning_rate(ann, h);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	fann_train_on_data(ann, data, n_epochs, 1, eps);
	if (out) fann_save(ann, out);

	fann_destroy(ann);
	fann_destroy_train(data);
	return 0;
}

int main_apply(int argc, char *argv[])
{
	struct fann *ann;
	float **x;
	int n_rows, n_in, n_out, i, j;
	char **row_names;

	if (argc < 3) {
		fprintf(stderr, "Usage: sann-fann <model.fann> <input.snd>\n");
		return 1;
	}

	ann = fann_create_from_file(argv[1]);
	n_out = fann_get_num_output(ann);
	x = sann_data_read(argv[2], &n_rows, &n_in, &row_names, 0);
	for (i = 0; i < n_rows; ++i) {
		float *y;
		printf("%s", row_names[i]);
		y = fann_run(ann, x[i]);
		for (j = 0; j < n_out; ++j)
			printf("\t%g", y[j] + 1.0f - 1.0f);
		putchar('\n');
		free(x[i]); free(row_names[i]);
	}
	free(x); free(row_names);

	fann_destroy(ann);
	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;
	if (argc == 1) {
		fprintf(stderr, "Usage: sann-fann <command> <arguments>\n");
		fprintf(stderr, "Commands:\n");
		fprintf(stderr, "  train      train the autoencoder\n");
		fprintf(stderr, "  apply      apply the model\n");
		return 1;
	}
	if (strcmp(argv[1], "train") == 0) ret = main_train(argc-1, argv+1);
	else if (strcmp(argv[1], "apply") == 0) ret = main_apply(argc-1, argv+1);
	else {
		fprintf(stderr, "[E::%s] unknown command\n", __func__);
		return 1;
	}
	return ret;
}
