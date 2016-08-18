#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include "genann.h"
#include "sann.h"

int main_train(int argc, char *argv[])
{
	int c, k, n_rows, n_in, n_out, n_rows_out, i, n_epochs = 20;
	int n_hidden = 50;
	float **x, **y, h = .01;
	genann *ann;

	while ((c = getopt(argc, argv, "h:n:e:")) >= 0) {
		if (c == 'h') n_hidden = atoi(optarg);
		else if (c == 'n') n_epochs = atoi(optarg);
		else if (c == 'e') h = atof(optarg);
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: sann-genann train [options] <input.snd> <output.snd>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -h INT     number of hidden neurons [%d]\n", n_hidden);
		fprintf(stderr, "  -n INT     number of epochs [%d]\n", n_epochs);
		fprintf(stderr, "  -e FLOAT   learning rate [%g]\n", h);
		return 1;
	}

	x = sann_data_read(argv[optind+0], &n_rows, &n_in, 0, 0);
	y = sann_data_read(argv[optind+1], &n_rows_out, &n_out, 0, 0);
	assert(n_rows == n_rows_out);
	fprintf(stderr, "[M::%s] read %d samples; (%d,%d)\n", __func__, n_rows, n_in, n_out);

	ann = genann_init(n_in, 1, n_hidden, n_out);
	for (k = 0; k < n_epochs; ++k) {
		fprintf(stderr, "[M::%s] entering epoch %d...\n", __func__, k+1);
		for (i = 0; i < n_rows; ++i)
			genann_train(ann, x[i], y[i], h);
	}
	genann_write(ann, stdout);
	genann_free(ann);

	for (i = 0; i < n_in; ++i) free(x[i]);
	for (i = 0; i < n_out; ++i) free(y[i]);
	free(x); free(y);
	return 0;
}

int main_apply(int argc, char *argv[])
{
	genann *ann = 0;
	FILE *fp;
	float **x;
	int n_rows, n_in, n_out, i, j;
	char **row_names;

	if (argc < 3) {
		fprintf(stderr, "Usage: sann-genann apply <model.fann> <input.snd>\n");
		return 1;
	}

	if ((fp = fopen(argv[1], "rb")) != 0) {
		ann = genann_read(fp);
		fclose(fp);
	}
	n_out = ann->outputs;
	x = sann_data_read(argv[2], &n_rows, &n_in, &row_names, 0);
	for (i = 0; i < n_rows; ++i) {
		const float *y;
		printf("%s", row_names[i]);
		y = genann_run(ann, x[i]);
		for (j = 0; j < n_out; ++j)
			printf("\t%g", y[j] + 1.0f - 1.0f);
		putchar('\n');
		free(x[i]); free(row_names[i]);
	}
	free(x); free(row_names);

	genann_free(ann);
	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;
	if (argc == 1) {
		fprintf(stderr, "Usage: sann-genann <command> <arguments>\n");
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
