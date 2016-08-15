#include <stdio.h>
#include "mnist.h"

int main(int argc, char *argv[])
{
	dan_mnist_t *img, *lbl;
	int i, j, k;
	if (argc < 3) {
		fprintf(stderr, "Usage: mnist2snd <images> <labels>\n");
		return 1;
	}
	img = dan_mnist_read(argv[1], 3);
	lbl = dan_mnist_read(argv[2], 1);
	if (img->n_dat != lbl->n_dat) {
		fprintf(stderr, "ERROR: inconsistant images and labels\n");
		return 1; // FIXME: memory leak here
	}
	printf("#no:truth");
	for (i = 0; i < img->n_row; ++i)
		for (j = 0; j < img->n_col; ++j)
			printf("\t%d:%d", i, j);
	putchar('\n');
	for (k = 0; k < img->n_dat; ++k) {
		const uint8_t *dat = &img->dat[k * img->n_row * img->n_col];
		printf("%d:%d", k+1, lbl->dat[k]);
		for (i = 0; i < img->n_row; ++i) {
			const uint8_t *dati = &dat[i * img->n_col];
			for (j = 0; j < img->n_col; ++j)
				printf("\t%g", dati[j] / 255.);
		}
		putchar('\n');
	}
	dan_mnist_destroy(lbl);
	dan_mnist_destroy(img);
	return 0;
}
