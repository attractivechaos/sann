#include <stdio.h>
#include "sann.h"

void sae_model_print(const sae_t *m)
{
	const float *b1, *b2, *w10;
	int i, j;
	sae_model_par2ptr(m, &b1, &b2, &w10);
	printf("NI\t%d\n", m->n_in);
	printf("NH\t%d\n", m->n_hidden);
	printf("KS\t%d\n", m->k_sparse);
	printf("F1\t%d\n", m->f1);
	printf("SC\t%d\n", m->scaled);
	printf("B1");
	for (i = 0; i < m->n_hidden; ++i)
		printf("\t%g", b1[i]);
	printf("\nB2");
	for (i = 0; i < m->n_in; ++i)
		printf("\t%g", b2[i]);
	putchar('\n');
	for (j = 0; j < m->n_hidden; ++j) {
		const float *w10j = w10 + j * m->n_in;
		printf("WW\t%d", j);
		for (i = 0; i < m->n_in; ++i)
			printf("\t%g", w10j[i]);
		putchar('\n');
	}
}
