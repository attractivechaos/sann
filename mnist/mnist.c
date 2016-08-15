#include <zlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "mnist.h"
#include "kseq.h"
KSTREAM_INIT(gzFile, gzread, 65536)

static inline int dan_is_big_endian()
{
	long one= 1;
	return !(*((char *)(&one)));
}

static inline uint32_t dan_swap_endian_4(uint32_t v)
{
	v = ((v & 0x0000FFFFU) << 16) | (v >> 16);
	return ((v & 0x00FF00FFU) << 8) | ((v & 0xFF00FF00U) >> 8);
}

dan_mnist_t *dan_mnist_read(const char *fn, int k)
{
	gzFile fp;
	kstream_t *ks;
	dan_mnist_t *mn = 0;
	uint32_t x[4];
	int is_be, i;

	assert(k == 1 || k == 3);
	is_be = dan_is_big_endian();

	uint64_t size, ret;
	fp = fn == 0 || strcmp(fn, "-") == 0? gzdopen(fileno(stdin), "r") : gzopen(fn, "r");
	ks = ks_init(fp);
	ks_read(ks, (uint8_t*)&x, 4 * (k + 1));
	if (!is_be)
		for (i = 0; i <= k; ++i)
			x[i] = dan_swap_endian_4(x[i]);
	assert(x[0] == (0x800 | k));
	if (k == 1) x[2] = x[3] = 1;
	mn = (dan_mnist_t*)calloc(1, sizeof(dan_mnist_t));
	mn->n_dat = x[1], mn->n_row = x[2], mn->n_col = x[3];
	size = (uint64_t)mn->n_dat * mn->n_row * mn->n_col;
	mn->dat = (uint8_t*)calloc(size, 1);
	ret = ks_read(ks, mn->dat, size);
	assert(size == ret);
	ks_destroy(ks);
	gzclose(fp);
	return mn;
}

void dan_mnist_destroy(dan_mnist_t *mn)
{
	if (mn == 0) return;
	free(mn->dat); free(mn);
}

void dan_mnist2pgm(FILE *fp, const dan_mnist_t *mn, int w)
{
	int i, j;
	const uint8_t *dat;
	if (mn == 0 || mn->n_row < 2 || mn->n_col < 2 || w >= mn->n_dat) return;
	fprintf(fp, "P2\n%d %d\n255\n", mn->n_col, mn->n_row);
	dat = &mn->dat[w * mn->n_row * mn->n_col];
	for (i = 0; i < mn->n_row; ++i) {
		for (j = 0; j < mn->n_col; ++j)
			fprintf(fp, "%-3d ", dat[i * mn->n_col + j]);
		fprintf(fp, "\n");
	}
}
