#ifndef HL_MNIST_H
#define HL_MNIST_H

#include <stdint.h>

typedef struct {
	int n_dat, n_row, n_col;
	uint8_t *dat;
} dan_mnist_t;

#ifdef __cplusplus
extern "C" {
#endif

dan_mnist_t *dan_mnist_read(const char *fn, int k);
void dan_mnist_destroy(dan_mnist_t *mn);

#ifdef __cplusplus
}
#endif

#endif
