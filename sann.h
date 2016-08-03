#ifndef SANN_H
#define SANN_H

#define SAE_VERSION "r41"

#include <stdint.h>

#define SANN_MIN_SGD     1
#define SANN_MIN_RMSPROP 2

#define SANN_AF_SIGM     1
#define SANN_AF_TANH     2
#define SANN_AF_RECLIN   3

#define SAE_SC_NONE     0
#define SAE_SC_SQRT     1
#define SAE_SC_FULL     2

#define sae_n_par(n_in, n_hidden) ((n_in) * (n_hidden) + (n_in) + (n_hidden))
#define sae_par2ptr(n_in, n_hidden, p, b1, b2, w) (*(b1) = (p), *(b2) = (p) + (n_hidden), *(w) = (p) + (n_hidden) + (n_in))
#define sae_model_par2ptr(m, b1, b2, w) sae_par2ptr((m)->n_in, (m)->n_hidden, (m)->t, (b1), (b2), (w))
#define sae_buf_size(n_in, n_hidden) (3 * (n_in) + 2 * (n_hidden))

typedef struct {
	int32_t n_in, n_hidden, f1, k_sparse, scaled;
	float *t;
} sae_t;

typedef struct {
	int32_t n_layers; // including input and output layers
	int32_t *n_neurons; // of size $n_layers
	int32_t *af; // of size $n_layers - 1
	float *t;
} smln_t;

typedef struct {
	int32_t is_mln;
	union {
		sae_t *ae;
		smln_t *mln;
	} m;
} sann_t;

typedef struct {
	int malgo;
	int mini_batch; // mini-batch size
	float r; // ratio of noises
	float h; // SGD/RMSprop learning rate
	float decay; // for RMSprop
	float L2_par; // L2 regularization, for mln only
} sann_tconf_t;

#ifdef __cplusplus
extern "C" {
#endif

sae_t *sae_init(int n_in, int n_hidden, int scaled);
void sae_destroy(sae_t *m);
float sae_run(const sae_t *m, const float *x, float *z, float *y);
int sae_dump(const char *fn, const sae_t *m, char *const* col_names);
sae_t *sae_restore(const char *fn, char ***col_names);

int smln_n_par(int n_layers, const int32_t *n_neurons);
smln_t *smln_init(int n_layers, const int *n_neurons);
void smln_destroy(smln_t *m);
void smln_run(const smln_t *m, const float *x, float *y);
int smln_dump(const char *fn, const smln_t *m);
smln_t *smln_restore(const char *fn);

void sann_tconf_init(sann_tconf_t *t, int malgo);
float sann_train(void *model, const sann_tconf_t *tc, int n, float *const* x, float *const* y);

float **sann_data_read(const char *fn, int *n_rows, int *n_cols, char ***row_names, char ***col_names);

void sae_model_print(const sae_t *m);

#ifdef __cplusplus
}
#endif

#endif
