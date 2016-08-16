#ifndef SANN_H
#define SANN_H

#define SANN_VERSION "r38"

#include <stdint.h>

#define SANN_MIN_MINI_SGD     1
#define SANN_MIN_MINI_RMSPROP 2

#define SANN_MIN_BATCH_FIXED  1
#define SANN_MIN_BATCH_RPROP  2

#define SANN_AF_SIGM     1
#define SANN_AF_TANH     2
#define SANN_AF_RECLIN   3

#define SAE_SC_NONE     0
#define SAE_SC_SQRT     1
#define SAE_SC_FULL     2

#define sann_n_in(m) ((m)->n_neurons[0])
#define sann_n_out(m) ((m)->n_neurons[(m)->n_layers - 1])
#define sae_n_hidden(m) ((m)->n_neurons[1])
#define sae_n_in(m) sann_n_in(m)

extern int sann_verbose;

typedef struct {
	int32_t is_mln;
	int32_t k_sparse, scaled; // so far these two are for AE only
	int32_t n_layers;
	int32_t *n_neurons;
	int32_t *af;
	float *t;
} sann_t;

typedef struct {
	int n_epochs;
	float vfrac; // fraction of samples used for validation

	float L2_par; // L2 regularization, for mlnn only
	float r; // ratio of noises, for autoencoder only
	float h; // learning rate

	// optimizer for minibatches
	int malgo;
	int mini_batch; // mini-batch size
	float decay; // for RMSprop

	// optimizer for complete batches
	int balgo;
	float h_min, h_max; // for RPROP
	float rprop_dec, rprop_inc;
	int max_inc; // stop training if cost on validation samples increases for $max_inc epochs continuously
} sann_tconf_t;

#ifdef __cplusplus
extern "C" {
#endif

sann_t *sann_init_ae(int n_in, int n_hidden, int scaled);
sann_t *sann_init_mln(int n_layers, const int *n_neurons);
void sann_destroy(sann_t *m);
void sann_apply(const sann_t *m, const float *x, float *y, float *z);
float sann_cost(int n, const float *y0, const float *y);
int sann_n_par(const sann_t *m);

int sann_dump(const char *fn, const sann_t *m, char *const* col_names_in, char *const* col_names_out);
sann_t *sann_restore(const char *fn, char ***col_names_in, char ***col_names_out);
void sann_free_names(int n, char **s);
void sann_free_vectors(int n, float **x);

void sann_tconf_init(sann_tconf_t *t, int balgo, int malgo);
float sann_train_epoch(sann_t *m, const sann_tconf_t *tc, const float *h, int n, float *const* x, float *const* y, float **_buf);
int sann_train(sann_t *m, const sann_tconf_t *_tc, int N, float *const* x, float *const* y);
float sann_test(const sann_t *m, int n, float *const* x, float *const* y);

float **sann_data_read(const char *fn, int *n_rows, int *n_cols, char ***row_names, char ***col_names);
void sann_data_shuffle(int n, float **x, float **y, char **names);

void sann_print(const sann_t *m);

#ifdef __cplusplus
}
#endif

#endif
