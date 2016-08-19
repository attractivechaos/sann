#ifndef SANN_H
#define SANN_H

#define SANN_VERSION "r60"

#include <stdint.h>

//! mini-batch minimization algorithm
#define SANN_MIN_MINI_SGD     1
#define SANN_MIN_MINI_RMSPROP 2

//! how to adjust learning rate after an entire batch
#define SANN_MIN_BATCH_FIXED  1  //! fixed learning rate
#define SANN_MIN_BATCH_RPROP  2  //! iRprop- (Igel and Husken, 2000)

//! activation functions
#define SANN_AF_SIGM     1  //! sigmoid
#define SANN_AF_TANH     2  //! tanh
#define SANN_AF_RECLIN   3  //! rectified linear, aka. ReLU

//! autoencoder scaling
#define SAE_SC_NONE     0   //! no scaling (standard autoencoder)
#define SAE_SC_SQRT     1   //! scaled by 1/sqrt(n_neurons_in_prev_layer); this is the default
#define SAE_SC_FULL     2   //! scaled by 1/n_neurons_in_prev_layer

//! number of input neurons
#define sann_n_in(m) ((m)->n_neurons[0])

//! number of output neurons
#define sann_n_out(m) ((m)->n_neurons[(m)->n_layers - 1])

//! number of hidden neurons in an autoencoder
#define sae_n_hidden(m) ((m)->n_neurons[1])

//! number of input neurons in an autoencoder (an alias of sann_n_in())
#define sae_n_in(m) sann_n_in(m)

//! verbose level. 0: no stderr output; 1: error only; 2: error+warning; 3: error+warning+message (default)
extern int sann_verbose;

//! SANN model
typedef struct {
	int32_t is_mln;     //! whether the model is MLNN or AE 
	int32_t k_sparse;   //! at most $k_sparse hidden neurons activate (AE only; see also arXiv:1312.5663)
	int32_t scaled;     //! how to scale the weight; valid values defined by SAE_SC_* macros (AE only)
	int32_t n_layers;   //! number of layers; always 3 for autoencoder
	int32_t *n_neurons; //! number of neurons in each layer; of size $n_layers
	int32_t *af;        //! activation function; valid values defined by SANN_AF_* macros; of size $n_layers-1
	float *t;           //! array of all parameters; size computed by function sann_n_par()
} sann_t;

//! training parameters
typedef struct {
	int n_epochs;       //! max number of epochs for training
	int max_inc;        //! stop training if cost on validation samples increases for $max_inc epochs continuously
	float vfrac;        //! fraction of samples used for validation

	float L2_par;       //! L2 regularization (MLNN only)
	float r;            //! ratio of noises (autoencoder only)
	float h;            //! learning rate, or starting learning rate

	// optimizer for mini-batches
	int malgo;          //! mini-batch minimization algorithm; values defined by SANN_MIN_MINI_* macros
	int mini_batch;     //! mini-batch size
	float decay;        //! for RMSprop

	// optimizer for complete batches
	int balgo;          //! complete-batch minimization algorithm; values defined by SANN_MIN_BATCH_* macros
	float h_min, h_max; //! min and max learning rate for iRprop-
	float rprop_dec, rprop_inc; //! learning rate adjusting factors for iRprop-
} sann_tconf_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize an autoencoder with tied weights
 *
 * @param n_int      number of input/output neurons
 * @param n_hidden   number of hidden neurons
 * @param scaled     how to scahle the weight; valid values defined by SAE_SC_*
 *
 * @return pointer to the model
 */
sann_t *sann_init_ae(int n_in, int n_hidden, int scaled);

/**
 * Initialize a multi-layer/feedforward neuron network
 *
 * @param n_layers   number of layers
 * @param n_neurons  number of neurons in each layer, an array of size $n_layers
 *
 * @return pointer to the model
 */
sann_t *sann_init_mln(int n_layers, const int *n_neurons);

/**
 * Deallocate a model
 *
 * @param m          pointer to the model
 */
void sann_destroy(sann_t *m);

void sann_apply(const sann_t *m, const float *x, float *y, float *z);
float sann_cost(int n, const float *y0, const float *y);
int sann_n_par(const sann_t *m);

int sann_dump(const char *fn, const sann_t *m, char *const* col_names_in, char *const* col_names_out);
sann_t *sann_restore(const char *fn, char ***col_names_in, char ***col_names_out);
void sann_free_names(int n, char **s);
void sann_free_vectors(int n, float **x);

void sann_tconf_init(sann_tconf_t *t, int balgo, int malgo);
int sann_train(sann_t *m, const sann_tconf_t *_tc, int N, float *const* x, float *const* y);
float sann_test(const sann_t *m, int n, float *const* x, float *const* y);

float **sann_data_read(const char *fn, int *n_rows, int *n_cols, char ***row_names, char ***col_names);
void sann_data_shuffle(int n, float **x, float **y, char **names);

#ifdef __cplusplus
}
#endif

#endif
