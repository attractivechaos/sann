#ifndef SANN_H
#define SANN_H

#define SANN_VERSION "r68"

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
	int32_t scaled;     //! how to scale the weight; valid values defined by SAE_SC_* macros (AE only)
	int32_t n_layers;   //! number of layers; always 3 for autoencoder
	int32_t *n_neurons; //! number of neurons in each layer; of size $n_layers
	int32_t *af;        //! activation function; values defined by SANN_AF_*; of size $n_layers-1; output MUST BE sigmoid
	float *t;           //! array of all parameters; size computed by function sann_n_par()
} sann_t;

//! training parameters
typedef struct {
	int n_epochs;       //! max number of epochs for training
	int max_inc;        //! stop training if cost on validation samples increases for $max_inc epochs continuously
	float vfrac;        //! fraction of samples used for validation

	float L2_par;       //! L2 regularization (MLNN only)
	float r_in;         //! input neuron dropout rate
	float r_hidden;     //! hidden neuron dropout rate (MLNN only for now; can be applied to AE in principle)
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

/**
 * Total number of parameters
 *
 * @param m          the model
 *
 * @return number of parameters
 */
int sann_n_par(const sann_t *m);

/**
 * Apply the model to data
 *
 * @param m          the model
 * @param x          input, an array of size sann_n_in(m)
 * @param y          output, an array of size sann_n_out(m)
 * @param z          hidden activation, an array of size sae_n_hidden(m) - autoencoder only; use NULL for MLNN
 */
void sann_apply(const sann_t *m, const float *x, float *y, float *z);

/**
 * Compute the sigmoid cost of two output vectors
 *
 * @param n          dimension
 * @param y0         truth
 * @param y          prediction
 *
 * @return averaged sigmoid cost per output neuron
 */
float sann_cost(int n, const float *y0, const float *y);

/**
 * Initialize training parameters
 *
 * @param t          training parameters
 * @param balgo      algorithm to adjust learning rate after a complete batch; values defined SANN_MIN_BATCH_*
 * @param malgo      mini-batch training algorith; values defined by SANN_MIN_MINI_*
 */
void sann_tconf_init(sann_tconf_t *t, int balgo, int malgo);

/**
 * Train for multiple epochs
 *
 * This function takes the first $N * (1 - $tc->vfrac) samples as training
 * samples and the rest as validation samples. It stops if the cost of
 * validation samples does not imporve after $tc->max_inc epochs. The final
 * model is taken at the epoch that is over $tc->max_inc and optimizes the
 * validation samples. To avoid batch effect, it is highly recommended to
 * shuffle the input with sann_data_shuffle() before calling this function.
 *
 * @param m          the model
 * @param tc         traning parameters
 * @param N          number of samples
 * @param x          input data; x[i] is a vector of size sann_n_in(m)
 * @param y          truth output data; NULL for autoencoder; for mlnn, y[i] is a vector of size sann_n_out(m)
 *
 * @return number of epochs
 */
int sann_train(sann_t *m, const sann_tconf_t *tc, int N, float *const* x, float *const* y);

/**
 * Compute the per-neuron cost given truth
 *
 * @param m          the model
 * @param n          number of samples
 * @param x          input data; x[i] is a vector of size sann_n_in(m)
 * @param y          truth output; NULL for autoencoder; for mlnn, y[i] is a vector of size sann_n_out(m)
 *
 * @return averaged sigmoid cost per sample per output neuron
 */
float sann_evaluate(const sann_t *m, int n, float *const* x, float *const* y);

/**
 * Save the model
 *
 * @param fn         output file name; NULL or "-" for stdout
 * @param m          the model
 * @param cnames_in  input column names, of size sann_n_in(m); can be NULL if not available
 * @param cnames_out output column names, of size sann_n_out(m); can be NULL if not available
 *
 * @return 0 for success; others for errors
 */
int sann_dump(const char *fn, const sann_t *m, char *const* cnames_in, char *const* cnames_out);

/**
 * Load the model
 *
 * @param fn         input file name
 * @param cnames_in  input column names, of size sann_n_in(m); set to NULL if not available; can be NULL if not needed
 * @param cnames_out output column names, of size sann_n_out(m); set to NULL if not available; can be NULL if not needed
 *
 * @return the model; NULL on error
 */
sann_t *sann_restore(const char *fn, char ***cnames_in, char ***cnames_out);

/**
 * Read data from file in the SANN data format (SND)
 *
 * @param fn         file name
 * @param n_rows     number of samples
 * @param n_cols     number of data columns
 * @param row_names  row names (the 1st column of each data row); can be NULL if not needed
 * @param col_names  column names (the 1st row starting with #); can be NULL if not needed
 *
 * @return 2-d array of dimension [n_rows][n_cols]
 */
float **sann_data_read(const char *fn, int *n_rows, int *n_cols, char ***row_names, char ***col_names);

/**
 * Shuffle samples (important when using validation samples)
 *
 * @param n          number of samples
 * @param x          input vectors
 * @param y          output vectors; can be NULL
 * @param names      row names; can be NULL
 */
void sann_data_shuffle(int n, float **x, float **y, char **names);

/**
 * Free a list of names
 *
 * @param n          number of strings
 * @param s          pointer to strings
 */
void sann_free_names(int n, char **s);

/**
 * Free a list of vectors
 *
 * @param n          number of vectors
 * @param s          pointer to vectors
 */
void sann_free_vectors(int n, float **x);

#ifdef __cplusplus
}
#endif

#endif
