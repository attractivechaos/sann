#ifndef SANN_PRIV_H
#define SANN_PRIV_H

/**
 * This header contains low-level structures and functions. It is not intended
 * to be included by enduser programs.
 */

#include "sann.h"

typedef const float *cfloat_p;
typedef float *float_p;
typedef const char *ccstr_p;

typedef float (*sann_activate_f)(float t, float *deriv);
typedef void (*sann_gradient_f)(int n, const float *x, float *gradient, void *data);

typedef struct sfnn_buf_t {
	cfloat_p *w, *b;
	float_p *db, *dw;
	float *buf;
	float **out, **deriv, **delta;
} sfnn_buf_t;

#define sae_n_par(n_in, n_hidden) ((n_in) * (n_hidden) + (n_in) + (n_hidden))
#define sae_par2ptr(n_in, n_hidden, p, b1, b2, w) (*(b1) = (p), *(b2) = (p) + (n_hidden), *(w) = (p) + (n_hidden) + (n_in))
#define sae_buf_size(n_in, n_hidden) (3 * (n_in) + 2 * (n_hidden))

#ifdef __cplusplus
extern "C" {
#endif

float sann_sigm(float x, float *deriv);
float sann_tanh(float x, float *deriv);
float sann_reclin(float x, float *deriv);

sann_activate_f sann_get_af(int type);
float sann_sigm_cost(float y0, float y);

double sann_normal(int *iset, double *gset);

float sann_sdot(int n, const float *x, const float *y);
void sann_saxpy(int n, float a, const float *x, float *y);

void sann_SGD(int n, float h, float *t, float *g, sann_gradient_f func, void *data);
void sann_RMSprop(int n, float h0, const float *h, float decay, float *t, float *g, float *r, sann_gradient_f func, void *data);

void sae_core_randpar(int n_in, int n_hidden, float *t, int scaled);
void sae_core_forward(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, float r, const float *x, float *z, float *y, float *deriv1, int scaled);
void sae_core_backprop(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, float r, const float *x, float *d, float *buf, int scaled);

void sfnn_core_randpar(int n_layers, const int32_t *n_neurons, float *t);
void sfnn_core_forward(int n_layers, const int32_t *n_neurons, const int32_t *af, float r_in, float r_hidden, cfloat_p t, cfloat_p x, sfnn_buf_t *b);
void sfnn_core_backprop(int n_layers, const int32_t *n_neurons, const int32_t *af, float r_in, float r_hidden, cfloat_p t, cfloat_p x, cfloat_p y, float *g, sfnn_buf_t *b);
void sfnn_core_jacobian(int n_layers, const int32_t *n_neurons, const int32_t *af, cfloat_p t, cfloat_p x, int w, float *d, sfnn_buf_t *b);

int sfnn_n_par(int n_layers, const int32_t *n_neurons);
sfnn_buf_t *sfnn_buf_init(int n_layers, const int32_t *n_neurons, cfloat_p t);
void sfnn_buf_destroy(sfnn_buf_t *b);

#ifdef __cplusplus
}
#endif

#endif
