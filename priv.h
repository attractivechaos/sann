#ifndef SANN_PRIV_H
#define SANN_PRIV_H

typedef const float *cfloat_p;
typedef float *float_p;

typedef float (*sann_activate_f)(float t, float *deriv);
typedef void (*sann_gradient_f)(int n, const float *x, float *gradient, void *data);

typedef struct smln_buf_t {
	cfloat_p *w, *b;
	float_p *db, *dw;
	float *buf;
	float **out, **deriv, **delta;
} smln_buf_t;

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

void smln_core_backprop(int n_layers, const int32_t *n_neurons, const int32_t *af, cfloat_p t, cfloat_p x, cfloat_p y, float *g, smln_buf_t *b);
void sae_core_backprop(int n_in, int n_hidden, const float *t, sann_activate_f f1, sann_activate_f f2, int k_sparse, float r, const float *x, float *d, float *buf, int scaled);

smln_buf_t *smln_buf_init(int n_layers, const int32_t *n_neurons, cfloat_p t);
void smln_buf_destroy(smln_buf_t *b);

#ifdef __cplusplus
}
#endif

#endif
