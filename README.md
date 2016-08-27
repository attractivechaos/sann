## <a name="started"></a>Getting Started
```sh
git clone http://github.com/attractivechaos/sann
cd sann && make all demo
wget -O- URL-to-mnist-data | tar xf -
./sann train -o mnist-mln.snm train-x.snd.gz train-y.snd.gz
./sann test mnist-mln.snm test-x.snd.gz > test-out.snd
mnist/eval.pl test-y.snd.gz test-out.snd
```

## Table of Contents

- [Getting Started](#started)
- [Introduction](#intro)
  - [Features](#feat)
  - [Limitations](#limit)
- [Guide to the Command-Line Tool](#cli)
  - [The SANN Data Format (SND)](#snd)
  - [Model training](#cli-train)
  - [Applying a trained model](#cli-apply)
- [Guide to the SANN Library](#api-guide)
- [Hackers' Guide](#hacker)


## <a name="intro"></a>Introduction

SANN is a lightweight, standalone and efficient C library as well as a
command-line tool that implements multi-layer [feedforward neural
networks][fnn] (FNN; not including [CNN][cnn], though) and tied-weight
[autoencoders][ae] (AE). It comes with some recent techniques to improve
training, such as [RMSprop][rmsprop] and [dropout][dropout], and fixes a couple
of theoretical flaws in tied-weight denoising AEs which may reduce the
accuracy. On [MNIST][mnist], SANN achieves similar performance and accuracy to
an FNN implemented with [Keras][keras], better than other lightweight libraries
including [FANN][fann] and [genann][genann].

### <a name="feat"></a>Features

 * Efficient. Time-consuming inner loops are optimized to reduce cache misses
   and are vectorized with SSE. Little room for further speedup with CPU only.

 * Portable. Written in C and compatible with C++ compilers. Use only standard
   Linux libraries.

### <a name="limit"></a>Limitations

 * No [convolutional neural network][cnn] (CNN) or [recurrent neural
   network][rnn] (RNN). No [Batch Normalization][bn] (a recent technique to
   improve training). Not supporting CUDA.

 * Not foolproof. Users need to manually tune hyper-parameters, in particular
   the learning rate.


## <a name="cli"></a>Guide to the Command-Line Tool

### <a name="snd"></a>The SANN Data Format (SND)

The SANN data format (SND) is a TAB-delimited text format with each row
representing a sample vector. The file may optionally have a header line
starting with `#`, which gives the name of each field. The first column of each
data line is the name of the sample; the following columns give the values.
Here is part of SND for the MNIST data set:
```txt
#r:n  14:7  14:8  14:9  14:10 14:11 14:12 14:13 14:14 14:15 14:16 14:17
1:5   0     0     0     0     0     0     0.317 0.941 0.992 0.992 0.466
2:0   0.776 0.992 0.745 0     0     0     0     0     0     0     0
3:4   0     0     0.184 0.192 0.454 0.564 0.588 0.945 0.952 0.917 0.701
4:1   0     0     0     0     0     0.313 0.941 0.984 0.756 0.090 0
5:9   0.988 0.988 0.117 0.086 0.466 0.772 0.945 0.992 0.988 0.984 0.301
```
For training, the network output is in the same format. For MNIST:
```txt
#r:n    0   1   2   3   4   5   6   7   8   9
1:5     0   0   0   0   0   1   0   0   0   0
2:0     1   0   0   0   0   0   0   0   0   0
3:4     0   0   0   0   1   0   0   0   0   0
4:1     0   1   0   0   0   0   0   0   0   0
5:9     0   0   0   0   0   0   0   0   0   1
```

### <a name="cli-train"></a>Model training

To train an FNN, you need to provide network input and output, both in the
SND format:
```sh
sann train -h50 -e.001 input.snd.gz output.snd.gz > model.snm
```
For training, the most important parameters are 1) `-h`, the number of
hidden layers and the number of neurons in each hidden layers, and 2) `-e`, the
initial learning rate. Although SANN adjusts learning rate after each batch, it
may still yield low accuracy if the starting learning rate is off.  Users
are advised to try a few different learning rates, typically from 0.0001 to
0.1. By default, SANN uses 10% of training data for validation (option `-T`).
It stops training if it reaches the maximum number of epochs (option `-n`) or
when the validation accuracy stops improving after 10 rounds (option `-l`).
After training, SANN writes the trained model to STDOUT or to a file specified
with `-o`. It retains the input and output column names if present.

Training an AE is similar, except that SANN only needs the network input and
that only one hidden layer is allowed. In particular, you may use option `-r`
to train a [denoising autoencoder][dA].

For the time being, SANN hard codes the output activation ([sigmoid][sigm]) and
cost function ([cross-entropy][ce-cost]). As a result, the output of FNN and
the input of AE must range from 0 to 1. We may lift these constrains in future
releases.

### <a name="cli-apply"></a>Applying a trained model

To apply a trained model:
```sh
sann apply model.snm model-input.snd.gz > output.snd
```
The output is also in the SND format.


## <a name="api-guide"></a>Guide to the SANN Library

The SANN library only includes two structs. `sann_t` is the minimal
representation of an FNN model. It describes the topology (number of layers,
number of neurons in each layer and activation functions) and all the model
parameters (weights and biases). `sann_tconf_t` specifies hyperparameters
needed for training, such as learning rate, dropout rates, minibatch size, etc.
Training takes a predefined model and training parameters; prediction uses the
model only. All developer-oriented structs are defined or declared in `sann.h`.
No other header files are needed for the primary functionality.

To train a model by calling the C APIs, you should first initialize a model
with either `sann_init_fnn` or `sann_init_ae`:
```c
int n_neurons[3] = { 784, 50, 10 };
sann_t *fnn;
fnn = sann_init_fnn(3, n_neurons);
```
After creating the model, you need to set the training parameters
```c
sann_tconf_t conf;
sann_tconf_init(&conf, 0, 0);
conf.h = 0.01; // change the default learning rate
```
where the last two parameters of `sann_tconf_init` specifiy the training
algorithms for each minibatch ([RMSprop][rmsprop] by default) and for each
complete batch ([iRprop-][rprop] by default). If you don't have training data
in two-dimension arrays, you may load the data from SND files:
```c
float **input, **output;
int N, n_in, N_out;
input = sann_data_read("input.snd", &N, &n_in, 0, 0);
output = sann_data_read("output.snd", &N_out, &n_out, 0, 0);
assert(N == N_out); // check if network input matches output
```
and train the model:
```c
sann_train(fnn, &conf, N, input, output);
```
where `N` is the number of training samples, `input[i]` is the input vector of
the *i*-th sample and `output[i]` the output vector of the *i*-th sample.
After training, you may save the model to a file:
```c
sann_dump("myfnn.snm", fnn, 0, 0);
```
or apply it to a test sample:
```c
float *in1, *out1, *hidden1;
out1 = (float*)malloc(sann_n_out(fnn) * sizeof(float));
sann_apply(fnn, in1, out1, 0);
```
Remeber deallocate the model with `sann_destroy` and free two-dimension arrays
with `sann_free_vectors`.

`demo.c` gives a complete example about how to use the library.


## <a name="hacker"></a>Hackers' Guide

SANN consists of the following header and C source code files:

* `sann.h`: all developer-oriented functions and structs.

* `sann_priv.h`: low-level functions not intended to be exposed.

* `math.c`: activation functions, two vectorized [BLAS][blas] routines (sdot
  and saxpy), pseudo-random number generator and RMSprop. The majority of
  computing time is spent on functions in this file.

* `sfnn.c` and `sae.c`: barebone backprop and parameter initialization routines
  for feedforward neuron networks and autoencoders, respectively. The core of
  backprop, with optional dropout, is implemented here.

* `sann.c`: unified wrapper for FNN and AE; batch training routines.

* `io.c`: SANN model I/O.

* `data.c`: SND format parser

* `cli.c` and `cli_priv.c`: command line interface

SANN also comes with the following side recipes:

* `keras/sann-keras.py`: similar SANN functionalities implemented on top of
  [Keras][keras], used for comparison purposes.

* `mnist/`: routines to convert MNIST data to SND.


[fnn]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[fann]: http://leenissen.dk/fann/wp/
[genann]: https://github.com/codeplea/genann
[ae]: https://en.wikipedia.org/wiki/Autoencoder
[rmsprop]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp
[rprop]: https://en.wikipedia.org/wiki/Rprop
[dropout]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
[mnist]: http://yann.lecun.com/exdb/mnist/
[keras]: https://keras.io/
[bn]: https://arxiv.org/abs/1502.03167
[blas]: http://www.netlib.org/lapack/lug/node145.html
[backprop]: https://en.wikipedia.org/wiki/Backpropagation
[dA]: https://en.wikipedia.org/wiki/Autoencoder#Denoising_autoencoder
[sigm]: https://en.wikipedia.org/wiki/Sigmoid_function
[ce-cost]: https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
