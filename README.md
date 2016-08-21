## Getting Started
```sh
git clone http://github.com/attractivechaos/sann
cd sann && make
wget -O- URL-to-mnist-data | tar xf -
./sann train -o mnist-mln.snm train-x.snd.gz train-y.snd.gz
./sann test mnist-mln.snm test-x.snd.gz > test-out.snd
mnist/eval.pl test-y.snd.gz test-out.snd
```

## Introduction

SANN is a lightweight, standalone and efficient C library as well as a
command-line tool that implements multi-layer [feedforward neural
networks][fnn] (FNN; not including [CNN][cnn], though) and tied-weight
[autoencoders][ae] (AE). It comes with some recent techniques to improve
training, such as [RMSprop][rmsprop] and [dropout][dropout], and fixes a couple
of theoretical flaws in tied-weight denoising AEs which may reduce the
accuracy. On [MNIST][mnist], SANN achieves similar performance and accuracy to
an FNN implemented with [Keras][keras], better than other lightweight libraries
including [FANN][fann] and [genann][genann].

### Features

* Efficient. Time-consuming inner loops are optimized to reduce cache misses
  and are vectorized with SSE. Little room for further speedup with CPU only.

* Portable. Written in C and compatible with C++ compilers. Use only standard
  Linux libraries.

### Limitations

 * No [convolutional neural network][cnn] (CNN) or [recurrent neural network][rnn] (RNN).

 * No [Batch Normalization][bn].

## Guide to the Command-Line Tool

### The SANN Data Format (SND)

The SANN data format (SND) is a TAB-delimited text format with each row
representing a sample vector. The file may optionally have a header line
providing the name of each field. The first column of each data line is the
name of the sample; the following columns give the values.

[fnn]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[fann]: http://leenissen.dk/fann/wp/
[genann]: https://github.com/codeplea/genann
[ae]: https://en.wikipedia.org/wiki/Autoencoder
[rmsprop]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp
[dropout]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
[mnist]: http://yann.lecun.com/exdb/mnist/
[keras]: https://keras.io/
[bn]: https://arxiv.org/abs/1502.03167
[blas]: http://www.netlib.org/lapack/lug/node145.html
[backprop]: https://en.wikipedia.org/wiki/Backpropagation
