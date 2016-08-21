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

SANN is a portable and efficient C library as well as a command-line tool for
working with multi-layer [feedforward neural networks][fnn] (FNN; not including
[CNN][cnn], though) and [autoencoders][ae] (AE). It implements some of the
recent techniques to improve training, such as [RMSprop][rmsprop] and
[dropout][dropout], and also fixes a couple of theoretical flaws in tied-weight
denoising AEs. On [MNIST][mnist], SANN achieves similar performance and
accuracy to an FNN implemented with [Keras][keras].

[fnn]: https://en.wikipedia.org/wiki/Feedforward_neural_network
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[ae]: https://en.wikipedia.org/wiki/Autoencoder
[rmsprop]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp
[dropout]: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
[mnist]: http://yann.lecun.com/exdb/mnist/
[keras]: https://keras.io/
