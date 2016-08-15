This directory implements a command-line interface to the [libfann](fann)
library. To compile it, you need to have libfann preinstalled on your system,
and then
```sh
# compile (FANN_PREFIX not needed if libfann is installed in system directories)
make FANN_PREFIX=/path/to/fann/install

# train
./sann-fann train -o mnist.fann mnist-train-x.snd mnist-train-y.snd

# apply the model to test data
./sann-fann apply mnist.fann mnist-test-x.snd > predict.snd

# evalute (this gives an error rate of 5.99%)
../mnist/eval.pl predict.snd
```
Note that the command-line interface is mostly using the default training
parameters. For now, it may not demonstrate the full power of libfann.

[fann]: http://leenissen.dk/fann/wp/
