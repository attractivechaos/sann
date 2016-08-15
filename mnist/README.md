This directory contains tools to convert [MNIST][mnist] data to the SANN data
format (SND). The following command lines show the procedure:
```sh
# download the MNIST data set
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# compile mnist2snd
make

# generate the input (truth encoded in the row names)
./mnist2snd train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz | gzip > train-x.snd.gz
./mnist2snd t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz | gzip > test-x.snd.gz

# generate the output (extract the truth from the row names)
gzip -dc train-x.snd.gz | ./gen_truth.pl | gzip > train-y.snd.gz
gzip -dc test-x.snd.gz | ./gen_truth.pl | gzip > test-y.snd.gz

# train SANN (for r19, this stops at epoch 22)
../sann train -n50 -e.001 train-x.snd.gz train-y.snd.gz > model.snm

# apply to test samples (for r19, this gives 2.87% error rate)
../sann apply model.snm test-x.snd.gz | ./eval.pl
```

[mnist]: http://yann.lecun.com/exdb/mnist/
