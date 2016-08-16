#!/usr/bin/env python

import sys, getopt, re, gzip
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import RMSprop

def sann_data_read(fn):
	x, row_names, col_names = [], [], []

	def _process_fp(fp):
		for l in fp:
			t = l[:-1].split('\t')
			if l[0] == '#':
				col_names = t[1:]
			else:
				row_names.append(t[0])
				x.append(t[1:]);

	if re.search(r'\.gz$', fn):
		with gzip.open(fn, 'r') as fp:
			_process_fp(fp)
	else:
		with open(fn, 'r') as fp:
			_process_fp(fp)
	return np.array(x).astype('float32'), row_names, col_names

def main_train(argv):
	n_hidden, n_epochs, minibatch, lr, cost_func, outfn = 50, 20, 64, .001, 'binary_crossentropy', None

	def train_help():
		print("Usage: sann-keras train [options] <input.snd> <output.snd>")
		print("Options:")
		print("  -h INT     number of hidden neurons [50]")
		print("  -B INT     minibatch size [64]")
		print("  -e FLOAT   learning rate [0.001]")
		print("  -o FILE    save model to FILE []")
		print("  -n INT     number of epochs [20]")
		sys.exit(1)

	try:
		opts, args = getopt.getopt(argv, "h:n:B:o:e:c")
	except getopt.GetoptError:
		train_help()
	if len(args) < 2:
		train_help()

	for opt, arg in opts:
		if opt == '-h': n_hidden = int(arg)
		elif opt == '-n': n_epochs = int(arg)
		elif opt == '-B': minibatch = int(arg)
		elif opt == '-o': outfn = arg
		elif opt == '-e': lr = float(arg)
		elif opt == '-c': cost_func = 'categorical_crossentropy'

	x, x_rnames, x_cnames = sann_data_read(args[0])
	y, y_rnames, y_cnames = sann_data_read(args[1])
	model = Sequential()
	model.add(Dense(n_hidden, input_dim=len(x[0]), activation='relu'))
	model.add(Dense(len(y[0]), activation='sigmoid'))
	model.compile(loss=cost_func, optimizer=RMSprop(lr=lr), metrics=['accuracy'])
	model.fit(x, y, nb_epoch=n_epochs, batch_size=minibatch)
	if outfn: model.save(outfn)

def main_apply(argv):
	if len(argv) < 2:
		print("Usage: sann-keras apply <model> <input.snd>")
		sys.exit(1)
	model = load_model(argv[0])
	x, x_rnames, x_cnames = sann_data_read(argv[1])
	y = model.predict(x)
	for i in range(len(y)):
		sys.stdout.write(x_rnames[i])
		for j in range(len(y[i])):
			sys.stdout.write("\t%g" % y[i][j])
		sys.stdout.write('\n')

def main(argv):
	if len(argv) < 2:
		print("Usage: sann-keras <command> <arguments>")
		print("Command:")
		print("  train     train the model")
		print("  apply     apply the model")
		sys.exit(1)
	elif argv[1] == 'train':
		main_train(argv[2:])
	elif argv[1] == 'apply':
		main_apply(argv[2:])

if __name__ == "__main__":
	main(sys.argv)
