import sys
sys.path.insert(0,'/home/2013/adoomr/.local/lib/python2.7/site-packages')
sys.path.remove('/usr/lib/python2.7') # This is because Python versions were conflicting and I want to use the one from anaconda. 

import matplotlib
matplotlib.use('Agg') # So that I can use matshow while SSHing. 
import matplotlib.pyplot as plt

from lasagne.regularization import regularize_layer_params, l2
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import glob
import lasagne
import numpy as np
import os
import os
import theano
import theano.tensor as T
import timeit

#########################
#### Hyperparameters ####
#########################

NUM_EPOCHS = 8000
LEARNING_RATE = 0.005
N_CLASSES = 15
L2_REG = 0.0001
BATCH_SIZE = 16
VALIDATION_FREQUENCY = 50
PARAM_SAVE_DIR = './params'
FIGURE_SAVE_DIR = './figures'
DATASET_DIR = '/home/ml/adoomr/Archive/mlproject4/sanctuary/lbp_dataset_2_8'
INIT_PATIENCE = np.inf # if after INIT_PATIENCE validation checks, the score is not improved, then the training stops

def plot_curves(x, y1, y2, curve1_name, curve2_name, savename):
	plt.grid()
	plt.plot(x, y1, '-', color='r', label=curve1_name)
	plt.plot(x, y2, '-', color='g', label=curve2_name)
	# plt.axis([0, max(np.asarray(x)), 0, 1.3*max(np.maximum(y1,y2))])
	plt.legend(loc='best')
	plt.xlabel('Iteration')
	if 'error' in savename:
		plt.ylabel('Error')
	else:
		plt.ylabel('Cost')
	plt.savefig(os.path.join(FIGURE_SAVE_DIR, savename))
	plt.clf()
	plt.close()
	return

def plot_cm(pred_y, true_y, title, filename, display_threshold=0.15):
	cm = confusion_matrix(true_y, pred_y)
	cm = cm.astype('float')/cm.sum(axis=1)
	plt.matshow(cm, cmap=plt.cm.Greens)
	plt.title(title)
	plt.xlabel('Predicted Class')
	plt.ylabel('Actual Class')
	plt.colorbar()

	ax = plt.gca()
	ax.xaxis.set_ticks_position('bottom')

	for x in range(true_y.max()+1):
		for y in range(true_y.max()+1):
			if cm[x,y]>=display_threshold:
				ax.annotate('%4.2f' % cm[x,y], xy=(y,x), horizontalalignment='center', verticalalignment='center', size=6.3)

	plt.savefig(os.path.join(FIGURE_SAVE_DIR, filename))
	plt.clf()
	plt.close()
	return



print "[X] Emerald is loading its data."
# Because the given data gives classes from 1 to 15, we subtract 1 so that we get classes from 0 to 14. 
trainX = np.load(os.path.join(DATASET_DIR, 'lbp_trainX.npy'))
trainY = np.load(os.path.join(DATASET_DIR, 'lbp_trainY.npy'))-1

valX = np.load(os.path.join(DATASET_DIR, 'lbp_valX.npy'))
valY = np.load(os.path.join(DATASET_DIR, 'lbp_valY.npy'))-1

testX = np.load(os.path.join(DATASET_DIR, 'lbp_testX.npy'))
testY = np.load(os.path.join(DATASET_DIR, 'lbp_testY.npy'))-1

# The feature values are LBP histogram counts. They are always greater than or equal to zero so we chose a max normalization according to the max value in the training set but we multiply it by 1.1 to get a higher maximum in case there is a max value even greater in the testing set. 
norm_value = 1.1*trainX.max()
trainX /= norm_value
valX /= norm_value
testX /= norm_value

# Shuffle the data.
trainX, trainY = shuffle(trainX, trainY)
valX, valY = shuffle(valX, valY)
testX, testY = shuffle(testX, testY)

################
#### Layers ####
################
print "[X] Emerald building its layers."
model = lasagne.layers.InputLayer((BATCH_SIZE, trainX.shape[1]))
model = lasagne.layers.DenseLayer(model, num_units=256)
model = lasagne.layers.DenseLayer(model, num_units=256)
model = lasagne.layers.DenseLayer(model, num_units=N_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)

x = T.matrix()
y = T.ivector()

print "[X] Emerald defining its goals."
model_params = lasagne.layers.get_all_params(model, trainable=True)
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

# All noisy variables are in the event we want to use dropout or add Gaussian noise to the training. 

noisy_output = lasagne.layers.get_output(model, x, deterministic=False)
true_output = lasagne.layers.get_output(model, x, deterministic=True)

noisy_prediction = T.argmax(noisy_output, 1)
true_prediction = T.argmax(true_output, 1)

l2_loss = regularize_layer_params(model, l2)*L2_REG

noisy_cost = T.mean(T.nnet.categorical_crossentropy(noisy_output, y)) + l2_loss
true_cost = T.mean(T.nnet.categorical_crossentropy(true_output, y)) + l2_loss

# noisy_error = T.mean(T.neq(T.argmax(noisy_output, 1), y))
# true_error = T.mean(T.neq(T.argmax(true_output, 1), y))
noisy_error = 1-T.mean(lasagne.objectives.categorical_accuracy(noisy_output, y))
true_error = 1-T.mean(lasagne.objectives.categorical_accuracy(true_output, y))

updates = lasagne.updates.sgd(noisy_cost, model_params, learning_rate=sh_lr)

train = theano.function([x,y], [noisy_cost, noisy_error, noisy_output, noisy_prediction], updates=updates, 
	allow_input_downcast=True)

get_score = theano.function([x,y], [true_cost, true_error, true_output, true_prediction], 
	allow_input_downcast=True)

best_validation_cost = np.inf
best_validation_error = np.inf

best_test_cost = np.inf
best_test_error = np.inf


best_iter = 0
start_time = timeit.default_timer()

n_train_batches = int(np.ceil(trainX.shape[0] / float(BATCH_SIZE)))

plot_iters = []

plot_train_cost = []
plot_train_error = []

plot_valid_cost = []
plot_valid_error = []

plot_test_cost = []
plot_test_error = []

epoch = 0
done_looping = False
patience = INIT_PATIENCE

print "[X] Emerald begins its training."
while not done_looping:
	try:
		epoch = epoch + 1
		trainX, trainY = shuffle(trainX, trainY)
		for minibatch_index in xrange(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index	
			if iter % 100 == 0:
				print "[O] Training at iteration %d." % iter
			cost_ij = train(trainX[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainX.shape[0])], 
					trainY[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainY.shape[0])])
			if (iter+1) % VALIDATION_FREQUENCY == 0:
				train_cost, train_error, train_output, train_pred = get_score(trainX, trainY)
				valid_cost, valid_error, valid_output, valid_pred = get_score(valX, valY)
				test_cost, test_error, test_output, test_pred = get_score(testX, testY)

				plot_train_cost.append(train_cost)
				plot_train_error.append(train_error)

				plot_valid_cost.append(valid_cost)
				plot_valid_error.append(valid_error)

				plot_test_cost.append(test_cost)
				plot_test_error.append(test_error)

				plot_iters.append(iter)
				
				if not os.path.exists(FIGURE_SAVE_DIR):
					os.makedirs(FIGURE_SAVE_DIR)

				plot_curves(plot_iters, plot_train_cost, plot_valid_cost, 'Training Cost', 'Validation Cost', 'train_val_cost.pdf')
				plot_curves(plot_iters, plot_train_error, plot_valid_error, 'Training Error', 'Validation Error', 'train_val_error.pdf')


				# print "--> Epoch %i, minibatch %i/%i has training true cost \t %f." % (epoch, minibatch_index+1, n_train_batches, train_cost)
				# print "--> Epoch %i, minibatch %i/%i has validation true cost \t %f and error of \t %f." % (epoch, minibatch_index+1, n_train_batches, valid_cost, valid_error)

				if valid_cost < best_validation_cost:
					patience = INIT_PATIENCE
					print "----> New best score found!"
					print "--> Valid cost of %f and valid error of %f." % (valid_cost, valid_error)
					# print "--> Test cost of %f and test error of %f." % (test_cost, test_error)
					if not os.path.exists(PARAM_SAVE_DIR):
						os.makedirs(PARAM_SAVE_DIR)
					for f in glob.glob(PARAM_SAVE_DIR+'/*'):
						os.remove(f)
					all_param_values = lasagne.layers.get_all_param_values(model)
					joblib.dump(all_param_values, os.path.join(PARAM_SAVE_DIR, 'params.pkl'))
					# print "----> Parameters saved."

					plot_cm(train_pred, trainY, 'Confusion Matrix on the Training Set', 'cm_train.pdf')
					plot_cm(valid_pred, valY, 'Confusion Matrix on the Validation Set', 'cm_valid.pdf')
					plot_cm(test_pred, testY, 'Confusion Matrix on the Test Set', 'cm_test.pdf')

					best_validation_cost = valid_cost
					best_validation_error = valid_error

					best_test_cost = test_cost
					best_test_error = test_error


					best_iter = iter
				else:
					patience -= 1
					if patience <= 0:
						done_looping = True
						break
				print 'Patience:' + str(patience)

	except KeyboardInterrupt:
		done_looping = True

end_time = timeit.default_timer()

print "--> Best validation score of %f with error %f." % (best_validation_cost, best_validation_error)
print "--> Best testing score of %f with error %f." % (best_test_cost, best_test_error)
print "--> Total runtime %.2f minutes." % ((end_time-start_time) / 60.)
print "[X] Saving the scores."

plot_curves(plot_iters, plot_train_cost, plot_valid_cost, 'Training Cost', 'Validation Cost', 'train_val_cost.pdf')
plot_curves(plot_iters, plot_train_error, plot_valid_error, 'Training Error', 'Validation Error', 'train_val_error.pdf')

joblib.dump(plot_iters, os.path.join(PARAM_SAVE_DIR, "iters.pkl"))

joblib.dump(plot_train_cost, os.path.join(PARAM_SAVE_DIR, "train_cost.pkl"))
joblib.dump(plot_train_error, os.path.join(PARAM_SAVE_DIR, "train_error.pkl"))

joblib.dump(plot_valid_cost, os.path.join(PARAM_SAVE_DIR, "valid_cost.pkl"))
joblib.dump(plot_valid_error, os.path.join(PARAM_SAVE_DIR, "valid_error.pkl"))

joblib.dump(plot_test_cost, os.path.join(PARAM_SAVE_DIR, "test_cost.pkl"))
joblib.dump(plot_test_error, os.path.join(PARAM_SAVE_DIR, "test_error.pkl"))

print "[X] Prediction over."
