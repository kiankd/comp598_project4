import sys
sys.path.insert(0,'/home/2013/adoomr/.local/lib/python2.7/site-packages')
from sklearn.cross_validation import train_test_split
import glob
import lasagne
from lasagne.regularization import regularize_layer_params, l2
import numpy as np
import os
import theano
import theano.tensor as T


NUM_EPOCHS = 8000
LEARNING_RATE = 0.005
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
N_CLASSES = 10
L2_REG = 0.0001
BATCH_SIZE = 100
VALIDATION_FREQUENCY = 100
PARAM_SAVE_DIR = './params'
DATASET_DIR = '/home/ml/adoomr/Archive/mlproject4/sanctuary/lbp_dataset_2_8'

print "[X] Emerald is loading its data."
trainX = np.load(os.path.join(DATASET_DIR, 'lbp_trainX.npy'))
trainY = np.load(os.path.join(DATASET_DIR, 'lbp_trainY.npy'))
valX = np.load(os.path.join(DATASET_DIR, 'lbp_valX.npy'))
valY = np.load(os.path.join(DATASET_DIR, 'lbp_valY.npy'))
testX = np.load(os.path.join(DATASET_DIR, 'lbp_testX.npy'))
testY = np.load(os.path.join(DATASET_DIR, 'lbp_testY.npy'))


print "[X] Emerald building its layers."
layer0 = lasagne.layers.InputLayer(shape=(None, 1, IMAGE_WIDTH, IMAGE_HEIGHT))
layer1 = lasagne.layers.Conv2DLayer(layer0, num_filters=32, filter_size=(7,7))
layer2 = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(2,2))
layer3 = lasagne.layers.Conv2DLayer(layer2, num_filters=48, filter_size=(5,5))
layer4 = lasagne.layers.MaxPool2DLayer(layer3, pool_size=(2,2))
layer5 = lasagne.layers.DenseLayer(layer4, num_units=256)
layer6 = lasagne.layers.DenseLayer(layer5, num_units=N_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
model = layer6

x = T.tensor4()
y = T.ivector()

print "[X] Emerald defining its goals."
model_params = lasagne.layers.get_all_params(model, trainable=True)
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

noisy_output = lasagne.layers.get_output(model, x, deterministic=False)
true_output = lasagne.layers.get_output(model, x, deterministic=True)

l2_loss = regularize_layer_params(model, l2)*L2_REG

noisy_cost = T.mean(T.nnet.categorical_crossentropy(noisy_output, y)) + l2_loss
true_cost = T.mean(T.nnet.categorical_crossentropy(true_output, y)) + l2_loss

noisy_error = T.mean(T.neq(noisy_output, y))
true_error = T.mean(T.neq(true_output, y))

updates = lasagne.updates.sgd(noisy_cost, model_params, learning_rate=sh_lr)

train = theano.function([x,y], [train_cost, train_output], updates=updates, 
	allow_input_downcast=True)

get_score = theano.function([x,y], [true_cost, true_error], 
	allow_input_downcast=True)

validate_model = theano.function([x,y], [valid_zero_one_loss, valid_cost, valid_output], 
	allow_input_downcast=True)

best_validation_cost = numpy.inf
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

print "[X] Emerald begins its training."
try: 
	while True:
		epoch = epoch + 1

		for minibatch_index in xrange(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index	
			if iter % 100 == 0:
				print "[O] Training at iteration %d." % iter
			cost_ij = train(trainX[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainX.shape[0])], trainY[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainY.shape[0])])
			if (iter+1) % VALIDATION_FREQUENCY == 0:
				train_cost, train_error = get_score(trainX, trainY)
				valid_cost, valid_error = get_score(validX, validY)
				test_cost, test_error = get_score(testX, testY)

				plot_train_cost.append(train_cost)
				plot_train_error.append(train_error)

				plot_valid_cost.append(valid_cost)
				plot_valid_error.append(valid_error)

				plot_test_cost.append(test_cost)
				plot_test_error.append(test_error)

				plot_iters.append(iter)


				print "--> Epoch %i, minibatch %i/%i has training true cost \t %f." % (epoch, minibatch_index+1, n_train_batches, train_cost)
				print "--> Epoch %i, minibatch %i/%i has validation true cost \t %f and error of \t %f %%." % (epoch, minibatch_index+, n_train_batches, valid_cost, valid_error)

				if valid_cost < best_validation_cost:
					print "----> New best score found!"
					print "--> Test cost of %f and test error of %f." % (test_cost, test_error)
					if not os.path.exists(PARAM_SAVE_DIR):
						os.makedirs(PARAM_SAVE_DIR)
					for f in glob.glob(PARAM_SAVE_DIR+'/*'):
						os.remove(f)
					all_param_values = lasagne.layers.get_all_param_values(model)
					joblib.dump(all_param_values, os.path.join(PARAM_SAVE_DIR, 'params.pkl'))
					print "----> Parameters saved."
					best_validation_cost = valid_cost
					best_iter = iter
except KeyboardInterrupt:
	pass

end_time = timeit.default_timer()

print "--> Best validation score of %f." % best_validation_cost
print "--> Total runtime %.2f minutes." % ((end_time-start_time) / 60.)
print "[X] Saving the scores."

joblib.dump(plot_iters, os.path.join(PARAM_SAVE_DIR, "iters.pkl"))

joblib.dump(plot_train_cost, os.path.join(PARAM_SAVE_DIR, "train_cost.pkl")
joblib.dump(plot_train_error, os.path.join(PARAM_SAVE_DIR, "train_error.pkl")

joblib.dump(plot_valid_cost, os.path.join(PARAM_SAVE_DIR, "valid_cost.pkl")
joblib.dump(plot_valid_error, os.path.join(PARAM_SAVE_DIR, "valid_error.pkl")

joblib.dump(plot_test_cost, os.path.join(PARAM_SAVE_DIR, "test_cost.pkl")
joblib.dump(plot_test_error, os.path.join(PARAM_SAVE_DIR, "test_error.pkl")

print "[X] Prediction over."
