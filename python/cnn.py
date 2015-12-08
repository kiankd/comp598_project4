import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lasagne.regularization import regularize_layer_params, l2
import lasagne.objectives
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

from get_data import get_train_val_test as pull_data

DIM = 41
# Hyperparameters
NUM_EPOCHS = 8000
LEARNING_RATE = 0.005
N_CLASSES = 15
L2_REG = 0.0001
BATCH_SIZE = 16
VALIDATION_FREQUENCY = 50
PARAM_SAVE_DIR = './params'
FIGURE_SAVE_DIR = './figures'
DATASET_DIR = '../sanctuary/lbp_dataset_2_8'
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


def build_cnn(input_var):
    # Input layer, as usual:
    layer0 = lasagne.layers.InputLayer(shape=(None, 1, DIM, DIM),
                                        )#input_var=input_var)
    # We do not apply input dropout as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 16 kernels of size 8x8. Strided and padded
    # convolutions are supported as well; see the docstring.
    layer1 = lasagne.layers.Conv2DLayer(
            layer0, num_filters=16, filter_size=(8, 8),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    layer2 = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(2, 2))

    # Another convolution with 48 5x5 kernels, and another 2x2 pooling:
    layer3 = lasagne.layers.Conv2DLayer(
            layer2, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    layer4 = lasagne.layers.MaxPool2DLayer(layer3, pool_size=(2,2))

    ## Convolution with 60 2x2 kernels and another 2x2 pooling
    layer5 = lasagne.layers.Conv2DLayer(layer4, num_filters=60, filter_size=(2,2))

    layer6 = lasagne.layers.MaxPool2DLayer(layer5, pool_size=(2,2))


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    layer7 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(layer6, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # And, finally, the 15-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(layer7, p=.5),
            num_units=N_CLASSES,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform())

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test= pull_data()
    trainX, trainY, valX, valY, testX, testY = pull_data()
    trainX = trainX.reshape(trainX.shape[0],1, DIM, DIM)
    trainY = trainY-1
    valY = valY -1
    testY = testY - 1

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    output_var = T.ivector('targets')

    model = build_cnn(input_var)
    print "[X] CNN defining its goals."
    
    model_params = lasagne.layers.get_all_params(model, trainable=True)
    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

    #why do we want to compute output expressions for model and input_var???
    noisy_output = lasagne.layers.get_output(model, input_var, deterministic=False)
    true_output = lasagne.layers.get_output(model, input_var, deterministic=True)

    noisy_prediction = T.argmax(noisy_output, 1)
    true_prediction = T.argmax(true_output, 1)

    l2_loss = regularize_layer_params(model, l2)*L2_REG

    noisy_cost = T.mean(T.nnet.categorical_crossentropy(noisy_output, output_var)) + l2_loss
    true_cost = T.mean(T.nnet.categorical_crossentropy(true_output, output_var)) + l2_loss

    
   # noisy_error = T.mean(T.neq(noisy_output, output_var))
   # true_error = T.mean(T.neq(true_output, output_var))
    noisy_error = 1-T.mean(lasagne.objectives.categorical_accuracy(noisy_output, output_var))
    true_error = 1-T.mean(lasagne.objectives.categorical_accuracy(true_output, output_var))



    ## stochastic gradient descent updates
    #updates = lasagne.updates.sgd(noisy_cost, model_params, learning_rate=sh_lr)
    ##stochastic gradient descent with Nesterov momentum

    updates = lasagne.updates.nesterov_momentum(
            noisy_cost, model_params, learning_rate=sh_lr, momentum=0.9)

    train = theano.function([input_var,output_var], [noisy_cost, noisy_error], 
        updates=updates, 
        allow_input_downcast=True)

    get_score = theano.function([input_var,output_var], [true_cost, true_error], 
        updates=updates,
        allow_input_downcast=True)


    best_validation_cost = np.inf
    best_iter = 0
    

    n_train_batches = int(np.ceil(trainX.shape[0] / float(BATCH_SIZE)))

    plot_iters = []

    plot_train_cost = []
    plot_train_error = []

    plot_valid_cost = []
    plot_valid_error = []

    plot_test_cost = []
    plot_test_error = []

    epoch = 0
    print "[X] CNN begins its training."
    try: 
        while True:
            epoch = epoch + 1

            for minibatch_index in xrange(n_train_batches):
                iter = (epoch - 1) * n_train_batches + minibatch_index  
                if iter % 100 == 0:
                    print "[O] Training at iteration %d." % iter

                cost_ij = train(trainX[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainX.shape[0])], 
                    trainY[minibatch_index*BATCH_SIZE:np.minimum((minibatch_index+1)*BATCH_SIZE, trainY.shape[0])])

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
                    print "--> Epoch %i, minibatch %i/%i has validation true cost \t %f and error of \t %f %%." % (epoch, minibatch_index+1, n_train_batches, valid_cost, valid_error)

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

    joblib.dump(plot_train_cost, os.path.join(PARAM_SAVE_DIR, "train_cost.pkl"))
    joblib.dump(plot_train_error, os.path.join(PARAM_SAVE_DIR, "train_error.pkl"))

    joblib.dump(plot_valid_cost, os.path.join(PARAM_SAVE_DIR, "valid_cost.pkl"))
    joblib.dump(plot_valid_error, os.path.join(PARAM_SAVE_DIR, "valid_error.pkl"))

    joblib.dump(plot_test_cost, os.path.join(PARAM_SAVE_DIR, "test_cost.pkl"))
    joblib.dump(plot_test_error, os.path.join(PARAM_SAVE_DIR, "test_error.pkl"))

      


if __name__ == '__main__':
    main()

