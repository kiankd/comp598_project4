# Authors:
#   Kian Kenyon-Dean
#   Alan Do-Omri

import numpy as np
import theano
import theano.tensor as T
import lasagne
from import_data import get_train_val_test as get_data


PARAM_SAVE_DIR = './params'


NUM_EPOCHS = 8000
IMAGE_WIDTH = 41
IMAGE_HEIGHT = 41

if __name__ == '__main__':

    
    xtrain,ytrain, xval,yval, xtest,test = get_data()
    
    


