import os
import scipy.misc
import mahotas
import sys
import numpy as np
sys.path.append('/home/ml/adoomr/Archive/mlproject4/python')
from get_data import get_train_val_test
from normalize import normalize


SAVE_DIRECTORY = '/home/ml/adoomr/Archive/mlproject4/sanctuary/'
RADIUS = 2
NPOINTS = 8

def lbp_transform(X, radius, npts, img_height=41, img_width=41): # X is a matrix of images
	def single_lbp_transform(x): # x is a single image
		return mahotas.features.lbp(x.reshape(img_height, img_width), radius, npts)
	return np.asarray(map(single_lbp_transform, X))
		

trainX, trainY, valX, valY, testX, testY = get_train_val_test()

changed_train_X = lbp_transform(trainX, RADIUS, NPOINTS)
changed_val_X = lbp_transform(valX, RADIUS, NPOINTS)
changed_test_X = lbp_transform(testX, RADIUS, NPOINTS)

save_dir = os.path.join(SAVE_DIRECTORY, 'lbp_dataset_%d_%d' % (RADIUS, NPOINTS))

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'lbp_trainX.npy'), changed_train_X)
np.save(os.path.join(save_dir, 'lbp_valX.npy'), changed_val_X)
np.save(os.path.join(save_dir, 'lbp_testX.npy'), changed_test_X)

np.save(os.path.join(save_dir, 'lbp_trainY.npy'), trainY)
np.save(os.path.join(save_dir, 'lbp_valY.npy'), valY)
np.save(os.path.join(save_dir, 'lbp_testY.npy'), testY)
