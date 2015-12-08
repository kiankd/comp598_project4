import numpy as np

def get_data():
    return np.load('../npy_data/rotated_data_features.npy'),np.load('../npy_data/y_classifications.npy')

def get_train_val_test():
    split_dir = '../split/'
    x,y = get_data()
    x = x.reshape(18000,41*41)
    y = y.astype(int)


    train_indexes = np.load(split_dir+'train_indexes.npy')
    val_indexes = np.load(split_dir+'val_indexes.npy')
    test_indexes = np.load(split_dir+'test_indexes.npy')
    
    return x[train_indexes],y[train_indexes], x[val_indexes],y[val_indexes], x[test_indexes],y[test_indexes]

