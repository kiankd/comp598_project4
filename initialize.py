# Please run this file as the first thing you do when you clone the repo.
# This is only compatable with linux and OSX.
# Author: Kian Kenyon-Dean

import numpy as np
from python.import_data import import_csv
from os import system,chdir

if __name__ == '__main__':
    system('unzip csv_data/rotated_images.csv.zip')
    
    x,y = import_csv('rotated_images.csv')
    system('mkdir npy_data')
    np.save('npy_data/rotated_features',x)
    np.save('npy_data/y_classifications',y)

    system('rm rotated_images.csv')
    system('mkdir split')
    chdir('python')    
    system('python train_val_test_split.py')
    chdir('..')
