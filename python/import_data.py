# Author: Kian Kenyon-Dean
# The purpose of this small file is to import the virus image data.

import numpy as np
import csv
import os
from scipy import misc

ORIGINAL_DIRECTORY = '../resampled-to-8bit'

def import_csv(file_path):
    """ Here we are taking in a csv file and returning it as a numpy array, but we build it with python lists. """
    print 'Loading file: %s'%file_path

    X = []
    y = []
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header row.
        for sample in reader:
            # Get the class.
            y.append(sample[0])

            # Turn inputs into floats and then reshape the data into the 41x41 matrix that we want for the convnet.
            X.append(np.array(sample[2:]).astype(np.float32).reshape(41,41)) # Skip the class and ID columns.

    return np.array(X).astype(int),np.array(y)

def get_pixels(directory, image_file_name):
    s = image_file_name.split('-')
    class_ = s[1]
    sample = s[3]
    
    # Flatten because these are grayscale. Flatten again to not get a 41x41 matrix but a 1681x1 vector.
    return [class_, sample] + list(misc.imread(directory +'/'+ image_file_name, flatten=True).flatten())

def images_to_csv(directory):
    csv_matrix = [['class','class_sample_number']+['']*(41*41)]
    for png_file in os.listdir(directory):
        if png_file.endswith('.png'):
            csv_matrix.append(get_pixels(directory, png_file))

    with open(directory + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in csv_matrix:
            writer.writerow(row)

if __name__ == '__main__':
    images_to_csv(ORIGINAL_DIRECTORY)


