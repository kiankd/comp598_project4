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

    dataset = []
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header row.
        for sample in reader:
            row_values = []
            for value in sample[1:]: # Skip the first index - the ID column.
                row_values.append(float(value))
            if len(row_values) == 1:
                dataset.append(row_values[0])
            else:
                dataset.append(np.array(row_values))

    return np.array(dataset)

def get_png_pixel_values(directory, image_file_name):
    # Flatten because these are grayscale. Flatten again to not get a 41x41 matrix but a 1681x1 vector.
    s = image_file_name.split('-')
    class_ = s[1]
    sample = s[3]
    return [class_, sample] + list(misc.imread(directory +'/'+ image_file_name, flatten=True).flatten())

def images_to_csv(directory):
    csv_matrix = [['class','class_sample_number']+['']*(41*41)]
    for png_file in os.listdir(directory):
        if png_file.endswith('.png'):
            csv_matrix.append(get_png_pixel_values(directory, png_file))

    with open(directory + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in csv_matrix:
            writer.writerow(row)

if __name__ == '__main__':
    images_to_csv(ORIGINAL_DIRECTORY)


