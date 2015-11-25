# Author: Kian Kenyon-Dean
# The purpose of this program is to make 12 new images for each image
# after a random rotation of d degrees for each one.

from import_data import import_csv,ORIGINAL_DIRECTORY,images_to_csv
from scipy import misc
import numpy as np

if __name__ == '__main__':
    destination = '../rotated_images/'
    degree_rotations = []
    x,y = import_csv(ORIGINAL_DIRECTORY+'.csv')

    np.random.seed(1917)
    for i in range(len(x)):
        image_matrix = x[i]
        image_class = y[i]
     
        randomized_degrees = np.random.uniform(0,360,12) 
        degree_rotations.append(randomized_degrees)  
        for d in randomized_degrees:
            new_image = misc.imrotate(image_matrix, d)

            name = 'class-%d-sample-%d-degree-%d-%s.png'\
                    %(int(image_class), (i+1)%101, int(d), str(d).split('.')[1][:5])

            misc.imsave(destination + name, new_image)

    with open('../random_degree_rotations.csv','w') as f:
        f.write(','*12)
        for arr in degree_rotations:
            for i in range(len(arr)):
                if i == 0:
                    f.write('%0.5f'%arr[i])
                    continue
                f.write(',%0.5f'%arr[i])
            f.write('\n')

    images_to_csv(destination)
    
