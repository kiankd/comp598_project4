# Virus Texture Classification
Project 4 - COMP 598

Relevant paper: http://www.cb.uu.se/~gustaf/virustexture/

Dependencies:
Numpy
Theano
Lasagne
Scipy
Scikit-learn
Mahotas


The dataset can be found in the folder "resampled-to-8bit", found on https://github.com/kiankd/comp598_project4/. 

To turn the dataset into raw pixel values, change directory to "python" and then run "new_images.py". This will create a large (~200mb) csv file with the pixel values for 18000 rotated images of the original dataset. Zip this file. 

After this, run "initialize.py" in the main project directory. This will create ".npy" files with the data files that allow for quick importing. This will create large files.

You are now set to run any of the classifiers (see "python" folder) and neural networks (see the "santuary" folder). 
