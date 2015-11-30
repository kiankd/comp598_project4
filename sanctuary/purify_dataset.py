import os
import scipy.misc
import mahotas

raw_dataset_directory = '/home/ml/adoomr/Archive/mlproject4/raw_dataset' 
filelist = os.listdir(raw_dataset_directory)

shapes = [] 

for img_name in filelist:
	img = scipy.misc.imread(os.path.join(raw_dataset_directory, img_name))
	shapes += mahotas.features.lbp(img, 2, 9).shape
	# y = int(img_name.split('-')[1])

print shapes
