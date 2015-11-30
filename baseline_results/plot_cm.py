# Author: Alan Do-Omri

import numpy as np
import matplotlib.pyplot as plt

for string in ['GaussianNB', 'LinearSVC', 'LogisticRegression', 'SVC']:
	cm = np.load('lbp_%s_confusion_matrix.npy' % string)
	cm = cm.astype('float')/cm.sum(axis=1)

	plt.matshow(cm, cmap=plt.cm.Greens)
	plt.title('Confusion Matrix for %s' % string)
	plt.colorbar()

	ax = plt.gca()
	ax.xaxis.set_ticks_position('bottom')

	for x in range(15):
		for y in range(15):
			if cm[x,y]>=0.15:
				ax.annotate('%4.2f' % cm[x,y], xy=(y,x), horizontalalignment='center', verticalalignment='center', size=6.3)

	plt.savefig('lbp_%s_cm.pdf' % string)

