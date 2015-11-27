import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

for string in ['GaussianNB', 'LinearSVC', 'LogisticRegression', 'SVC']:
	cm = np.load('%s_confusion_matrix.npy' % string)
	cm = cm.astype('float')/cm.sum(axis=1)

	plt.matshow(cm, cmap=plt.cm.Greens)
	plt.title('Confusion Matrix for %s' % string)
	plt.colorbar()
	plt.tick_params(bottom=True)
	plt.savefig('%s_cm.pdf' % string)
