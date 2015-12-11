# Author: Alan Do-Omri

from sys import argv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

file_names = os.listdir('.')
file_names = [x for x in file_names if x.endswith('.npy')]

for s in file_names: 
    idx = s.find('confusion')
    PREFIX = s[:idx]    

    strings = PREFIX.split('_')
    string = ''
    for sub in strings:
        string += sub

    cm = np.load(PREFIX+'confusion_matrix.npy')
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
        
    plt.savefig(PREFIX+'_%s_cm.pdf' % string)

