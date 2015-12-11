# Author: Kian Kenyon-Dean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

"""
lbp_baseline_results.txt                                 lbp_minsplit2_minsamples2_nfeatures_baseline_results.txt  lbp_minsplit4_minsamples4_baseline_results.txt
lbp_minsplit2_minsamples2_baseline_results.txt           lbp_minsplit4_baseline_results.txt
lbp_minsplit2_minsamples2_infogain_baseline_results.txt  lbp_minsplit4_minsamples2_maxdepth5_baseline_results.txt

RandomForestClassifier100 - f1_score: 0.413
"""

prefix = 'lbp_'

param_names = {'minsplit2':'min_samples_split=2',
               'minsplit4':'min_samples_split=4',
               'minsamples2':'min_samples_leaf=2',
               'minsamples4':'min_samples_leaf=4',
               'nfeatures':'nfeatures',
               'infogain':'criterion=entropy',
               'maxdepth5':'max_depth=5',
               '':''
              }

files = [x for x in os.listdir('.') if x.endswith('.txt')] 
params = [x[len(prefix) : x.find('_baseline')] for x in os.listdir('.') if x.endswith('.txt')]

def list_str(l):
    s = ''
    for v in l:
        s += str(v)+', '
    return s[:-2]

RFC = 'RandomForestClassifier'
for i in range(len(files)):
    fname = files[i]
    with open(fname,'r') as f:
        result_strs = [l for l in f.readlines() if 'RandomForestClassifier' in l]
        x,y = [],[]
        
        for s in result_strs:
            num_estimators = int(s[len(RFC):s.find(' ')])
            acc = float(s.split(': ')[1])
            x.append(num_estimators)
            y.append(acc)
        
        x = np.array(x)
        y = np.array(y)
        
        parameters_file_struct = params[i].split('_')
        parameters = [param_names[k] for k in parameters_file_struct]

        print parameters
        print parameters_file_struct
        

        plt.figure()
        plt.plot(x,y,'go')
        plt.suptitle('Random Forest Classification Results')
        plt.title('Params: %s'%list_str(parameters))
        plt.axis([-2, 102, 0.0, 0.5])
        plt.xticks([j*10 for j in range(11)])
        plt.xlabel('Number of Estimators')
        plt.ylabel('F1 Accuracy')
        plt.savefig(params[i] + '.pdf')
        

