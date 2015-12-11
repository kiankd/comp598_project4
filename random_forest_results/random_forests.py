# Author: Kian Kenyon-Dean

import matplotlib.pyplot as plt
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
               'maxdepth5':'max_depth=5'
              }

files = [x for x in os.listdir('.') if x.endswith('.txt')] 
params = [x[len(prefix) : x.find('_baseline')] for x in os.listdir('.') if x.endswith('.txt')]

RFC = 'RandomForestClassifier'
for fname in files:
    with open(fname,'r') as f:
        result_strs = [l for l in f.readlines() if 'RandomForestClassifier' in l]
        results = []
        for s in result_strs:
            num_estimators = int(s[len(RFC):s.find(' ')])
            acc = float(s.split(': ')[1])



