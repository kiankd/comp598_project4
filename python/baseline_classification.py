# Author: Kian Kenyon-Dean
# Purpose: test baseline classifiers on the dataset.

import numpy as np
from normalize import normalize
from get_data import get_train_val_test
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def test_classifier(model, X, y, testX, testy):
    print 'Fitting model %s...'%(type(model).__name__)
    model.fit(X,y)
    yhat = model.predict(testX)
    return f1_score(testy, yhat), confusion_matrix(testy, yhat), classification_report(testy, yhat)

if __name__ == '__main__':
    data_directory = '../sanctuary/lbp_dataset_2_8/'
    
    prefix = 'new_normalize'
    if prefix != 'lbp':
        trainX,trainy, valX,valy, testX,testy = get_train_val_test()
    else:
        trainX = np.load(data_directory + 'lbp_trainX.npy')
        trainy = np.load(data_directory + 'lbp_trainY.npy')
        testX = np.load(data_directory + 'lbp_testX.npy')
        testy = np.load(data_directory + 'lbp_testY.npy')
    
    trainX = normalize(trainX)
    #valX = normalize(valX)
    testX = normalize(testX)

    names = []
    scores = []
    matrices = []
    reports = []

    for model_class in [LinearSVC, GaussianNB, SVC, LogisticRegression]:
        model = model_class()
        print model
        names.append(type(model).__name__)
        f1, conf_mat, report = test_classifier(model, trainX, trainy, testX, testy)
        print 'Score: ',f1
        scores.append(f1)
        matrices.append(conf_mat)
        reports.append(report)

    with open('../'+prefix+'_baseline_results.txt','w') as f:
        for i in range(len(names)):
            f.write('%s - f1_score: %0.3f\n'%(names[i], scores[i]))
            np.save('../'+prefix+'_%s_confusion_matrix'%names[i], matrices[i])
            f.write('Classification Report:\n%s\n'%reports[i])

