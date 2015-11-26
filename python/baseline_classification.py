# Author: Kian Kenyon-Dean
# Purpose: test baseline classifiers on the dataset.

from get_data import get_train_val_test
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

def test_classifier(model, X, y, testX, testy):
    print 'Fitting model %s...'%(type(model).__name__)
    model.fit(X,y)
    yhat = model.predict(testX)
    return f1_score(testy, yhat), confusion_matrix(testy, yhat)

if __name__ == '__main__':
    trainX,trainy, valX,valy, testX,testy = get_train_val_test()
    names = []
    scores = []
    matrices = []

    for model_class in [SVC, LinearSVC, GaussianNB, LogisticRegression]:
        model = model_class()
        print model
        names.append(type(model).__name__)
        f1, conf_mat = test_classifier(model, trainX, trainy, testX, testy)
        print 'Score: ',f1
        scores.append(f1)
        matrices.append(conf_mat)

    with open('../baseline_results.txt','w') as f:
        for i in range(len(names)):
            f.write('%s - f1_score: %0.3f\n'%(names[i], scores[i]))
            np.save('../%s_confusion_matrix'%names[i], matrices[i])

