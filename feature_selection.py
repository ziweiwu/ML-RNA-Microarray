import time
import json
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics.classification import confusion_matrix, jaccard_similarity_score
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

# load the training data
print("Loading data sets...")
DATA_PATH = "data/"
X_train = pd.read_csv("%sX_train.csv" % DATA_PATH).values
Y_train = pd.read_csv("%sY_train.csv" % DATA_PATH).values
X_test = pd.read_csv("%sX_test.csv" % DATA_PATH).values
Y_test = pd.read_csv("%sY_test.csv" % DATA_PATH).values

# transform panda df into arrays
X_train = np.delete(X_train, 0, axis=1)
Y_train = np.delete(Y_train, 0, axis=1).flatten()
X_test = np.delete(X_test, 0, axis=1)
Y_test = np.delete(Y_test, 0, axis=1).flatten()

f = open("%sclass_names.txt" % DATA_PATH)
class_names = json.load(f)
f.close()

print("Dataset loaded.")

# define the models
sgd_clf_l1 = linear_model.SGDClassifier(random_state=100, penalty="l1", n_jobs=-1, max_iter=5, )
sgd_clf_l2 = linear_model.SGDClassifier(random_state=100, penalty="l1", n_jobs=-1, max_iter=5, )
rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1)

# test the models before parameter tuning
sgd_clf_l1 = sgd_clf_l1.fit(X_train, Y_train)
rf_clf = rf_clf.fit(X_train, Y_train)


def kfold_model_score(model, X_train, Y_train, numFolds=5):
    k_fold_shuttle = KFold(n_splits=numFolds, random_state=100).get_n_splits(X_train, Y_train)
    return np.mean(cross_val_score(model, X_train, Y_train, cv=k_fold_shuttle))


# sgd l1 is a linear svm model can bring sparsity in the feature spaces
sgd_clf_l1_score = kfold_model_score(sgd_clf_l1, X_train, Y_train)
print("Linear svm l1 score: {:5f}\n".format(sgd_clf_l1_score.mean()))

rf_score = kfold_model_score(rf_clf, X_train, Y_train)
print("Random Forest score: {:5f}\n".format(rf_score.mean()))

# test alpha value in increasing order, higher alpha implies higher weight of regalarization
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 50, 100, 200, 500]
for a in alpha:
    print(a)


def test_alpha(X_train, Y_train, X_test, alpha):
    for value in alpha:
        model = linear_model.SGDClassifier(random_state=100, penalty="l1", alpha=value, n_jobs=-1, max_iter=5, )
        model = model.fit(X_train, Y_train)
        model_pred = model.predict(X_test)
        model_score = jaccard_similarity_score(Y_test, model_pred)
        print("SGD score with alpha {:5f}: {:5f}".format(value, model_score.mean()))


test_alpha(X_train, Y_train, X_test, alpha)

# test the effect of number of tree on accuracy for random forest
rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1)
trees = [5, 10, 15, 20, 25, 50, 75, 100]


def test_trees(X_train, Y_train, X_test, trees):
    for tree in trees:
        model = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1, n_estimators=tree)
        model = model.fit(X_train, Y_train)
        model_pred = model.predict(X_test)
        model_score = jaccard_similarity_score(Y_test, model_pred)
        print("Random Forest score with {} trees: {:5f}".format(tree, model_score.mean()))


test_trees(X_train, Y_train, X_test, trees)

# test the effect of tree depth for random forest on accuracy
depth = [10, 25, 50, 75, 100, 125, 150, 175]


def test_depths(X_train, Y_train, X_test, trees):
    for d in depth:
        model = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1, n_estimators=25, max_depth=d)
        model = model.fit(X_train, Y_train)
        model_pred = model.predict(X_test)
        model_score = jaccard_similarity_score(Y_test, model_pred)
        print("Random Forest score with {} depth: {:5f}".format(d, model_score.mean()))


test_depths(X_train, Y_train, X_test, trees)
