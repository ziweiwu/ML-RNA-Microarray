import pandas as pd
import numpy as np
import time
from sklearn.metrics.classification import accuracy_score, classification_report, jaccard_similarity_score, f1_score
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier

# load the training data
print("Loading dataset...")
X_train = pd.read_csv("data/X_train.csv").values
Y_train = pd.read_csv("data/Y_train.csv").values
X_test = pd.read_csv("data/X_test.csv").values
Y_test = pd.read_csv("data/Y_test.csv").values
# transform panda df into arrays
X_train = np.delete(X_train, 0, axis=1)
Y_train = np.delete(Y_train, 0, axis=1).flatten()
X_test = np.delete(X_test, 0, axis=1)
Y_test = np.delete(Y_test, 0, axis=1).flatten()

print("Dataset loaded.")

# define the models
sgd_clf = linear_model.SGDClassifier(random_state=100, n_jobs=-1, max_iter=5)
svm_clf = svm.SVC(random_state=100)
rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1)
nn_clf = MLPClassifier(random_state=100)

# paremeter tuning for sgd, by default sgd fits a linear svm
print("Parameter tuning starts...")
parameters = {
    'alpha': [0.0001, 0.5, 1, 5, 50, 100, 200, 500],
    'penalty': ('l2', 'l1', 'elasticnet'),
}
sgd_clf = GridSearchCV(estimator=sgd_clf, param_grid=parameters).fit(X_train, Y_train)
print("Best params for sgd:", sgd_clf.best_params_, '\n')

# parameter tuning for non-linear svm kernel
parameters = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.001, 0.0001, 0.00001],
    'kernel': ('poly', 'rbf', 'sigmoid')
}
svm_clf = GridSearchCV(estimator=svm_clf, param_grid=parameters).fit(X_train, Y_train)
print("Best params for svm:", svm_clf.best_params_, '\n')

# parameter tuning for random forest
parameters = {
    'n_estimators': [10, 20, 50],
    'max_leaf_nodes': [50, 100, 150, 200]
}
rf_clf = GridSearchCV(estimator=rf_clf, param_grid=parameters).fit(X_train, Y_train)
print("Best params for rf:", rf_clf.best_params_, '\n')

# parameter tuning for neural network
parameters = {
    'hidden_layer_sizes': [50, 100, 150, 200],
    'alpha': [0.0001, 0.0005, 0.001, 0.005],
    'activation': ('relu', 'tanh', 'identity'),
}
nn_clf = GridSearchCV(estimator=nn_clf, param_grid=parameters).fit(X_train, Y_train)
print("Best params for nn:", nn_clf.best_params_, '\n')

print("Parameter tuning finished.")

print("Cross validation starts...")


# cross validation to select the best model
def kfold_model_score(model, X_train, Y_train, numFolds=5):
    k_fold_shuttle = KFold(n_splits=numFolds, random_state=100).get_n_splits(X_train, Y_train)
    return np.mean(cross_val_score(model, X_train, Y_train, cv=k_fold_shuttle))


sgd_clf_score = kfold_model_score(sgd_clf, X_train, Y_train)
print("Linear svm score: {:5f}\n".format(sgd_clf_score.mean()))

svm_score = kfold_model_score(svm_clf, X_train, Y_train)
print("Non-linear svm score: {:5f}\n".format(svm_score.mean()))

rf_score = kfold_model_score(rf_clf, X_train, Y_train)
print("Random Forest score: {:5f}\n".format(rf_score.mean()))

nn_score = kfold_model_score(nn_clf, X_train, Y_train)
print("MPL Neural Network score: {:5f}\n\n".format(nn_score.mean()))
print("Cross validation finished.")

# Fit the models
print("Model fitting begins...")
t0 = time.time()
sgd_clf = sgd_clf.fit(X_train, Y_train)
t1 = time.time()
print("sgd_clf training took %.2f seconds\n" % (t1 - t0))

t0 = time.time()
svm_clf = svm.fit(X_train, Y_train)
t1 = time.time()
print("SVM training took %.2f seconds\n" % (t1 - t0))

t0 = time.time()
rf_clf = rf_clf.fit(X_train, Y_train)
t1 = time.time()
print("Random Forest training took %.2f seconds\n" % (t1 - t0))

t0 = time.time()
nn_clf = nn_clf.fit(X_train, Y_train)
t1 = time.time()
print("Neural Network fitting took at %.2f seconds\n" % (t1 - t0))

print("Model fitting finished.")

print("Making predictions.")
# make predictions
sgd_pred = sgd_clf.predict(X_test)
svm_pred = svm_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
nn_pred = nn_clf.predict(X_test)

# measure and output accuracy
print("Jaccard similarity scores: ")
sgd_score = jaccard_similarity_score(Y_test, sgd_pred)
svm_score = jaccard_similarity_score(Y_test, svm_pred)
rf_score = jaccard_similarity_score(Y_test, rf_pred)
nn_score = jaccard_similarity_score(Y_test, nn_pred)
print("SGD Jaccard similarity score: {:5f}\n".format(sgd_score))
print("SVM Jaccard similarity : {:5f}\n".format(svm_pred))
print("Random Forest Jaccard similarity score: {:5f}\n".format(rf_score))
print("Neural Net Jaccard similarity score: {:5f}\n".format(nn_score))

# measure and output f1 score
print("F1 scores: ")
sgd_score = f1_score(Y_test, sgd_pred)
svm_score = f1_score(Y_test, svm_pred)
rf_score = f1_score(Y_test, rf_pred)
nn_score = f1_score(Y_test, nn_pred)
print("SGD accuracy f1 score: {:5f}\n".format(sgd_score))
print("SVM accuracy f1 score: {:5f}\n".format(svm_pred))
print("Random Forest f1 score: {:5f}\n".format(rf_score))
print("Neural Net f1 score: {:5f}\n".format(nn_score))
