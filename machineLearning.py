import pandas as pd
import time
import numpy as np
from sklearn import svm, ensemble
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# load the training data
X_train = pd.read_csv("data/X_train.csv").values
Y_train = pd.read_csv("data/Y_train.csv").values

X_test = pd.read_csv("data/X_test.csv").values
Y_test = pd.read_csv("data/Y_test.csv").values

# transform panda df into arrays
X_train = np.delete(X_train, 0, axis=1)
Y_train = np.delete(Y_train, 0, axis=1).flatten()

X_test = np.delete(X_test, 0, axis=1)
Y_test = np.delete(Y_test, 0, axis=1).flatten()

# define the models
svm_linear_kernel_clf = svm.SVC(kernel="linear", gamma=0.001, C=100., random_state=100)
svm_rbf_kernel_clf = svm.SVC(kernel="rbf", gamma=0.001, C=100., random_state=100)
rf_clf = ensemble.RandomForestClassifier(max_leaf_nodes=50, random_state=100)
nn_clf = MLPClassifier(random_state=100)


# setup cross validation
def kfold_cross_val_score(model, X_train, Y_train, n_folds=10):
    k_fold_shuttle = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train, Y_train)
    # rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=k_fold_shuttle))
    return (-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=k_fold_shuttle)).mean()


svm_linear_kernel_score = kfold_cross_val_score(svm_linear_kernel_clf, X_train, Y_train)
print("K fold SVM linear kernel score: {:5f}\n".format(svm_linear_kernel_score.mean()))

svm_rbf_kernel_score = kfold_cross_val_score(svm_rbf_kernel_clf, X_train, Y_train)
print("K fold SVM rbf kernel score: {:5f}\n".format(svm_rbf_kernel_score.mean()))

rf_score = kfold_cross_val_score(rf_clf, X_train, Y_train)
print("K fold Random Forest score: {:5f}\n".format(rf_score.mean()))

nn_score = kfold_cross_val_score(nn_clf, X_train, Y_train)
print("K fold Neural Network score: {:5f}\n\n".format(nn_score.mean()))

# Fit the models
t0 = time.time()
svm_linear_kernel_clf = svm_linear_kernel_clf.fit(X_train, Y_train)
t1 = time.time()
print("SVM linear fitting took at %.2f seconds\n" % (t1 - t0))

t0 = time.time()
svm_rbf_kernel_clf = svm_rbf_kernel_clf.fit(X_train, Y_train)
t1 = time.time()
print("SVM rbf fitting took at %.2f seconds\n" % (t1 - t0))

t0 = time.time()
rf_clf = rf_clf.fit(X_train, Y_train)
t1 = time.time()
print("Random Forest fitting took at %.2f seconds\n" % (t1 - t0))

t0 = time.time()
nn_clf = nn_clf.fit(X_train, Y_train)
t1 = time.time()
print("Neural Network fitting took at %.2f seconds\n" % (t1 - t0))

# make predictions
svm_linear_kernel_pred = svm_linear_kernel_clf.predict(X_test)
svm_rbf_kernel_pred = svm_rbf_kernel_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
nn_pred = nn_clf.predict(X_test)

# measure and output accuracy
svm_linear_kernel_score = accuracy_score(Y_test, svm_linear_kernel_pred)
svm_rbf_kernel_score = accuracy_score(Y_test, svm_rbf_kernel_pred)
rf_score = accuracy_score(Y_test, rf_pred)
nn_score = accuracy_score(Y_test, nn_pred)

print("SVM linear kernel accuracy score: {:5f}\n".format(svm_linear_kernel_score))
print("SVM rbf kernel accuracy score: {:5f}\n".format(svm_rbf_kernel_score))
print("Random Forest accuracy score: {:5f}\n".format(rf_score))
print("Neural Net accuracy score: {:5f}\n".format(nn_score))
