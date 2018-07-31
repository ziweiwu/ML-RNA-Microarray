import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics.classification import jaccard_similarity_score
from matplotlib.pyplot import figure

########################################################################################
#                    Load dataset
########################################################################################
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


########################################################################################
#                     parameter tuning for SVM, RF and logistic regression
########################################################################################
def kfold_model_score(model, X_train, Y_train, numFolds=5):
    k_fold_shuttle = KFold(n_splits=numFolds, random_state=100).get_n_splits(X_train, Y_train)
    return np.mean(cross_val_score(model, X_train, Y_train, cv=k_fold_shuttle))


# test C for svm
# test  C parameters for logistic regression
C_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


def tune_C_svm(C_params, X_train, Y_train, X_test, Y_test):
    for C in C_params:
        model = svm.LinearSVC(random_state=100, C=C, penalty="l1", dual=False, tol=1e-4)
        model = model.fit(X_train, Y_train)
        model_pred = model.predict(X_test)
        model_score = jaccard_similarity_score(Y_test, model_pred)
        print("SVM score with C={:5f}: {:5f}".format(C, model_score.mean()))


tune_C_svm(C_params, X_train, Y_train, X_test, Y_test)

# do a grid search to find optimal parameter for random forest
parameters = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_leaf_nodes': [50, 100, 150, 200],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1)
rf_clf = GridSearchCV(estimator=rf_clf, param_grid=parameters).fit(X_train, Y_train)
sorted(rf_clf.cv_results_.keys())
print("The optimal parameters for random forest classifier are: ")
print(rf_clf.best_params)

# test  C parameters for logistic regression
C_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]


def tune_C_logit(C_params, X_train, Y_train, X_test, Y_test):
    for C in C_params:
        model = linear_model.LogisticRegression(random_state=100, C=C, penalty="l1")
        model = model.fit(X_train, Y_train)
        model_pred = model.predict(X_test)
        model_score = jaccard_similarity_score(Y_test, model_pred)
        print("Logistic Regression score with C={}: {:5f}".format(C, model_score.mean()))


tune_C_logit(C_params, X_train, Y_train, X_test, Y_test)

########################################################################################
#                    Feature selection
########################################################################################
print("\nPerforming feature selection using Sparse SVM and RF \n")

# TODO
# use defealt threshold level
# 3 methods to feature selection, svm, rf, logistic regression
# 200 features, -> 20 -> 10
# draw graph to visualize different method for feature selection Y: accuracy, X: num of features

C_params = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 1,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()

n_features_svm = []
accuracy_svm = []


# perform feature selection using sparse svm
def svm_feature_selection(C_params):
    for C in C_params:
        svm_select = svm.LinearSVC(random_state=100, penalty="l1", C=C, dual=False, tol=1e-4)
        svm_select = SelectFromModel(svm_select).fit(X_train, Y_train)
        svm_features = svm_select.transform(X_train)
        print("\nWith C={}".format(C))
        print("Sparse SVM reduced number of features to {}.".format(svm_features.shape[1]))
        svm_clf_l1 = svm.LinearSVC(random_state=100, penalty="l1", C=1, dual=False, tol=1e-4)
        svm_clf_l1_score = kfold_model_score(svm_clf_l1, svm_features, Y_train)
        print("Linear SVC l1 CV score after FEATURE SELECTION: {:5f}".format(svm_clf_l1_score.mean()))
        n_features_svm.append(svm_features.shape[1])
        accuracy_svm.append(svm_clf_l1_score.mean())


svm_feature_selection(C_params)

# perform feature selection using rf, use mean as threshold
thresholds = [0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004,
              0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]

n_features_rf = []
accuracy_rf = []


def rf_feature_selection(thresholds):
    for threshold in thresholds:
        rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1, n_estimators=50)
        rf_select = SelectFromModel(estimator=rf_clf, threshold=threshold)
        rf_select = rf_select.fit(X_train, Y_train)
        rf_features = rf_select.transform(X_train)
        print("\nWith threshold {}".format(threshold))
        print("RF reduced number of features to {}.".format(rf_features.shape[1]))

        rf_clf = ensemble.RandomForestClassifier(random_state=100, n_jobs=-1, n_estimators=50)
        rf_clf_score = kfold_model_score(rf_clf, rf_features, Y_train)
        print("RF CV score after FEATURE SELECTION: {:5f}".format(rf_clf_score.mean()))
        n_features_rf.append(rf_features.shape[1])
        accuracy_rf.append(rf_clf_score.mean())


rf_feature_selection(thresholds)

# perform feature selection using logistic regression
C_params = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()

n_features_logit = []
accuracy_logit = []


def logit_feature_selection(C_params):
    for C in C_params:
        logit_select = linear_model.LogisticRegression(random_state=100, penalty="l1", C=C, tol=1e-4)
        logit_select = SelectFromModel(logit_select).fit(X_train, Y_train)
        logit_features = logit_select.transform(X_train)
        print("\nWith C={}".format(C))
        print("Logistic regression reduced number of features to {}.".format(logit_features.shape[1]))
        logit_clf = linear_model.LogisticRegression(random_state=100, penalty="l1", tol=1e-4)
        logit_clf_score = kfold_model_score(logit_clf, logit_features, Y_train)
        print("Logistic regression score after FEATURE SELECTION: {:5f}".format(logit_clf_score.mean()))
        n_features_logit.append(logit_features.shape[1])
        accuracy_logit.append(logit_clf_score.mean())


logit_feature_selection(C_params)

########################################################################################
#                    Feature Selection Performance
########################################################################################
print(n_features_svm)
print(accuracy_svm)
print(n_features_rf)
print(accuracy_rf)
print(n_features_logit)
print(accuracy_logit)

figure(num=None, figsize=(8, 6), dpi=500, facecolor='w', edgecolor='k')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title("Number of Features vs. Accuracy")
plt.plot(n_features_svm, accuracy_svm, 'o-')
plt.plot(n_features_rf, accuracy_rf, '^-', color='green')
plt.plot(n_features_logit, accuracy_logit, 's-', color='red')
plt.legend(['SVM', 'Random Forest', 'Logistic Regression'], loc=5)
plt.axis([0, 200, 0.5, 1])
plt.savefig('images/feature_selection_performance.png', dpi=600)
