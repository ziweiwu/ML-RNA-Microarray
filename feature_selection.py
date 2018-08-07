import json
import pandas as pd
import numpy as np
import matplotlib

# https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
matplotlib.use('TkAgg')  # to solve the issue of reporting python is not used as framework,
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score
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

# load models
logit = joblib.load('models/logit.pkl')
linear_svm = joblib.load('models/linear_svm.pkl')
none_linear_svm = joblib.load('models/none_linear_svm.pkl')
rf = joblib.load('models/rf.pkl')
nn = joblib.load('models/nn.pkl')
print("Models loaded")


########################################################################################
#                    Feature selection
########################################################################################
# param set for grid search for each model
def model_tune_params(model, params):
    new_model = GridSearchCV(estimator=model,
                             param_grid=params, cv=5,
                             scoring="recall_macro")
    return new_model


logit_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

linear_svm_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

rf_params = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_leaf_nodes': [50, 100, 150, 200],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy']
}

# feature selection
C_params = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 1,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()

n_features_svm = []
recall_svm = []

# perform feature selection using sparse svm
def svm_feature_selection(C_params):
    for C in C_params:
        model_select = svm.LinearSVC(random_state=100, penalty="l1", C=C, dual=False, tol=1e-4)
        model_select = SelectFromModel(model_select).fit(X_train, Y_train)
        train_features = model_select.transform(X_train)
        test_features = model_select.transform(X_test)
        print("\nWith C={}".format(C))
        print("Sparse SVM reduced number of features to {}.".format(test_features.shape[1]))

        model = svm.LinearSVC(random_state=100, dual=False)
        if test_features.shape[1] <= 200: model = model_tune_params(model, linear_svm_params)
        model.fit(train_features, Y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=Y_test, average="macro")
        print("Linear SVC score after FEATURE SELECTION: {:5f}".format(score))
        n_features_svm.append(test_features.shape[1])
        recall_svm.append(score)


svm_feature_selection(C_params)

# perform feature selection using rf, use mean as threshold
thresholds = [0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004,
              0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]

n_features_rf = []
recall_rf = []


def rf_feature_selection(thresholds):
    for threshold in thresholds:
        model = ensemble.RandomForestClassifier(random_state=100, n_estimators=50)
        model_select = SelectFromModel(model, threshold=threshold).fit(X_train, Y_train)
        train_features = model_select.transform(X_train)
        test_features = model_select.transform(X_test)
        print("\nWith threshold {}".format(threshold))
        print("RF reduced number of features to {}.".format(test_features.shape[1]))

        model = ensemble.RandomForestClassifier(random_state=100)
        if test_features.shape[1] <= 200: model = model_tune_params(model, rf_params)
        model.fit(train_features, Y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=Y_test, average="macro")
        print("RF accuracy after FEATURE SELECTION: {:5f}".format(score))
        n_features_rf.append(test_features.shape[1])
        recall_rf.append(score)


rf_feature_selection(thresholds)

# perform feature selection using logistic regression
C_params = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()

n_features_logit = []
recall_logit = []


def logit_feature_selection(C_params):
    for C in C_params:
        model = linear_model.LogisticRegression(random_state=100, penalty="l1", C=C, tol=1e-4)
        model_select = SelectFromModel(model).fit(X_train, Y_train)
        train_features = model_select.transform(X_train)
        test_features = model_select.transform(X_test)
        print("\nWith C={}".format(C))
        print("Logistic regression reduced number of features to {}.".format(test_features.shape[1]))

        model = linear_model.LogisticRegression(random_state=100)
        if test_features.shape[1] <= 200: model = model_tune_params(model, logit_params)
        model.fit(train_features, Y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=Y_test, average="macro")
        print("Logistic regression accuracy after FEATURE SELECTION: {:5f}".format(score))
        n_features_logit.append(test_features.shape[1])
        recall_logit.append(score)


logit_feature_selection(C_params)

########################################################################################
#                    Feature Selection Performance
########################################################################################
print(n_features_svm)
print(recall_svm)
print(n_features_rf)
print(recall_rf)
print(n_features_logit)
print(recall_logit)

figure(num=None, figsize=(6, 8), dpi=800, facecolor='w', edgecolor='k')
plt.xlabel('Number of Features')
plt.ylabel('Recall')
plt.title("Number of Features vs. Recall")
plt.plot(n_features_svm, recall_svm, 'o-')
plt.plot(n_features_rf, recall_rf, '^-', color='green')
plt.plot(n_features_logit, recall_logit, 's-', color='red')
plt.legend(['SVM', 'Random Forest', 'Logistic Regression'], loc=5)
plt.axis([0, 200, 0.5, 1])
plt.savefig('images/feature_selection_performance.png', dpi=600)
