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
DATA_PATH = "data/"
X_test = pd.read_csv("%sX_test.csv" % DATA_PATH).values
Y_test = pd.read_csv("%sY_test.csv" % DATA_PATH).values
# transform panda df into arrays

X_test = np.delete(X_test, 0, axis=1)
Y_test = np.delete(Y_test, 0, axis=1).flatten()

f = open("%sclass_names.txt" % DATA_PATH)
class_names = json.load(f)
f.close()

print("Dataset loaded.")

# load models
sgd_clf = joblib.load('sgd_clf.pkl')
svm_clf = joblib.load('svm_clf.pkl')
rf_clf = joblib.load('rf_clf.pkl')
nn_clf = joblib.load('nn_clf.pkl')
print("Models loaded")

# make predictions
sgd_pred = sgd_clf.predict(X_test)
svm_pred = svm_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
nn_pred = nn_clf.predict(X_test)

# measure and output accuracy
print("Scores after parameter tuning: ")
sgd_score = jaccard_similarity_score(Y_test, sgd_pred)
svm_score = jaccard_similarity_score(Y_test, svm_pred)
rf_score = jaccard_similarity_score(Y_test, rf_pred)
nn_score = jaccard_similarity_score(Y_test, nn_pred)
print("SGD Jaccard similarity score: {:5f}\n".format(sgd_score))
print("SVM Jaccard similarity : {:5f}\n".format(svm_score))
print("Random Forest Jaccard similarity score: {:5f}\n".format(rf_score))
print("Neural Net Jaccard similarity score: {:5f}\n".format(nn_score))


# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print("Confusion matrix plots")
# SGD
sgd_matrix = confusion_matrix(Y_test, sgd_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(sgd_matrix, classes=class_names, title='SGD')
plt.savefig('images/sgd_confusion_matrix.png')

# SVM
svm_matrix = confusion_matrix(Y_test, svm_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(svm_matrix, classes=class_names, title='SVM')
plt.savefig('images/svm_confusion_matrix.png')

# RF
rf_matrix = confusion_matrix(Y_test, rf_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(rf_matrix, classes=class_names, title='Random Forest')
plt.savefig('images/rf_confusion_matrix.png')

# NN
nn_matrix = confusion_matrix(Y_test, nn_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(nn_matrix, classes=class_names, title='Neural Network')
plt.savefig('images/nn_confusion_matrix.png')
