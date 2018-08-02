import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics.classification import confusion_matrix, jaccard_similarity_score

# load the training data
DATA_PATH = "data/"
X = pd.read_csv("%sX_train.csv" % DATA_PATH)
Y_train = pd.read_csv("%sY_train.csv" % DATA_PATH).values
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
sgd_clf = joblib.load('models/sgd_clf.pkl')
svm_clf = joblib.load('models/svm_clf.pkl')
rf_clf = joblib.load('models/rf_clf.pkl')
nn_clf = joblib.load('models/nn_clf.pkl')
print("Models loaded")

# Query the 10 most important gene for random forest classifier
importances = rf_clf.estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_clf.estimator],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(30):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


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
print("SGD Jaccard similarity score: {:5f}".format(sgd_score))
print("SVM Jaccard similarity : {:5f}".format(svm_score))
print("Random Forest Jaccard similarity score: {:5f}".format(rf_score))
print("Neural Net Jaccard similarity score: {:5f}".format(nn_score))


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
