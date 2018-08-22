import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load the dataset
DATA_PATH = "data/"
X = pd.read_csv("%sRNA_data/data.csv" % DATA_PATH)
Y = pd.read_csv("%sRNA_data/labels.csv" % DATA_PATH)

# basic overview of data dimension
print(X.head())
print(X.dtypes)
print(Y.head())
print(Y.dtypes)

# count the frequency of each class
class_count = Y['Class'].value_counts().to_dict()
print(class_count)

# convert dataframe into a numpy array
X = X.dropna()
# drop the first column which only contains strings
X = X.drop(X.columns[X.columns.str.contains('unnamed', case=False)], axis=1)
print(X.shape)
print(Y.shape)

# label encode the multiple class string into integer values
Y = Y.drop(Y.columns[0], axis=1)
le = LabelEncoder()
le.fit(Y)
class_names = list(le.classes_)
Y = Y.apply(le.fit_transform)
Y_data = Y.values.flatten()

# save class names to a txt file
f = open('%sclass_names.txt' % DATA_PATH, 'w')
json.dump(class_names, f)
f.close()

# use TSNE to visualize the high dimension data in 2D
t0 = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=100)
tsne_results = tsne.fit_transform(X)
t1 = time.time()
print("TSNE took at %.2f seconds" % (t1 - t0))

# visualize TSNE and save the plot
x_axis = tsne_results[:, 0]
y_axis = tsne_results[:, 1]
plt.scatter(x_axis, y_axis, c=Y_data, cmap=plt.cm.get_cmap("jet", 100))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title("TSNE Visualization")
plt.savefig("./images/tsne_graph.png", dpi=600)

# split data into training and testing set
X_train, X_test, Y_train, Y_test \
    = train_test_split(X, Y, test_size=0.20, random_state=100, stratify=Y, shuffle=True)

# save the train and test csv files
X_train.to_csv("%sX_train.csv" % DATA_PATH)
X_test.to_csv("%sX_test.csv" % DATA_PATH)
Y_train.to_csv("%sY_train.csv" % DATA_PATH)
Y_test.to_csv("%sY_test.csv" % DATA_PATH)
