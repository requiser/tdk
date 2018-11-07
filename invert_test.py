from os.path import dirname, join
import numpy as np
import csv
from os.path import dirname, join
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets.base import load_data, load_files
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import preprocessing
from sklearn import datasets
import sklearn.utils

df = datasets.load_iris()
# X = df.data[:, :4]
# y = df.target

module_path = dirname(__file__)
# data, target, target_names = load_data(module_path, 'dataset.csv')
data_file_name = join(module_path, 'data', 'dataset.csv')
with open(data_file_name) as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    n_samples = int(temp[0])
    n_features = int(temp[1])
    POP_SIZE = n_samples
    DNA_SIZE = n_features
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))
    temp = next(data_file)  # names of features
    # feature_names = np.array(temp)

    for i, d in enumerate(data_file):
        data[i] = np.asarray(d[:-1], dtype=np.float64)
        target[i] = np.asarray(d[-1], dtype=np.float64)

# with open(join(module_path, 'descr', 'dataset.rst')) as rst_file:
#    fdescr = rst_file.read()

# df = datasets.base.Bunch(data=data, target=target,feature_names=['input', 'output'])
dg = datasets.load_boston()

X = dg.data
POP_SIZE = X.shape[0]
DNA_SIZE = X.shape[1]
y = dg.target
print(POP_SIZE,DNA_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

clf = MLPRegressor(hidden_layer_sizes=(50, 60, 45, 32, 99, 53, 25),
                   activation='relu',
                   solver='adam',
                   learning_rate='adaptive',
                   max_iter=10000,
                   learning_rate_init=0.01,
                   alpha=0.1,
                   tol=0.000001,
                   verbose=False)

clf.fit(X_train, y_train)
y_pd = clf.predict(X_test)
z = clf.score(X_test, y_test)
print(z)
plt.subplot(3, 1, 1)
plt.plot(X_test, y_pd, 'o')
plt.subplot(3, 1, 2)

plt.plot(X_train, y_train, 'o')
plt.subplot(3, 1, 3)
plt.plot(X_test, y_pd, 'o')
plt.plot(X_train, y_train, 'o')
plt.show()
data_f = np.empty((POP_SIZE, DNA_SIZE))
for o in range(DNA_SIZE):
    for l in range(POP_SIZE):
        data_f[l, o] = np.random.uniform(0, np.max(X[:, o]))
#data_f.reshape((POP_SIZE,DNA_SIZE))
print(data_f.shape)
# examples = ['some text', 'another example text', 'example 3']

#    target = np.zeros((3,), dtype=np.int64)
#   target[0] = 0
#  target[1] = 1
# target[0] = 0

# testset = datasets.base.Bunch(data=examples, target=target)
# print(testset)
