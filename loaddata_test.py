import csv
from os.path import dirname, join

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

module_path = dirname(__file__)
# data, target, target_names = load_data(module_path, 'dataset.csv')
data_file_name = join(module_path, 'data', 'dataset.csv')
with open(data_file_name) as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    n_samples = int(temp[0])
    n_features = int(temp[1])
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))
    temp = next(data_file)  # names of features
    # feature_names = np.array(temp)

    for i, d in enumerate(data_file):
        data[i] = np.asarray(d[:-1], dtype=np.float64)
        target[i] = np.asarray(d[-1], dtype=np.float64)

# with open(join(module_path, 'descr', 'dataset.rst')) as rst_file:
#    fdescr = rst_file.read()

df = datasets.base.Bunch(data=data, target=target,
                         feature_names=['input', 'output'])


