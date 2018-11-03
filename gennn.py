import csv
import random
from os.path import join, dirname

import pandas as pd

import numpy as np
from sklearn import datasets

CROSS_RATE = 0.8  # mating probability (DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENERATIONS = 10
X_BOUND = [0, 5]

module_path = dirname(__file__)
# data, target, target_names = load_data(module_path, 'dataset.csv')
data_file_name = join(module_path, 'data', 'dataset.csv')
with open(data_file_name) as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    n_samples = int(temp[0])
    n_features = int(temp[1])
    data = np.empty((n_samples, n_features))
    DNA_SIZE = n_samples
    POP_SIZE = n_features
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

df = datasets.load_iris()


class Gen:

    # y_exp = pd.read_csv('exp.csv')

    @staticmethod
    def create_population():
        pop = np.random.randint(0, 9999, size=(POP_SIZE, DNA_SIZE))
        return pop


    def mutate(child):
        for point in range(DNA_SIZE):
            if np.random.rand() < MUTATION_RATE:
                child[point] = 1 if child[point] == 0 else 0
        return child


