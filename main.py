import numpy as np
import csv
from os.path import dirname, join
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from gennn import Gen

CROSS_RATE = 0.8  # mating probability (DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENERATIONS = 10000

module_path = dirname(__file__)
# data, target, target_names = load_data(module_path, 'dataset.csv')
data_file_name = join(module_path, 'data', 'iris.csv')
with open(data_file_name) as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    n_samples = int(temp[0])
    n_features = int(temp[1])
    data = np.empty((n_samples, n_features))
    POP_SIZE = n_samples
    DNA_SIZE = n_features
    target = np.empty((n_samples,))
    temp = next(data_file)  # names of features
    # feature_names = np.array(temp)

    for i, d in enumerate(data_file):
        data[i] = np.asarray(d[:-1], dtype=np.float64)
        target[i] = np.asarray(d[-1], dtype=np.float64)

    # with open(join(module_path, 'descr', 'dataset.rst')) as rst_file:
    #    fdescr = rst_file.read()
    feature_names = ['setosa', 'versicolor', 'virginica']
    # feature_names = ['input', 'output']
    df = datasets.base.Bunch(data=data, target=target,
                             feature_names=feature_names)


def main():
    # df = datasets.load_iris()
    # X = df.data[:, :4]

    X = df.data[:, :4]
    y = df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    mlpr = MLPRegressor(hidden_layer_sizes=(50, 60, 45, 32, 99, 53, 25),
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        max_iter=10000,
                        verbose=True)

    mlpr.fit(X_train, y_train)

    # mlpr.predict(X)

    # print(Gen.create_population().shape)
    # print(df.data)
    # Gen.create_population()
    # Gen(mlpr)
    pop = create_population().data

    for _ in range(N_GENERATIONS):
        F_values = mlpr.predict(pop)  # compute function value by extracting DNA
        # something about plotting
        # if 'sca' in globals(): sca.remove()
        # sca = plt.scatter(pop, F_values, s=200, lw=0, c='red', alpha=0.5)
        # plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    # mlpr.score(pop[100].reshape(1, -1), y[100].reshape(1, -1))
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    print(np.argmax(fitness))
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child


def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


def create_population():
    data_f = np.random.randint(0, 9999, size=(POP_SIZE, DNA_SIZE))
    pop = np.empty((POP_SIZE, DNA_SIZE))
    for z, g in enumerate(data_f):
        pop[z] = np.asarray(g[:], dtype=np.float64)

    return datasets.base.Bunch(data=pop)


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


if __name__ == '__main__':
    main()
