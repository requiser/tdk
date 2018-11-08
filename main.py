import numpy as np
import csv
from os.path import dirname, join
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy.linalg as LA
from termcolor import colored

# CROSS_RATE = 0.9  # mating probability (DNA crossover)
MUTATION_RATE = 0.3  # mutation probability
retain = 0.2
random_select = 0.01
N_GENERATIONS = 200

module_path = dirname(__file__)
# data, target, target_names = load_data(module_path, 'dataset.csv')
data_file_name = join(module_path, 'data', 'boston_house_prices.csv')
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
        data[i] = np.asarray(d[:-1])
        target[i] = np.asarray(d[-1])

    # with open(join(module_path, 'descr', 'dataset.rst')) as rst_file:
    #    fdescr = rst_file.read()
    # feature_names = ['sepal length (cm)', 'sepal width (cm)',
    # 'petal length (cm)', 'petal width (cm)']
    # feature_names = ['input', 'output']
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                     'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = datasets.base.Bunch(data=data, target=target,
                             feature_names=feature_names)
    # DNA_SIZE = 5
    # POP_SIZE = 15
    # print(DNA_SIZE)
    X = df.data[:POP_SIZE, :DNA_SIZE]
    y = df.target
    dt = []
    dt_a = []
    for u in range(DNA_SIZE):
        for t in range(POP_SIZE):
            if X[t, u] == 0:
                dt_a.append(0)
            else:
                if X[t, u] == 1:
                    dt_a.append(0)
                else:
                    dt_a.append(1)
        if np.asarray(dt_a).sum() == 0:
            dt.append(0)
        else:
            dt.append(1)
        dt_a = []
    dt = np.asarray(dt)
    print(dt)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    mlpr = MLPRegressor(hidden_layer_sizes=(POP_SIZE, 2 * POP_SIZE, 4 * POP_SIZE, 8 * POP_SIZE),
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        tol=0.000001,
                        max_iter=100,
                        verbose=True)

    mlpr.fit(X, y)


def main():
    # df = datasets.load_iris()
    # X = df.data[:, :4]

    # mlpr.predict(X)
    # print(Gen.create_population().shape)
    # print(df.data)
    # Gen.create_population()
    # Gen(mlpr)
    F_values = np.empty(POP_SIZE)
    for __ in range(2):
        pop = create_population()
        for _ in range(N_GENERATIONS):
            F_values = mlpr.predict(pop)
            score = mlpr.score(pop, y)
            # F_values = mlpr.predict(pop)  # compute function value by extracting DNA
            # something about plotting
            # if 'sca' in globals(): sca.remove()
            # sca = plt.scatter(pop, F_values, s=200, lw=0, c='red', alpha=0.5)
            # plt.pause(0.05)
            # GA part (evolution)
            fitness = get_fitness(F_values, y)

            print('Score:\t', score, '\tFitness:\t', np.average(fitness))

            # print(fitness)
            # mlpr.score(pop[100].reshape(1, -1), y[100].reshape(1, -1))
            # print(_, ". Most fitted DNA: ", pop[np.argmax(fitness), :DNA_SIZE])
            # print(np.max(fitness))
            # pop = select(pop, fitness)
            pop = evolve(pop, fitness, score)
            # pop_copy = pop.copy()
            # for parent in pop:
            #    child = crossover(parent, pop_copy)
            #    child = mutate(child)
            #    parent[:] = child

        # print(pop)
        # print(F_values)
        print(mlpr.score(pop, y))
        plt.subplot(2, 1, 1)
        plt.plot(pop, F_values, 'o')
        plt.subplot(2, 1, 2)
        plt.plot(X, y, 'o')
        plt.show()
        best = np.empty((2, DNA_SIZE))
        best[__, :] = pop[0, :]
    print(mlpr.score(best, y[:2]))
    plt.subplot(2, 1, 1)
    plt.plot(best, mlpr.predict(best), 'o')
    plt.subplot(2, 1, 2)
    plt.plot(X, y, 'o')
    plt.show()


def get_fitness(values, y_exp):
    # F_val = np.empty((POP_SIZE, POP_SIZE))
    # F_values = np.empty(POP_SIZE)
    # for l in range(POP_SIZE):
    #    for k in range(POP_SIZE):
    #        F_val[l, k] = mlpr.predict(pop[k].reshape(1, -1))
    #    F_values[l] = np.max(F_val[l])
    fness = np.empty(POP_SIZE)
    for j in range(POP_SIZE):
        fness[j] = pow((1 / 1 + (pow(LA.norm((values[j] - y_exp[j])), 4))), 4)
    return fness


def create_population():
    # data_f = np.random.uniform(0, np.argmax(df.data, 0) * 1.1, size=POP_SIZE)
    data_f = np.empty((POP_SIZE, DNA_SIZE))
    for o in range(DNA_SIZE):
        for l in range(POP_SIZE):
            if dt[o] == 0:
                data_f[l, o] = np.random.choice((0, 1), size=1,
                                                p=(1 - np.average(X[:, o]), np.average(X[:, o])))
            else:
                data_f[l, o] = np.random.uniform(0, np.average(X[:, o]) * 2, size=1)
    pop = np.empty((POP_SIZE, DNA_SIZE))
    for z, g in enumerate(data_f):
        pop[z] = np.asarray(g[:], dtype=np.float64)

    return pop


def mutate(child, parents):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if child[point] == 0:
                child[point] = child[point] + np.random.uniform(
                    np.average(parents[:, point]) * -np.random.uniform(-3, 3),
                    np.average(parents[:, point]) * np.random.uniform(-3, 3))
            else:
                child[point] * (np.random.uniform(0.3, 3))
            if dt[point] == 0:
                child[point] = np.random.choice((0, 1)
                                                # , p=(1 - np.average(X[:, point]), np.average(X[:, point]))
                                                )
    return child


# def select(pop, fitness):  # nature selection wrt pop's fitness
#     idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
#                            p=fitness / fitness.sum())
#     return pop[idx]


def crossover(father, mother, parents):  # mating process (genes crossover)
    # if np.random.rand() < CROSS_RATE:
    #    i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
    #    cross_points = np.random.randint(0, DNA_SIZE)  # choose crossover points
    #    parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    # return parent
    children = []
    for _ in range(2):
        child = []
        # Loop through the parameters and pick params for the kid.

        for q in range(DNA_SIZE):
            child.append(np.random.choice((father[q], mother[q])))
        # np.asarray(child)
        # Now create a network object.
        # network = Network(self.nn_param_choices)
        # network.create_set(child)

        # Randomly mutate some of the children.
        # if mutate_chance > random.random():
        #    network = self.mutate(network)
        child = mutate(child, parents)
        children.append(child)
    np.asarray(children)
    return children


def evolve(pop, fitness, score):
    """Evolve a population of networks.

    Args:
        pop (ndarray): An array of input parameters
        fitness (ndarray): An array of fitness scores
        score (float): The score of the mlpregressor

    Returns:
        (list): The evolved population of networks

    """
    # Get scores for each network.

    # graded = [(fitness, pop) for _ in range(POP_SIZE)]
    # print(graded)
    # Sort on the scores.
    # graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    graded = [x[1] for x in sorted(zip(fitness, pop), key=lambda x: x[0], reverse=False)]
    graded = np.asarray(graded)
    # Get the number we want to keep for the next gen.
    # if 1 / abs(score - 1) > 1 / (retain + 1):
    #    retains = 1 - (2 - retain - abs(score - 1))
    #    print('Retain:\t', retains)
    # else:
    retains = np.average((np.max(fitness), np.average(fitness))) / np.sum(fitness)

    retain_length = int(len(graded) * retains)

    # The parents are every network we want to keep.
    parents = np.asarray(graded[:retain_length])

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if random_select > np.random.rand():
            np.concatenate((parents, [individual]), axis=0)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    cf = 0
    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:

        # Get a random mom and dad.
        male = np.random.randint(0, parents_length - 1)
        female = np.random.randint(0, parents_length - 1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]
            # fm = np.concatenate((male, female), 0)
            # Breed them.
            babies = np.asarray(crossover(male, female, parents))
            # Add the children one at a time.
            k = 0
            if cf == 0:
                children = babies
                cf = cf + 1
            else:
                for _ in range(babies.shape[0]):
                    # print(children.shape)
                    # print(babies.shape)
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        # print(babies[k].shape)
                        # print(babies[k])
                        children = np.concatenate((children, [babies[k]]), axis=0)
                    k = k + 1
    parents = np.concatenate((parents, children), axis=0)
    return parents


if __name__ == '__main__':
    main()
