import csv
from os.path import dirname, join
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn import datasets
from sklearn.neural_network import MLPRegressor

MUTATION_RATE = 0.01  # mutation probability
retain = 0.2
random_select = 0.03
N_GENERATIONS = 200

module_path = dirname(__file__)
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

    for i, d in enumerate(data_file):
        data[i] = np.asarray(d[:-1])
        target[i] = np.asarray(d[-1])

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                     'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = datasets.base.Bunch(data=data, target=target,
                             feature_names=feature_names)

    X = df.data[:POP_SIZE, :DNA_SIZE]
    y = df.target[:POP_SIZE]
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
    hl = []
    for k in range(int(np.ceil(DNA_SIZE / 2))):
        hl.append(POP_SIZE * (k + 1))

    mlpr = MLPRegressor(hidden_layer_sizes=hl,
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        max_iter=100,
                        verbose=True)
    mlpr.fit(X, y)


def main():
    f_values = np.empty(POP_SIZE)
    pop = create_population()
    for _ in range(N_GENERATIONS):
        f_values = mlpr.predict(pop)
        score = mlpr.score(pop, y)
        fitness = get_fitness(f_values, y)
        print('Score:\t', score, '\tFitness:\t', np.average(fitness))
        pop = evolve(pop, fitness)
    for r in range(DNA_SIZE):
        plt.plot(pop[:, r], f_values, 'o')
        plt.savefig('plot/' + str(r + 1) + 'plots.pdf')
        plt.close()
    print(mlpr.score(pop, y))
    plt.subplot(2, 1, 1)
    plt.plot(pop, f_values, 'o')
    plt.subplot(2, 1, 2)
    plt.plot(X, y, 'o')
    plt.show()


def get_fitness(values, y_exp):
    """Get the fitness of the predicted outputs.

    Args:
        values (ndarray): Predicted outputs based on the generated population.
        y_exp (ndarray): Expected outputs.

    Return:
        fness (ndarray): Fitness if the predicted outputs.
    """
    fness = np.empty(POP_SIZE)
    fnessm = np.empty((POP_SIZE, POP_SIZE))
    for jj in range(POP_SIZE):
        for j in range(POP_SIZE):
            fnessm[jj, j] = pow((1 / (1 + abs(la.norm((values[jj] - y_exp[j]))))), 8)
        fness[jj] = np.max(fnessm[jj])
    return fness


def create_population():
    """Generate population based on the training data.

    Args:

    Return:
        pop (ndarray): Generated input data population.

    """
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
    """Randomly mutate some of the children.

    Args:
        child (ndarray): New children from crossover.
        parents (ndarray): Surviving parent population.

    Return:
        child(ndarray): Children that either got mutated or returned as it was.

    """
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if child[point] == 0:
                child[point] = child[point] + np.random.uniform(
                    np.average(parents[:, point]) * -np.random.uniform(-2, 2),
                    np.average(parents[:, point]) * np.random.uniform(-2, 2))
            else:
                child[point] * (np.random.uniform(0.5, 2))
            if dt[point] == 0:
                child[point] = np.random.choice((0, 1))
    return child


def crossover(father, mother, parents):  # mating process (genes crossover)
    children = []
    for _ in range(2):
        child = []
        for q in range(DNA_SIZE):
            child.append(np.random.choice((father[q], mother[q])))
        child = mutate(child, parents)
        children.append(child)
    np.asarray(children)
    return children


def evolve(pop, fitness):
    """Evolve a population of networks.

    Args:
        pop (ndarray): An array of input parameters
        fitness (ndarray): An array of fitness scores

    Returns:
        (list): The evolved population of networks

    """
    graded = [x[1] for x in sorted(zip(fitness, pop), key=lambda x: x[0], reverse=True)]
    graded = np.asarray(graded)
    fitness = [x for x in sorted(fitness, reverse=True)]
    fitness = np.asarray(fitness)
    retains = np.average(((pow(np.min(fitness), 1) / np.max(fitness)), retain))
    retain_length = int(np.floor(len(graded) * retains))
    parents = np.asarray(graded[:retain_length])

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if random_select > np.random.rand():
            np.concatenate((parents, [individual]), axis=0)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = POP_SIZE - parents_length
    children = np.empty(0)
    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:

        # Get a random mom and dad.
        male = np.random.randint(0, parents_length - 1)
        female = np.random.randint(0, parents_length - 1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]

            # Breed them.
            babies = np.asarray(crossover(male, female, parents))

            # Add the children one at a time.
            for _ in range(babies.shape[0]):

                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    if len(children) == 0:
                        children = babies
                    else:
                        children = np.concatenate((children, [babies[_]]), axis=0)
    if len(children) > (POP_SIZE - len(parents)):
        parents = np.concatenate((parents, [children[0]]), axis=0)
    else:
        parents = np.concatenate((parents, children), axis=0)
    return parents


if __name__ == '__main__':
    main()
