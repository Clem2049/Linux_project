# Module
import numpy as np
import matplotlib as plt

def pdf(x):
    """
    :param x: réel x
    :return: fonction de répartition de la loi normale
    """
    res = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)
    return res


def acc_rej_method(pdf, n=500, xmin=0, xmax=1):
    """
    :param pdf: fonction de répartition utilisée
    :param n: nombre de subdivision
    :param xmin: x minimum
    :param xmax: x maximum
    :return: retourne un graphique des points, avec couleur
            pour savoir si oui ou non ils sont sous la fonction de densité
    """
    M = np.sqrt(2 / np.pi) * np.exp(1 / 2)
    xIn, yIn = [], []
    xOut, yOut = [], []

    for i in range(n):
        nb = np.random.uniform(-1, 1)
        if nb >= 0:
            y = np.random.exponential(1)
        else:
            y = -np.random.exponential(1)
        u = np.random.uniform(0, 1)
        if u < pdf(y):
            xIn.append(y)
            yIn.append(u)
        else:
            xOut.append(y)
            yOut.append(u)

    plt.scatter(xIn, yIn, c='green')
    plt.scatter(xOut, yOut, c='red')

    x = np.linspace(-10, 10, 1000)
    res = []
    for e in x:
        res.append(pdf(e))
    plt.plot(x, res)
    plt.show()
    return 1

acc_rej_method(pdf, 10000)