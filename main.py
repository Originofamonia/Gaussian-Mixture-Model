"""
https://github.com/Originofamonia/Gaussian-Mixture-Model
The smaller the epsilon is, the stronger the privacy will be.
"""
import pandas as pd
import numpy as np
from numpy import genfromtxt

from GMM import *
import util as plot


def get_iris_data():
    data = pd.read_csv("Data/Iris.csv", header=0)
    data = data.reset_index()
    replace_map = {'Species': {'Iris-virginica': 1, 'Iris-versicolor': 2, 'Iris-setosa': 3}}
    data.replace(replace_map, inplace=True)
    label = data[['Species']]
    col = ['SepalWidthCm', 'PetalLengthCm']
    x = data[col]
    x = np.array(x)
    return col, label, x


def for_iris(k=2):
    col, label, x = get_iris_data()
    gmm = GaussianMixModel(x, k)
    gmm.fit()
    plot.plot_2D(gmm, x, col, label)


def for_glass(k=2):
    data = pd.read_csv("Data/glass.csv", header=0)
    col = 'Fe'
    x = data[[col]]
    x = np.array(x)
    # k = 1

    gmm = GaussianMixModel(x, k)
    gmm.fit()
    plot.plot_1D(gmm, x, col)


def lifesci(k=3):
    lifesci = genfromtxt('data/lifesci.csv', delimiter=',')
    lifesci = lifesci[:, :10]
    normalizer = 1.0 / np.sqrt(np.amax(np.sum(lifesci ** 2, axis=1)))
    x = lifesci * normalizer
    col = ['1', '2']

    # col, label, x = get_iris_data()

    label = np.zeros(len(x))

    gmm = GaussianMixModel(x, total_eps=0.1, k=k)
    # gmm.total_eps = 0  # 0: non-DP version
    gmm.fit()

    plot.plot_2D(gmm, x, col, label)


def main():
    np.random.seed(44)
    # for_iris(k=3)
    lifesci()


if __name__ == "__main__":
    main()
