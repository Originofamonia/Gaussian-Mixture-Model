# Author -----Sohaib Kiani and Usman Sajid
import pandas as pd
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
    k = 1

    gmm = GaussianMixModel(x, k)
    gmm.fit()
    plot.plot_1D(gmm, x, col)


def main():
    # For_Glass()
    for_iris(k=3)


if __name__ == "__main__":
    main()
