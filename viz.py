import numpy as np
import matplotlib.pyplot as plt

def plot_dist(pdist, ax):
    side = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(side, side)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = np.concatenate([X_flatten, Y_flatten], 1)

    p = np.zeros(len(Z))
    for i in range(len(Z)):
        p[i] = pdist.prob(Z[i])
    p = np.reshape(p, shape)
    p /= np.sum(p)
    ax.contour(X, Y, p)

