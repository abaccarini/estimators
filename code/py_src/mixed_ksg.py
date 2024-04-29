import scipy.spatial as ss
from scipy.special import digamma
from math import log, log2, e
import numpy as np

# EPSILON = 0.0
EPSILON = 1e-15  # (used in the original implementation)
# this was resulting in infinities for uniform_int when computing the variance
"""
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator

    Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
    y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
    k: k-nearest neighbor parameter

    Output: onmber of I(X;Y)
"""


def Mixed_KSG(x, y, k=5):

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    N = len(x)

    if x.ndim == 1:
        x = x.reshape((N, 1))
    dx = len(x[0])

    if y.ndim == 1:
        y = y.reshape((N, 1))
    dy = len(y[0])

    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float("inf"))[0][k] for point in data]
    ans = 0
    for i in range(N):
        kp = k
        # if knn_dis[i] == 0: # this was creating problems for purely discrete input distributions
        if np.isclose(knn_dis[i] , 0):  # is functionally zero, and we treat it as such
            kp = len(tree_xy.query_ball_point(data[i], EPSILON, p=float("inf")))  # type: ignore
            nx = len(tree_x.query_ball_point(x[i], EPSILON, p=float("inf")))  # type: ignore
            ny = len(tree_y.query_ball_point(y[i], EPSILON, p=float("inf")))  # type: ignore
        else:
            nx = len(tree_x.query_ball_point(x[i], knn_dis[i] - EPSILON, p=float("inf")))  # type: ignore
            ny = len(tree_y.query_ball_point(y[i], knn_dis[i] - EPSILON, p=float("inf")))  # type: ignore
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans * log2(e)  # used to convert from nats to bits
