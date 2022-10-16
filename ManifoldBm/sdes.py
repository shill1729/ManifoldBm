# This module contains functions concerning SDEs. The basic routines are
# 1. Simulating a single sample path given a drift and diffusion function
# 2. Simulating an ensemble of sample paths given a drift and diffusion
# 3. Estimating the drift and diffusion coefficients from an ensemble of paths
#
# Here we assume that the coefficient functions are autonomous, i.e.
# functions of the state-space only: mu(x) and sigma(x) where x in R^d
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm


def euler_maruyama(x0, tn, drift, diffusion, ntime, noise_dim=2, sys_dim=None):
    """ Assumes x0, mu are vectors and b is matrix such that bb^T is the covariance matrix.

    Parameters:
        x0: initial point in N dimensional space.
        tn: the length of time for the simulation.
        drift: the drift function of mu(x).
        diffusion: diffusion coefficient, function of b(x) must be N x d.
        ntime: the number of time sub-intervals in the approximate solution.
        noise_dim: number of Brownian motions.
        sys_dim: the dimensions of system.

    Returns:
        An array of shape (n+1, m) representing the approximate solution to the SDE.
    """
    rng = default_rng()
    if sys_dim is None:
        sys_dim = x0.shape[0]
    h = tn / ntime
    x = np.zeros((ntime + 1, sys_dim))
    x[0, :] = x0
    for i in range(ntime):
        z = rng.normal(scale=np.sqrt(h), size=noise_dim)
        x[i + 1, :] = x[i, :] + drift(x[i, :]) * h + diffusion(x[i, :]).dot(z)
    return x


def sample_ensemble(x0, tn, drift, diffusion, npaths, ntime, noise_dim, sys_dim=None):
    """ Simulate an ensemble of sample paths.

    Parameters:
        x0: the initial point, a numpy 1-d array.
        tn: the time horizon to simulate over.
        drift: the drift coefficient function.
        diffusion: the diffusion coefficient function (the square root of the covariance function).
        npaths: the number of sample-paths in the ensemble to generate.
        ntime: the number of time-steps in each sample-path.
        noise_dim: the dimension of the noise.
        sys_dim: the dimension of the system.

    Returns:
        if sys_dim is equal to one, the returned array has shape (ntime+1, npaths)
        otherwise it has shape (npaths, ntime+1, sys_dim
    """
    if sys_dim is None:
        sys_dim = x0.shape[0]

    if sys_dim > 1:
        xt = np.zeros((npaths, ntime + 1, sys_dim))
        for i in range(npaths):
            xt[i] = euler_maruyama(x0, tn, drift, diffusion, ntime, noise_dim, sys_dim)
        return xt
    elif sys_dim == 1:
        xt = np.zeros((ntime + 1, npaths))
        for i in range(npaths):
            xt[:, i] = euler_maruyama(x0, tn, drift, diffusion, ntime, noise_dim, sys_dim)[:, 0]
        return xt
    elif sys_dim == 0:
        raise ValueError("input 'sys_dim' must be a positive integer.")


def ensemble_average(ensemble):
    """ Compute the sample average of a function of the process over independent sample-paths, conditional on the
    initial point of the process.

    Parameters:
        ensemble: the array of sample-paths.
    """
    # If ensemble has shape (N, n, d) where N is number of sample-paths, n is the number of
    # time steps, and d, the dimension of the process, then you can use
    # np.mean(..., axis=0) to sample-average over the ensemble at each time-step.
    n = len(ensemble.shape)
    if n == 3:
        return np.mean(ensemble, axis=0)
    elif n == 2:
        return np.mean(ensemble, axis=1)


def ensemble_std(ensemble):
    n = len(ensemble.shape)
    if n == 3:
        return np.std(ensemble, axis=0)
    elif n == 2:
        return np.std(ensemble, axis=1)


def estimate_drift(ensemble, x, h=10 ** -4, alpha=0.05):
    """ The simplest way to estimate the drift coefficient at a point of an SDE is to take the sample average
    of finite-differences over a small time-interval. The formula is simply
    (X_h_hat-x)/h where x is the initial point of the SDE, X_h_hat is the ensemble average of the terminal value
    and h is the length of time, which is assumed to be small.

    Parameters:
        ensemble: the (N,n+1,d) array of N sample paths of n+1 time-steps, and d dimensions.
        x: the array of d coordinates, the initial point of the SDE
        h: the small-time length to sample over.
        alpha: the confidence interval
    """
    # Compute the average trajectory and take the terminal value
    x_h_hat = ensemble_average(ensemble)
    x_h_hat = x_h_hat[-1]
    mu_hat = (x_h_hat - x) / h

    k = len(ensemble.shape)
    if k == 2:
        # Easily provide confidence intervals for drift coefficient of one-dimensional diffusions
        # In this case the ensemble shape is (n+1, N)
        epsilon = ensemble_std(ensemble)[-1] / np.sqrt(ensemble.shape[1])
        mu_hat_upper = (x_h_hat + epsilon * norm.cdf(1 - alpha / 2) - x) / h
        mu_hat_lower = (x_h_hat - epsilon * norm.cdf(1 - alpha / 2) - x) / h
        return mu_hat, mu_hat_lower, mu_hat_upper
    else:
        return mu_hat


def estimate_cov(ensemble, h=10 ** -3):
    """ The simplest way to estimate the covariance coefficient at a point of an SDE is to take the sample covariance
    of coordinates over the ensemble. The formula is simply cov(X^i, X^j)/h where h is the time horizon of the
    sample paths being simulated, and $X^i, X^j$ are the i,j-th coordinates of the terminal value X_h.

    Parameters:
        ensemble: the (N,n+1,d) array of N sample paths of n+1 timesteps, and d dimensions.
        h: the small-time length to sample over.
    """
    n = ensemble.shape[0]
    d = ensemble.shape[2]
    ijs = np.zeros((n, d))
    for k in range(n):
        ijs[k, :] = ensemble[k][-1, :]
    sigma = np.cov(ijs, rowvar=False) / h
    return sigma


def plot_ensemble(Y, d, what="Intrinsic"):
    if d == 1:
        fig = plt.figure()
        plt.plot(Y)
        plt.title(what+" sample ensemble")
        plt.show()
    elif d > 2:
        N = Y.shape[0]
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        for i in range(N):
            ax.plot3D(Y[i][:, 0], Y[i][:, 1], Y[i][:, 2])
        plt.title(what+" sample ensemble")
        plt.show()
    elif d == 2:
        N = Y.shape[0]
        fig = plt.figure()
        for i in range(N):
            plt.plot(Y[i][:, 0], Y[i][:, 1])
        plt.title(what+" sample ensemble")
        plt.show()