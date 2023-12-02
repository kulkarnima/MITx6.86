"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    from scipy.stats import multivariate_normal

    n, _ = X.shape
    K, = mixture.p.shape

    soft_probs = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            soft_probs[i][j] = mixture.p[j]*multivariate_normal.pdf(X[i,:], mean=mixture.mu[j], cov=mixture.var[j])

    probs_x_theta = soft_probs.sum(axis=1).reshape(-1, 1)

    return soft_probs/probs_x_theta, np.log(probs_x_theta).sum()

    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape 

    post_sum = post.sum(axis=0)
    hat_mu = np.dot(post.T, X)/post_sum.reshape(-1,1)
    hat_p = post_sum/n
    hat_var = ((post*np.array([np.linalg.norm(X - hat_mu[j,:], axis=1)**2 for j in range(K)]).T).sum(axis=0))/(d*post_sum)
    
    return GaussianMixture(hat_mu, hat_var, hat_p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = np.finfo(dtype=np.float32).min
    new_ll = 0

    while abs(new_ll - old_ll) >= 1e-6*abs(new_ll):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_ll

    raise NotImplementedError
