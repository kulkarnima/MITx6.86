"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    mu, var, pi = mixture

    delta = X.astype(bool).astype(int)

    f = -(np.sum(X**2, axis=1)[:,None] + delta@mu.T**2 - 2*X@mu.T)/(2*var)
    f -= (np.sum(delta, axis=1).reshape(-1, 1)/2.) @ np.log(2*np.pi*var).reshape(-1, 1).T

    f += np.log(pi + 1e-16)

    log_sums = logsumexp(f, axis=1).reshape(-1, 1)
    log_posts = f - log_sums

    log_likelihoods = np.sum(log_sums, axis=0).item()

    return np.exp(log_posts), log_likelihoods

    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, _ = X.shape
    hat_mu, _, _ = mixture
    hat_pi = np.sum(post, axis=0)/n

    delta = X.astype(bool).astype(int)

    hat_mu_d = post.T@delta
    hat_mu_n = post.T@X
    index = np.where(hat_mu_d >= 1)
    hat_mu[index] = hat_mu_n[index]/hat_mu_d[index]

    norms = np.sum(X**2, axis=1)[:, None] + delta@hat_mu.T**2 - 2*X@hat_mu.T

    hat_var_d = np.sum(post*np.sum(delta, axis=1).reshape(-1, 1), axis=0)
    hat_var_n = np.sum(post*norms, axis=0)
    hat_var = np.maximum(hat_var_n/hat_var_d, min_variance)

    return GaussianMixture(hat_mu, hat_var, hat_pi)

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
        mixture = mstep(X, post, mixture)
        
    return mixture, post, new_ll

    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()

    mu, _, _ = mixture

    posts, _ = estep(X, mixture)
    
    missing_index = np.where(X == 0)
    X_pred[missing_index] = (posts@mu)[missing_index]

    return X_pred

    raise NotImplementedError
