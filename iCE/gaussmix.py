# ============================================================================
# useful functions for Gaussian mixture 
# ============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM
# ============================================================================
# * Geyer et al. (2018) - Cross entropy-based importance sampling using Gaussian
#   densities revisited. Structural Safety.
# * Some ideas from:
#   https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
# ============================================================================
# Version 2018-12
# ============================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
realmin = np.finfo(np.double).tiny
# from sklearn.cluster import DBSCAN SpectralClustering
# from sklearn import mixture



"""
============================================================================
=================Generate samples from a Gaussian mixture===================
============================================================================
"""
def GM_rvs(wi_hat, mu_hat, Sigma_hat, N):
    k_mix = mu_hat.shape[0]
    if (k_mix == 1):
        theta = np.random.multivariate_normal(mean=mu_hat.flatten(), cov=Sigma_hat[0], size=N)
    else:
        # determine number of samples for each mixture component
        N_k  = np.round(wi_hat*N)       
        sN_k = sum(N_k) 
        if (sN_k != N):
            idx       = np.argmax(N_k)
            N_k[idx] -= sN_k - N
        N_k   = N_k.astype(int) 
        theta = np.concatenate( [np.random.multivariate_normal(mean=mu, cov=sigma, size=N_k[i])
                                for i, wi, mu, sigma in zip(range(k_mix), wi_hat, mu_hat, Sigma_hat)] )
    #
    return theta.T



"""
============================================================================
========================Evaluate Gaussian mixture===========================
============================================================================
"""
def GM_pdf(theta, wi_hat, mu_hat, Sigma_hat):
    #
    N         = theta.shape[0]
    gmix_eval = np.zeros(N)
    for wi, mu, sigma in zip(wi_hat, mu_hat, Sigma_hat):
        gmix_eval += wi * sps.multivariate_normal.pdf(theta, mean=mu, cov=sigma)
    #
    return gmix_eval 



"""
============================================================================
======================Evaluate logGaussian mixture==========================
============================================================================
"""
def GM_logpdf(theta, wi_hat, mu_hat, Sigma_hat):
    #
    N         = theta.shape[0]
    k_mix     = len(wi_hat)
    loggmix_k = np.zeros((N, k_mix))
    for k in range(k_mix):
        loggmix_k[:,k] = np.log(wi_hat[k]) + sps.multivariate_normal.logpdf(theta, mean=mu_hat[k,:], cov=Sigma_hat[k])
    loggmix_eval = sp.special.logsumexp(loggmix_k, axis=1)
    #
    return loggmix_eval 



"""
============================================================================
========================Expectation-Maximization============================
============================================================================
"""
def EM(theta, w_j, k_mix, tol=1e-5, maxit=700):
    #
    N         = theta.shape[0]
    ll_old, t = 0, 0
    # db     = DBSCAN(eps=0.3, min_samples=10).fit(theta)
    # labels = db.labels_
    # k_mix  = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters: %d' % k_mix)

    # random initialization of mixture values
    u = np.zeros(1000)   # large value to enter the while
    while (k_mix != len(u)):
        idx       = np.random.choice(range(N), k_mix)
        theta_rnd = theta[idx, :]
        label     = np.argmax( (theta_rnd @ theta.T) - 0.5*(theta_rnd*theta_rnd).sum(axis=1)[:, None], axis=0)
        u         = np.unique(label)
    l_rnd = np.zeros((N, k_mix), dtype=int)
    for i in range(N):
        l_rnd[i, label[i]] = 1

    # EM algorithm
    while True:
        # M-step 
        wi_hat, mu_hat, Sigma_hat = maximization(theta, w_j, l_rnd)     

        # E-step
        ll_new_wj, l_rnd = expectation_log(theta, w_j, wi_hat, mu_hat, Sigma_hat)
   
        # check convergence 
        delta = abs(ll_new_wj - ll_old)
        if (delta < tol*abs(ll_new_wj)):
            print('\tEM converged in', t, 'iterations')
            break
        if (t > maxit):
            raise RuntimeWarning('WARNING: maximum number of iterations in EM exceeded')
        t     += 1
        ll_old = ll_new_wj
    #
    return wi_hat, mu_hat, Sigma_hat    

#==============================================================================================
def maximization(theta, w_j, l_rnd):
    '''given the current estimate of the mixture, determine the parameters'''
    dim   = theta.shape[1]
    k_mix = l_rnd.shape[1]

    # 0. assign samples to each mixture
    l_rnd = w_j * l_rnd         # apply the weights to the mixture values
    N_k   = l_rnd.sum(axis=0)
    if any(N_k == 0):
        N_k += realmin

    # 1. update mixture weights
    wi_hat = N_k / sum(N_k)
    if ((1 - np.isclose(sum(wi_hat), 1)) != 0):
        raise RuntimeError('ERROR: weights do not sum to 1')  

    # 2. update mixture means
    mu_hat = (l_rnd.T @ theta) / N_k[:,None]

    # 3. update mixture covariances
    Sigma_hat = np.zeros((k_mix, dim, dim))
    for k in range(k_mix):
        thetap       = theta - mu_hat[k, :]
        Sigma_hat[k] = (l_rnd.T[k, :, None, None] * np.multiply(thetap[:, :, None], thetap[:, None, :])).sum(axis=0)
    Sigma_hat /= N_k[:, None, None]
    
    # 'stabilize' covariance
    for k in range(k_mix):
        Sigma_hat[k] += np.eye(dim)*(1e-7)
    #
    return wi_hat, mu_hat, Sigma_hat

#==============================================================================================
def expectation_log(theta, w_j, wi_hat, mu_hat, Sigma_hat):
    '''given the current estimate of the parameters, determine the mixture'''
    N     = theta.shape[0]
    k_mix = len(wi_hat)

    # 0. mixture values (in logspace)
    logl_rnd = np.zeros((N, k_mix))
    for k in range(k_mix):
        logl_rnd[:,k] = np.log(wi_hat[k]) + sps.multivariate_normal.logpdf(theta, mean=mu_hat[k,:], cov=Sigma_hat[k])

    # 1. update loglikelihood
    logmix      = sp.special.logsumexp(logl_rnd, axis=1)   # np.log(np.sum(np.exp(logl_rnd))) 
    logl_new_wj = sum(w_j.flatten() * logmix) / sum(w_j)

    # 2. new estimate of mixture values (in logspace)
    logl_rnd -= logmix[:,None]
    l_rnd     = np.exp(logl_rnd)

    # adjust the l_rnd
    label = np.argmax(l_rnd, axis=1)
    u     = np.unique(label)
    if (np.size(l_rnd, axis=1) != len(u)):
        l_rnd = l_rnd[:,u] 
    #           
    return logl_new_wj, l_rnd

#==============================================================================================
def expectation(theta, w_j, wi_hat, mu_hat, Sigma_hat):
    '''given the current estimate of the parameters, determine the mixture'''
    N     = theta.shape[0]
    k_mix = len(wi_hat)

    # 1. new estimate of mixture values
    l_rnd = np.zeros((N, k_mix))
    for k in range(k_mix):
        l_rnd[:,k] = wi_hat[k] * sps.multivariate_normal.pdf(theta, mean=mu_hat[k,:], cov=Sigma_hat[k])
    mix    = l_rnd.sum(axis=1)
    l_rnd /= mix[:,None]

    # adjust the l_rnd
    label = np.argmax(l_rnd, axis=1)
    u     = np.unique(label)
    if (np.size(l_rnd, axis=1) != len(u)):
        l_rnd = l_rnd[:,u] 

    # 2. update loglikelihood
    logmix      = np.log(mix)
    logl_new_wj = sum(w_j.flatten() * logmix) / sum(w_j)
    #      
    return logl_new_wj, l_rnd