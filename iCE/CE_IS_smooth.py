# ============================================================================
# Cross entropy method with smooth indicator function
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM, DTU
# =============================================================================
# * The methods work on the standard Gaussian space, transform your input random
# variables accordingly
# * SG and GM work only in low/moderate dimensions, use vMFNM instead
# * TO DO: - improve stability with logsumexp trick as in SMC
#          - tidy-up vMFNmix.py code before upload
# ============================================================================
# Based on:
# 1."Improved cross entropy-based importance sampling with a flexible mixture model"
#    Papaioannou et al.
#    Reliability engineering and system safety 191 (2019) 106564
# =============================================================================
# Version 2020-04
# =============================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.optimize as spo
#
import gaussmix as gaussmix
epsmin = np.finfo(np.double).tiny



"""
============================================================================
=====================Cross-entropy with single Gaussian=====================
============================================================================
"""
def SG(dim, N, delta_star, g_LSF, I_smooth, se, maxit=int(30)):    
    np.random.seed(seed=se)

    #==============================================
    # initialization of variables and storage
    samples  = list()                 # store the samples
    LSF_eval = np.empty((maxit,N))    # space for the sorted LSF evaluations
    s_param  = np.zeros(maxit+1)      # space for smoothness parameters
    
    # log prior of the random variables: standard Gaussian
    log_pi_pdf = lambda x: np.sum( sps.norm.logpdf(x, loc=0, scale=1), axis=0 )
    
    # log biasing parametric family (single Gaussian)
    gbias_rnd     = lambda N, mu_hat, Sigma_hat: np.random.multivariate_normal(mean=mu_hat, cov=Sigma_hat, size=N).T
    log_gbias_fun = lambda x, mu_hat, Sigma_hat: sps.multivariate_normal.logpdf(x.T, mean=mu_hat, cov=Sigma_hat)
    
    #==============================================
    # initial level
    j          = 0  
    s_param[j] = np.inf
    mu_hat     = np.zeros(dim) 
    Sigma_hat  = np.eye(dim) 

    # iCE method
    while True:
        # generate samples from biasing distribution
        theta = gbias_rnd(N, mu_hat, Sigma_hat)
        samples.append(theta)   # save

        # compute IS weights: 
        pi_pdf_eval    = log_pi_pdf(theta) 
        gbias_fun_eval = log_gbias_fun(theta, mu_hat, Sigma_hat)
        wIS_j          = np.exp(pi_pdf_eval - gbias_fun_eval)      

        # evaluate limit state function, indicator function and smooth indicator
        geval         = g_LSF(theta)
        Indf          = (geval <= 0).astype(int) 
        f_smooth_j    = I_smooth(s_param[j], geval)
        LSF_eval[j,:] = geval      # save
        print('Number of failure samples at level', j, '=', sum(Indf))  

        # compute cv of the ratio between the indicator and its smooth approximation
        ratio = Indf / f_smooth_j 
        if (np.isnan(ratio).any() == True):
            ratio = np.nan_to_num(ratio)
        #
        CoV = np.std(ratio)/ abs(np.mean(ratio))
        if (CoV <= delta_star) or (j >= maxit): 
            break 
        
        # compute smoothness parameters 
        fmin = lambda s: ( (  np.std(wIS_j*I_smooth(s,geval)) / \
                             np.mean(wIS_j*I_smooth(s,geval)) ) - delta_star)**2
        if j == 0:
            s_param[j+1] = spo.fminbound(fmin, 0, 10*np.mean(geval))
        else:
            s_param[j+1] = spo.fminbound(fmin, 0, s_param[j]) 
        print('\tSmoothness parameter:', s_param[j+1])

        # level weights associated to the smooth approximation of the indicator
        f_smooth_jp1 = I_smooth(s_param[j+1], geval)
        w_til_j      = (wIS_j * f_smooth_jp1).reshape((N,1))
        W_til        = sum(w_til_j)
        
        # parameter update: closed-form update
        mu_hat    = ((w_til_j.T @ theta.T) / W_til).flatten()
        num       = np.multiply(np.sqrt(w_til_j), (theta.T - mu_hat))
        Sigma_hat = (num.T @ num) / W_til + (1e-6*np.eye(dim))

        # next level
        j += 1

    # smoothness parameters
    s_param = s_param[:j+1]

    # calculation of the probability of failure (importance sampling)
    Pf = (1/N) * np.sum(Indf * wIS_j) 
    
    return Pf, s_param, samples, LSF_eval



"""
============================================================================
====================Cross-entropy with Gaussian mixture=====================
============================================================================
"""
def GM(dim, N, delta_star, k_mix, g_LSF, I_smooth, se, maxit=30):
    np.random.seed(seed=se)

    #==============================================
    # initialization of variables and storage
    samples  = list()                 # store the samples
    LSF_eval = np.empty((maxit,N))    # space for the sorted LSF evaluations
    s_param  = np.zeros(maxit+1)      # space for smoothness parameters
    
    # log prior of the random variables: standard Gaussian
    log_pi_pdf = lambda x: np.sum( sps.norm.logpdf(x, loc=0, scale=1), axis=0 )
           
    # biasing distribution parametric family
    gbias_rnd     = lambda N, wi_hat, mu_hat, Sigma_hat: gaussmix.GM_rvs(wi_hat, mu_hat, Sigma_hat, N)
    log_gbias_fun = lambda x, wi_hat, mu_hat, Sigma_hat: gaussmix.GM_logpdf(x, wi_hat, mu_hat, Sigma_hat) 

    # ==============================================
    # initial level
    j          = 0  
    s_param[j] = np.inf
    wi_hat     = (1/k_mix)*np.ones(k_mix)     # initial weights of the mixture, sum = 1
    mu_hat     = np.zeros((k_mix,dim))        # initial means of the mixture
    Sigma_hat  = np.zeros((k_mix,dim,dim))    # initial covariances of the mixture
    for k in range(k_mix):
        Sigma_hat[k] = np.eye(dim)

    # iCE method
    while True:
        # generate samples from biasing distribution
        theta = gbias_rnd(N, wi_hat, mu_hat, Sigma_hat)
        samples.append(theta)   # save

        # compute IS weights: 
        pi_pdf_eval    = log_pi_pdf(theta) 
        gbias_fun_eval = log_gbias_fun(theta.T, wi_hat, mu_hat, Sigma_hat)
        wIS_j          = np.exp(pi_pdf_eval - gbias_fun_eval)

        # evaluate limit state function, indicator function and smooth indicator
        geval         = g_LSF(theta)
        Indf          = (geval <= 0).astype(int) 
        f_smooth_j    = I_smooth(s_param[j], geval)
        LSF_eval[j,:] = geval      # save
        print('Number of failure samples at level', j, '=', sum(Indf))  

        # compute cv of the ratio between the indicator and its smooth approximation
        ratio = Indf / f_smooth_j 
        if (np.isnan(ratio).any() == True):
            ratio = np.nan_to_num(ratio)
        #
        CoV = np.std(ratio)/ abs(np.mean(ratio))
        if (CoV <= delta_star) or (j >= maxit): 
            break 
                
        # compute smoothness parameters 
        fmin = lambda s: ( (  np.std(wIS_j*I_smooth(s, geval)) / \
                             np.mean(wIS_j*I_smooth(s, geval)) ) - delta_star)**2
        if j == 0:
            s_param[j+1] = spo.fminbound(fmin, 0, 10*np.mean(geval))
        else:
            s_param[j+1] = spo.fminbound(fmin, 0, s_param[j]) 
        print('\tSmoothness parameter:', s_param[j+1])

        # level weights associated to the smooth approximation of the indicator
        f_smooth_jp1 = I_smooth(s_param[j+1], geval)
        w_til_j      = (wIS_j * f_smooth_jp1).reshape((N,1))

        # parameter update: Expectation-Maximization
        wi_hat, mu_hat, Sigma_hat = gaussmix.EM(theta.T, w_til_j, k_mix)

        # next level
        j += 1

    # smoothness parameters
    s_param = s_param[:j+1]

    # calculation of the probability of failure (importance sampling)
    Pf = (1/N) * sum(Indf * wIS_j) 
    
    return Pf, s_param, samples, LSF_eval