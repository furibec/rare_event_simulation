# ============================================================================
# Failure-informed cross-entropy-based importance sampling
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM
# =============================================================================
# * The method work on the standard Gaussian space, transform your input random
# variables accordingly
# ============================================================================
# Based on:
# 1."Cross-entropy-based importance sampling with failure-informed dimension 
#    reduction for rare event simulation"
#    Uribe et al. 2020 (submitted)
# =============================================================================
# Version 2020-04
# =============================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.optimize as spo
import matplotlib
import matplotlib.pyplot as plt
#
epsmin = np.finfo(np.double).tiny



"""
============================================================================
=====================Cross-entropy with single Gaussian=====================
================================Local FIS===================================
============================================================================
"""
# ==========================================================================
def SG_local(dim, N, delta_star, g_LSF, I_smooth, grad_logI_smooth, refin, eps, se, maxit=int(30)): 
    np.random.seed(seed=se)

    #==============================================
    # initialization of variables and storage
    samples  = list()                 # store the samples
    LSF_eval = np.empty((maxit,N))    # space for the LSF evaluations
    ranks    = np.empty(maxit+1)      # space for the ranks
    s_param  = np.zeros(maxit+1)      # space for smoothness parameters
    g_call, grad_call = 0, 0          # track number of LSF and grad evaluations

    # prior of the random variables: standard Gaussian
    pi_rnd     = lambda N, d: np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=N).T if d>0 else np.tile(np.array([]),[N,1]).T
    log_pi_pdf = lambda x: np.sum( sps.norm.logpdf(x, loc=0, scale=1), axis=0 )
    
    # biasing parametric family (single Gaussian)
    gbias_rnd     = lambda N, mu_hat, Sigma_hat: np.random.multivariate_normal(mean=mu_hat, cov=Sigma_hat, size=N).T
    log_gbias_fun = lambda x, mu_hat, Sigma_hat: sps.multivariate_normal.logpdf(x.T, mean=mu_hat, cov=Sigma_hat)
    
    #==============================================
    # initial level
    j          = 0  
    s_param[j] = np.inf          # initial smoothing parameter
    mu_hat     = np.zeros(dim)   # initial biasing mean 
    Sigma_hat  = np.eye(dim)     # initial biasing covariance

    # to avoid 'before assignment warnings'
    tilde_theta_p            = []
    Phi_r, Phi_p             = [], []
    r, mu_hat_r, Sigma_hat_r = [], [], [] 

    # iCEred method    
    while True:
        if j == 0:   # at the initial level use the whole space
            # draw samples in d (theta in R^{d x N})
            theta = gbias_rnd(N, mu_hat, Sigma_hat)

            # evaluate components of the weigths
            # pi_pdf_eval    = log_pi_pdf(theta) 
            # gbias_fun_eval = log_gbias_fun(theta, mu_hat, Sigma_hat)
            wIS_j = np.ones(N)   # IS weights are 1 at j=0
        else:
            # draw samples in FIS and CS
            tilde_theta_r = gbias_rnd(N, mu_hat_r, Sigma_hat_r)
            # tilde_theta_p = pi_rnd(N, dim-r).reshape((dim-r,N))  # 'refresh' CS samples (not necessary)

            # evaluate components of the weigths in r
            pi_pdf_eval    = log_pi_pdf(tilde_theta_r) 
            gbias_fun_eval = log_gbias_fun(tilde_theta_r, mu_hat_r, Sigma_hat_r)

            # compute IS weights: 
            wIS_j = np.exp(pi_pdf_eval - gbias_fun_eval)       

            # construct full theta 
            theta_r = Phi_r @ tilde_theta_r
            theta_p = Phi_p @ tilde_theta_p
            theta   = theta_r + theta_p
        samples.append(theta)   # store samples      

        #============================================================================
        # evaluate limit state function, indicator function and smooth indicator
        geval          = g_LSF(theta)
        Indf           = (geval <= 0).astype(int) 
        f_smooth_j     = I_smooth(s_param[j], geval)
        LSF_eval[j,:]  = geval      # save
        g_call        += N
        print('Number of failure samples at level', j, '=', sum(Indf))  

        # compute cv of the ratio between the indicator and its smooth approximation
        ratio = Indf / f_smooth_j 
        if (np.isnan(ratio).any() == True):
            ratio = np.nan_to_num(ratio)
        #
        CoV = np.std(ratio)/ abs(np.mean(ratio)) if sum(ratio)>0 else 10
        if (CoV <= delta_star) or (j >= maxit): 
            break 
        
        # compute smoothing parameter
        fmin = lambda s: ( (  np.std(wIS_j*I_smooth(s,geval)) / \
                             np.mean(wIS_j*I_smooth(s,geval)) ) - delta_star)**2
        if j == 0:
            s_param[j+1] = spo.fminbound(fmin, 0, 10*np.mean(geval))
        else:
            s_param[j+1] = spo.fminbound(fmin, 0, s_param[j]) 
        print('\tSmoothing parameter:', s_param[j+1])

        # level weights associated to the smooth approximation of the indicator
        f_smooth_jp1 = I_smooth(s_param[j+1], geval)
        w_til_j      = (wIS_j * f_smooth_jp1).reshape((N,1))
        W_til        = sum(w_til_j)
 
        #============================================================================
        # estimate the matrix H
        grad_logI_eval = grad_logI_smooth(s_param[j+1], geval, theta)  # in R^{N x d}
        grad_call     += N 
        num_H          = np.multiply(np.sqrt(w_til_j), grad_logI_eval)
        H_hat          = (1/W_til) * (num_H.T @ num_H)
        #
        if (np.isinf(H_hat).any() == True) or (np.sum(H_hat) <= 1e-5):
            print('\tAlmost zero gradient at smoothing parameter', s_param[j+1], ', keeping previous level rank and basis')
        else:
            # solve eigenvalue problem for H_hat
            eigval, eigvec = sp.linalg.eigh(H_hat)
            idx    = np.argsort(-np.real(eigval))    # index sorting descending
            eigval = np.real(eigval[idx])
            eigvec = eigvec[:,idx]
            
            # find the rank
            ss       = 0.5*np.cumsum(eigval[::-1])[::-1]   # cumsum backwards
            r        = np.argmax(ss <= eps)                # stops at the 1st True
            ranks[j] = r                                   # store level ranks
            print('\tRank:', r)
            # or
            # for i in range(dim):
            #     eigv_sum_i = 0.5*sum(eigval[i:])
            #     if (eigv_sum_i <= eps):
            #         r = i
            #         break     
            # r = max(1, r)   # the rank is at least 1     

        # construct the projectors
        if (j > 0):   # store basis of the previous level
            Phi_r_old, Phi_p_old = np.copy(Phi_r), np.copy(Phi_p)
        Phi_r = eigvec[:,:r]      # basis of the local FIS
        Phi_p = eigvec[:,r:]      # basis of the local CS

        # samples in the local FIS and CS
        tilde_theta_r = Phi_r.T @ theta
        tilde_theta_p = Phi_p.T @ theta

        # compute corrected weights due to the change in basis between levels
        if (j > 0):
            # new param
            new_tilde_theta = np.vstack([tilde_theta_r, tilde_theta_p])

            # new reference mean (mu_hat_p = np.zeros(dim-r))
            const1       = Phi_r_old @ mu_hat_r #+ Phi_p_old @ mu_hat_p
            new_mu_hat_r = Phi_r.T @ const1
            new_mu_hat_p = Phi_p.T @ const1 
            new_mu       = np.hstack([new_mu_hat_r, new_mu_hat_p])

            # new reference covariance
            const2           = Phi_r_old @ (Sigma_hat_r @ Phi_r_old.T)
            const3           = Phi_p_old @ Phi_p_old.T #Phi_p_old @ (Sigma_hat_p @ Phi_p_old.T)
            new_Sigma_hat_r  = (Phi_r.T @ const2 @ Phi_r) + (Phi_r.T @ const3 @ Phi_r)  
            new_Sigma_hat_rp = (Phi_r.T @ const2 @ Phi_p) + (Phi_r.T @ const3 @ Phi_p) 
            new_Sigma_hat_pr = (Phi_p.T @ const2 @ Phi_r) + (Phi_p.T @ const3 @ Phi_r) 
            new_Sigma_hat_p  = (Phi_p.T @ const2 @ Phi_p) + (Phi_p.T @ const3 @ Phi_p)
            new_Sigma_a      = np.hstack([new_Sigma_hat_r, new_Sigma_hat_rp])
            new_Sigma_b      = np.hstack([new_Sigma_hat_pr, new_Sigma_hat_p])
            new_Sigma        = np.vstack([new_Sigma_a, new_Sigma_b])
            
            # compute corrected weights
            pi_pdf_eval    = log_pi_pdf(new_tilde_theta) 
            gbias_fun_eval = log_gbias_fun(new_tilde_theta, new_mu, new_Sigma)
            new_wIS_j      = np.exp(pi_pdf_eval - gbias_fun_eval)
            new_w_til_j    = (new_wIS_j * f_smooth_jp1).reshape((N,1))
            new_W_til      = sum(new_w_til_j)
        else:
            new_w_til_j, new_W_til = w_til_j, W_til

        # closed-form update of reference parameters along the FIS directions 
        mu_hat_r    = ((new_w_til_j.T @ tilde_theta_r.T) / new_W_til).flatten()
        num_r       = np.multiply(np.sqrt(new_w_til_j), (tilde_theta_r.T - mu_hat_r))
        Sigma_hat_r = (num_r.T @ num_r) / new_W_til + (1e-6*np.eye(r)) # adding a bit for stability

        # next level
        j += 1
        
    # smoothing parameters and level ranks
    s_param = s_param[1:j+1]
    ranks   = ranks[:j]

    # calculation of the probability of failure (importance sampling)
    if (refin == 0):
        weight_Indf = Indf * wIS_j
        Pf_hat      = np.mean(weight_Indf)
        Pf_var_hat  = np.var(weight_Indf)/(N-1)
        cv_hat      = np.sqrt(Pf_var_hat)/Pf_hat
        out_cost    = np.hstack([g_call, grad_call, cv_hat])
    else:
        Pf_hat, g_call_extra, cv_hat = refin_final(N, dim, r, g_LSF, geval, Indf, wIS_j, \
            log_pi_pdf, pi_rnd, log_gbias_fun, gbias_rnd, mu_hat_r, Sigma_hat_r, Phi_r, Phi_p)
        out_cost = np.hstack([g_call+g_call_extra, grad_call, np.array(cv_hat)])
        
    return Pf_hat, s_param, samples, LSF_eval, ranks, out_cost



"""
============================================================================
===========================final refinnement step===========================
============================================================================
"""
def refin_final(N, dim, r, g_LSF, geval, Indf, wIS_j, log_pi_pdf, pi_rnd, log_gbias_fun, gbias_rnd, mu_hat_r, Sigma_hat_r, Phi_r, Phi_p):
    #
    NN        = 100      # increase samples every NN
    delta_tar = 0.07     # desired cv of the Pf estimate (how 'good' your final estimate will be)
    maxit     = 200      # max iterations
    #
    k, Nold = 0, N
    cv_hat  = list()    
    #
    while True:
        weight_Indf = Indf * wIS_j
        Pf_hat      = np.mean(weight_Indf) 
        Pf_var_hat  = np.var(weight_Indf)/(N-1)
        cv_hat.append(np.sqrt(Pf_var_hat)/Pf_hat)
        #
        if (k >= 10):
            mu_cv_hat = np.mean(cv_hat[k-10:])
            if mu_cv_hat <= delta_tar or k > maxit:
                break   
            
        # draw extra samples
        theta_new_r = gbias_rnd(NN, mu_hat_r, Sigma_hat_r)
        theta_new_p = pi_rnd(NN, dim-r)

        # evaluate components of the weigths in r
        pi_pdf_eval    = log_pi_pdf(theta_new_r) 
        gbias_fun_eval = log_gbias_fun(theta_new_r, mu_hat_r, Sigma_hat_r)
        wIS_j          = np.hstack((wIS_j, np.exp(pi_pdf_eval - gbias_fun_eval) ))

        # full theta 
        theta_r = Phi_r @ theta_new_r
        theta_p = Phi_p @ theta_new_p
        theta   = theta_r + theta_p

        # evaluate limit state function
        geval_new = g_LSF(theta)
        geval     = np.hstack(( geval, geval_new ))
        Indf      = (geval <= 0).astype(int)    # indicator function
        N        += NN
        k        += 1
    g_call_extra = (N-Nold)
    #
    return Pf_hat, g_call_extra, cv_hat
