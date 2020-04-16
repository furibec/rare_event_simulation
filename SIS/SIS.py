# ============================================================================
# Sequential importance sampling with adaptive pCN algorithm
# ============================================================================
# Created by:
# Felipe Uribe @ TUM, DTU
# ============================================================================
# * The methods work on the standard Gaussian space, transform your input random
# variables accordingly
# * TO DO: improve stability with logsumexp trick as in SMC
# ============================================================================
# Based on:
# 1."Sequential importance sampling for structural reliability analysis"
#    Papaioannou et al.
#    Structural safety 62 (2016) 66-75.
# 2. "MCMC algorithms for subset simulation"
#    Papaioannou et al.
#    Probabilistic Engineering Mechanics 41 (2015) 83-103.
# ============================================================================
# Version 2019-11
# ============================================================================
import numpy as np
import scipy.optimize as spo



"""
============================================================================
======================Sequential importance sampling========================
============================================================================
"""
def SIS(dim, N, burnin, rho, delta_star, I_smooth, g_LSF, g_LSF_single, se, maxit=30):
    np.random.seed(seed=se)

    #==============================================
    # initialization of variables and storage
    beta     = 0.6                    # initial value for MCMC scaling
    Ns       = int(N*rho)             # number of seeds
    samples  = list()                 # store samples
    LSF_eval = np.empty((maxit,N))    # space for the sorted LSF evaluations
    prob     = np.empty(maxit)        # space for the failure probability at each level
    tau      = np.zeros(maxit)        # space for the intermediate levels
    Sn       = np.zeros(maxit)        # space for normalizing factors
    s_param  = np.zeros(maxit+1)      # smoothness parameters    

    #==============================================
    # prior samples
    u_j   = np.random.normal(size=(dim,N))     # samples in the standard Gaussian space 
    geval = g_LSF(u_j)
    
    #==============================================
    # SIS steps
    j          = 0
    s_param[j] = np.inf    
    #
    while True: 
        # storage
        LSF_eval[j,:] = geval
        samples.append( u_j )

        # compute cv of the ratio between the indicator and its smooth approximation
        Indf       = (geval <= 0).astype(int) 
        f_smooth_j = I_smooth(s_param[j], geval)
        #
        ratio = Indf / f_smooth_j
        CoV   = np.std(ratio) / abs(np.mean(ratio))
        if (CoV <= delta_star) or (j >= maxit): 
            break 

        # compute smoothing parameters for distribution fitting
        if (j == 0):
            fmin = lambda s: ( (  np.std(I_smooth(s, geval)) / \
                                 np.mean(I_smooth(s, geval)) ) - delta_star)**2
            s_param[j+1] = spo.fminbound(fmin, 0, 10*np.mean(geval))
            f_smooth_jp1 = I_smooth(s_param[j+1], geval)
            w_j          = f_smooth_jp1
        else:
            fmin = lambda s: ( (  np.std(I_smooth(s, geval)/f_smooth_j) / \
                                 np.mean(I_smooth(s, geval)/f_smooth_j) ) - delta_star)**2
            s_param[j+1] = spo.fminbound(fmin, 0, s_param[j]) 
            f_smooth_jp1 = I_smooth(s_param[j+1], geval)
            w_j          = f_smooth_jp1 / f_smooth_j
        print('Smoothness parameter:', s_param[j+1])

        # compute ratio of normalizing constants and normalized weights
        Sn[j]    = np.mean(w_j)
        w_j_norm = (1/N) * (w_j/Sn[j])
        
        # resampling step
        idx         = np.random.choice(N, Ns, replace=True, p=w_j_norm)
        u_j_seeds   = u_j[:,idx]
        geval_seeds = geval[idx]

        # move the samples: draw N samples from the next intermediate
        u_j, geval, beta, accrate = apCN(N, burnin, beta, u_j_seeds, geval_seeds, g_LSF_single, I_smooth, s_param[j+1])
        print('\t*apCN beta =', beta, '\t*apCN accrate =', accrate) 
        
        # next level
        j += 1
    #
    LSF_eval[j+1,:] = geval   # store final LSF values
    samples.append(u_j)       # store final failure samples

    # discard unused storage
    s_param  = s_param[:j+1]
    LSF_eval = LSF_eval[:j+2,:]

    # calculation of the probability of failure (importance sampling)
    p_hat = np.prod(Sn[:j])
    Pf    = (p_hat/N) * np.sum(ratio) 

    return Pf, s_param, samples, LSF_eval



"""
==========================================================
==========adaptive preconditioned Crank-Nicolson==========
==========================================================
"""
def apCN(N, burnin, beta_old, theta_seeds, geval_seeds, g_LSF, I_smooth, s_param):
    d, Ns = theta_seeds.shape   # dimension and number of seeds
    Ntot  = int(N+burnin)       # total number of generated samples
    
    # number of samples per chain
    Nchain = np.ones(Ns, dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    theta_chain = np.zeros((d, Ntot))         # generated samples 
    geval       = np.zeros(Ntot)              # store lsf evaluations
    acc         = np.zeros(Ntot, dtype=int)   # store acceptance

    # adaptation
    Na      = int(np.ceil(100*Ns/N))             # number of chains after which the proposal is adapted 
    mu_acc  = np.zeros(int(np.floor(Ns/Na)+1))   # store acceptance
    hat_acc = np.zeros(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
    lambd   = np.zeros(int(np.floor(Ns/Na)+1))   # scaling parameter \in (0,1)

    # ==============================================
    # 1. compute the standard deviation
    opc = 'a'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(d)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        sigma_0 = np.std(theta_seeds, axis=1)
    else:
        raise RuntimeError('Choose a or b')

    # ==============================================
    # adaptation parameters    
    count, i = 0, 0
    #
    star_acc = 0.44       # target acceptance rate
    lambd[i] = beta_old   # initial scaling parameter \in (0,1)
    beta     = np.minimum(lambd[i]*sigma_0, np.ones(d)) 
    # 
    for k in range(Ns):
        idx = sum(Nchain[:k])

        # initial state
        theta_t = theta_seeds[:,k]
        geval_t = geval_seeds[k]       # = g_LSF(theta_seeds[:,k])
        feval_t = I_smooth(s_param, geval_t)

        # generate chain
        for t in range(Nchain[k] + burnin):
            if t == burnin:
                count -= burnin
                
            # generate candidate sample using pCN proposal
            xi         = np.random.normal(size=d)
            theta_star = np.sqrt(1-beta**2)*theta_t + beta*xi

            # check the location by system analysis      
            geval_star = g_LSF(theta_star)
            feval_star = I_smooth(s_param, geval_star)

            # acceptance probability
            ratio = feval_star / feval_t
            alpha = min( 1, ratio ) 
            
            # accept/reject
            u = np.random.rand(1)
            if u <= alpha:
                acc[count]           = 1
                theta_chain[:,count] = theta_star       # accept the candidate in failure region            
                geval[count]         = geval_star       # store the lsf evaluation
                #
                theta_t, geval_t, feval_t = theta_star, geval_star, feval_star
            else:
                acc[count]           = 0
                theta_chain[:,count] = theta_t          # reject the candidate and use the same state
                geval[count]         = geval_t          # store the lsf evaluation    
            count += 1

        # average of the accepted samples for each seed: not including burned samples
        mu_acc[i] += min(1, np.mean(acc[idx:idx+Nchain[k]] ))

        # adapt beta parameter
        if (k+1) % Na == 0:
            if Nchain[k] > 1:
                # c. evaluate average acceptance rate
                hat_acc[i] = mu_acc[i]/Na  # Ref. 2 Eq. 25
            
                # d. compute new scaling parameter
                zeta       = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd[i+1] = np.exp(np.log(lambd[i]) + zeta*(hat_acc[i]-star_acc))  
            
                # update parameters
                beta = np.minimum(lambd[i+1]*sigma_0, np.ones(d))
            
                # update counter
                i += 1
    #                   
    theta_chain = theta_chain[:,:N]
    geval       = geval[:N]

    # next level lambda
    beta_new = beta[0]

    # compute mean acceptance rate of all chains
    accrate = np.mean(hat_acc)
        
    return theta_chain, geval, beta_new, accrate