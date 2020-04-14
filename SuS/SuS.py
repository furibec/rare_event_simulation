# ============================================================================
# Subset simulation with adaptive pCN algorithm
# ============================================================================
# Created by:
# Felipe Uribe @ TUM, DTU
# =============================================================================
# The methods work on the standard Gaussian space, transform your input random
# variables accordingly
# ============================================================================
# Based on:
# 1."Estimation of small failure probabilities in high dimentions by SubSim"
#    Siu-Kui Au & James L. Beck.
#    Probabilistic Engineering Mechanics 16 (2001) 263-277.
# 2. "MCMC algorithms for subset simulation"
#    Papaioannou et al.
#    Probabilistic Engineering Mechanics 41 (2015) 83-103.
# =============================================================================
# Version 2020-04
# ============================================================================
import numpy as np



"""
==============================================
==========subset simulation function==========
==============================================
"""
def SuS(d, N, p0, g_fun, g_fun_single, se, maxit=30):
    np.random.seed(seed=se)
    
    # ==============================================
    if (N*p0 != np.fix(N*p0)) or (1/p0 != np.fix(1/p0)):
        raise RuntimeError('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly')
    
    # initialization of variables and storage
    beta     = 0.6                    # initial value for MCMC scaling
    samplesU = list()                 # store samples
    LSF_eval = np.empty((maxit,N))    # space for the sorted LSF evaluations
    prob     = np.empty(maxit)        # space for the failure probability at each level
    tau      = np.zeros(maxit)        # space for the intermediate levels

    # ==============================================
    # initial MCS stage
    u_j   = np.random.normal(size=(d,N))     # samples in the standard Gaussian space
    geval = g_fun(u_j)

    # ==============================================
    # SuS steps
    j      = 0  
    tau[j] = np.inf
    #
    while (tau[j] > 0) and (j <= maxit):
        # next level
        j += 1  

        # sort values in ascending total
        idx             = np.argsort(geval)
        LSF_eval[j-1,:] = geval[idx]   # store LSF values
        
        # total the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU.append(u_j_sort)   # store ordered samples

        # intermediate level 
        tau[j] = np.percentile(geval, p0*100)
        
        # number of failure points in the next level
        nF = int( sum(geval <= max(tau[j],0)) )

        # assign conditional probability to the level
        if (tau[j] <= 0):
            tau[j]    = 0
            prob[j-1] = nF/N
        else:
            prob[j-1] = p0
        print('-Threshold intermediate level ', j-1, ' = ', tau[j])

        # select seeds and randomize the ordering
        seeds       = u_j_sort[:,:nF]
        geval_seeds = LSF_eval[j-1,:nF]
        #
        idx_rnd         = np.random.permutation(nF)
        rnd_seeds       = seeds[:,idx_rnd]            # non-ordered seeds
        rnd_geval_seeds = geval_seeds[idx_rnd]
        
        # sampling process using adaptive conditional sampling
        u_j, geval, beta, accrate = apCN(N, beta, tau[j], rnd_seeds, rnd_geval_seeds, g_fun_single)
        print('\t*apCN beta =', beta, '\t*apCN accrate =', accrate)
    
    # ==============================================  
    LSF_eval[j,:] = geval   # store final LSF values
    samplesU.append(u_j)    # store final failure samples

    # discard unused storage
    prob     = prob[:j]
    tau      = tau[:j+1]
    LSF_eval = LSF_eval[:j+1,:]

    # probability of failure estimate
    Pf_SuS = np.prod(prob)

    return Pf_SuS, tau, samplesU, LSF_eval



"""
==========================================================
==========adaptive preconditioned Crank-Nicolson==========
============aka adaptive conditional sampling=============
==========================================================
"""
def apCN(N, beta_old, tau, theta_seeds, geval_seeds, g_LSF):
    d, Ns = theta_seeds.shape   # dimension and number of seeds
    
    # number of samples per chain
    Nchain = np.ones(Ns, dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    theta_chain = np.zeros((d,N))           # generated samples 
    geval       = np.zeros(N)               # store LSF evaluations
    acc         = np.zeros(N, dtype=int)    # store acceptance

    # adaptation
    Na      = int(np.ceil(100*Ns/N))             # number of chains after the proposal is adapted 
    mu_acc  = np.zeros(int(np.floor(Ns/Na)+1))   # store mean acceptance
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
    # 2. iteration
    star_acc = 0.44       # optimal acceptance rate, see Ref. 2
    lambd[0] = beta_old   # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i         = 0                                          # index for adaptation of lambda
    beta      = np.minimum(lambd[i]*sigma_0, np.ones(d))   # Ref. 2 Eq. 23
    mu_acc[i] = 0 

    # b. apply conditional sampling
    for k in range(Ns):
        idx   = sum(Nchain[:k])   # beginning of each chain total index
        count = idx               # re-start the counter to the seed sample index

        # initial chain values
        theta_chain[:,idx] = theta_seeds[:,k]

        # initial LSF evaluation
        geval[idx] = geval_seeds[k]   #g_LSF(theta_seeds[:,k])  

        # generate chain
        for t in range(Nchain[k]-1):
            # current state
            theta_t = theta_chain[:,count]
                
            # generate candidate sample using pCN proposal
            xi         = np.random.normal(size=d)
            theta_star = np.sqrt(1-beta**2)*theta_t + beta*xi

            # check the location by system analysis      
            geval_star = g_LSF(theta_star)
            if geval_star <= tau:
                theta_chain[:,count+1] = theta_star      # accept the candidate in failure region            
                geval[count+1]         = geval_star      # store the lsf evaluation
                acc[count+1]           = 1               # note the acceptance
            else:
                theta_chain[:,count+1] = theta_t         # reject the candidate and use the same state
                geval[count+1]         = geval[count]    # store the lsf evaluation    
                acc[count+1]           = 0               # note the rejection
            count += 1

        # average of the accepted samples for each seed: do not consider the seed 
        mu_acc[i] += min(1, np.mean( acc[idx+1:idx+Nchain[k]] ))

        # adapt beta parameter
        if ((k+1) % Na == 0):
            if Nchain[k] > 1:
                # c. evaluate average acceptance rate
                hat_acc[i] = mu_acc[i]/Na   # Ref. 2 Eq. 25
            
                # d. compute new scaling parameter
                zeta       = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd[i+1] = np.exp(np.log(lambd[i]) + zeta*(hat_acc[i]-star_acc))  # Ref. 2 Eq. 26
            
                # update parameters
                beta = np.minimum(lambd[i+1]*sigma_0, np.ones(d))  # Ref. 2 Eq. 23
            
                # update counter
                i += 1
                
    # next level lambda
    beta_new = beta[0]

    # compute mean acceptance rate of all chains
    accrate = np.mean(hat_acc)
        
    return theta_chain, geval, beta_new, accrate