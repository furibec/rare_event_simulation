# =============================================================================
# 1D diffusion equation: example in section 3.3.2 of myphdthesis
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM, DTU
# =============================================================================
# Version 2020-04
# =============================================================================
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
#
import ODE
import CE_IS_smooth

#============================================================================
# numerical model: defined in ODE.py
dim   = 5+1    # RF plus flux: <= 203+1, otherwise increase dim_max in ODE.py
u_ODE = lambda theta: ODE.analytical(theta) 

# LSF
u_thres      = 2.7    # maximum allowed pressure head
g_LSF_single = lambda theta: u_thres - max(u_ODE(theta)) 

#============================================================================
# choose smooth indicator function
f = 'erf'
if (f == 'erf'):
    I_smooth = lambda s, geval: 0.5 * (1 + sp.special.erf(-geval/(np.sqrt(2)*s)))#sp.stats.norm.cdf(-geval/s)
elif (f == 'tanh'):
    I_smooth = lambda s, geval: 0.5 * (1 + np.tanh(-geval/s))
 
#============================================================================
# iCE method
NSIM       = int(50)     # number of independent simulations
N          = int(1e3)    # total number of samples for each level
delta_star = 1.5         # target cv of the weights
k_mix      = 2

# LSF for all samples
g_LSF = lambda theta: np.array([ g_LSF_single(theta[:,i]) for i in range(N) ])

#============================================================================
np.random.seed(seed=1)
set_seed = np.random.randint(0, 2 ** 31 - 1, (NSIM, 1))
Pf       = np.empty((NSIM))
#
print('\n\n==========iCE==========')
for i in range(NSIM):  
    print('\n***Simulation ', i+1, '/', NSIM, '\n')
    Pf[i], s_param, samples, LSF_eval = CE_IS_smooth.SG(dim, N, delta_star, g_LSF, I_smooth, set_seed[i])
    # Pf[i], s_param, samples, LSF_eval = CE_IS_smooth.GM(dim, N, delta_star, k_mix, g_LSF, I_smooth, set_seed[i])
#
Pf_mean = np.mean(Pf)
Pf_std  = np.std(Pf)
Pf_cov  = Pf_std / Pf_mean
print('\n***** Average probability of failure is:', Pf_mean, 'cv=', Pf_cov, '*****\n')

#============================================================================
plt.figure()
plt.semilogy(Pf, 'r*')
plt.semilogy(Pf_mean*np.ones(NSIM), 'r-')
plt.xlabel('sims')
plt.ylabel('Pf')
plt.tight_layout()  
plt.show()