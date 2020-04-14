# =============================================================================
# 1D diffusion equation: example in section 3.3.2 of myphdthesis
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM, DTU
# =============================================================================
# Version 2020-04
# =============================================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#
import ODE
from SuS import SuS

#============================================================================
# numerical model: defined in ODE.py
dim   = 50+1    # RF plus flux: <= 203+1, otherwise increase dim_max in ODE.py
u_ODE = lambda theta: ODE.analytical(theta) 

# LSF
u_thres      = 2.7    # maximum allowed pressure head
g_LSF_single = lambda theta: u_thres - max(u_ODE(theta)) 

#============================================================================
# SuS method
NSIM = int(30)     # number of independent simulations
N    = int(1e3)    # total number of samples per level
p0   = 0.1         # prescribed level probability 

# LSF for all samples
g_LSF = lambda theta: np.array([ g_LSF_single(theta[:,i]) for i in range(N) ])

#============================================================================
np.random.seed(seed=1)
set_seed = np.random.randint(0, 2 ** 31 - 1, (NSIM, 1))
Pf       = np.empty((NSIM))
#
print('\n\n==========SuS==========')
for i in range(NSIM):  
    print('\n***Simulation ', i+1, '/', NSIM, '\n')
    Pf[i], tau, samplesU, LSF_eval = SuS(dim, N, p0, g_LSF, g_LSF_single, set_seed[i])
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