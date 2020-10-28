# =============================================================================
# Linear function of independent standard Gaussian: example in section 5.1
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM
# =============================================================================
# Based on:
# 1."Cross-entropy-based importance sampling with failure-informed dimension
#    reduction for rare event simulation"
#    Uribe et al. 2020 (SIAM)
# =============================================================================
# Version 2019-09
# =============================================================================
import numpy as np
import scipy as sp
import scipy.io as spio
import scipy.stats as sps
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
#
from iCEred import SG_local

# ============================================================================
# numerical model
dim = int(5e2)       # dimension of the inputs (>=2)
beta = 3.5           # threshold

# exact probability of failure
Pf_ex = sps.norm.cdf(-beta)

# ============================================================================
# iCEred method
NSIM = int(5e1)       # number of independent iCEred runs
N = int(5e2)          # number of samples for each level
delta_star = 1.5      # target cv of the weights
epsilon = 0.01        # desired tolerance of the certified approximation
refin = 1             # perform final refinement of Pf (=1) or not (=0)

# LSF and gradient for all samples (x in R^{dim x N})
def g_LSF(x): return beta - np.sum(x, axis=0)/np.sqrt(dim)
def grad_g_LSF(x): return -(1/np.sqrt(dim)) * np.ones((N, dim))

# choose smooth indicator function
f = 'tanh'
if (f == 'erf'):
    def I_smooth(s, geval): return 0.5 * (1 + sp.special.erf(-geval/(np.sqrt(2)*s)))  # sps.norm.cdf(-geval/s)
    def grad_logI_smooth(s, geval, x): return (sps.norm.pdf(-geval/s)/sps.norm.cdf(-geval/s)).reshape((N, 1)) \
                                        * (-grad_g_LSF(x)/s)
elif (f == 'tanh'):
    def I_smooth(s, geval): return 0.5 * (1 + np.tanh(-geval/s))
    def grad_logI_smooth(s, geval, x): return (1 + np.tanh(geval/s)).reshape((N, 1)) * (-grad_g_LSF(x)/s)

# ============================================================================
np.random.seed(seed=1)
set_seed = np.random.randint(0, 2**31 - 1, (NSIM, 1))
#
Pf = np.empty((NSIM))
s_params = list()
u_j_samples_F = list()
cost = list()
rank = list()
#
print('\n\n==========iCEred-based IS==========')
for i in range(NSIM):
    print('\n***Simulation ', i+1, '/', NSIM, '\n')
    Pf[i], s_param, samples, geval, rr, cc = SG_local(dim, N, delta_star, g_LSF, I_smooth,
                                                grad_logI_smooth, refin, epsilon, set_seed[i])
    #
    s_params.append(s_param)
    u_j_samples_F.append(samples[-1])
    cost.append(cc)   # 1st 2 positions are g_call and grad_g_call, the remainder are the cvs
    rank.append(rr)
#
Pf_mean = np.mean(Pf)
Pf_std = np.std(Pf)
Pf_cov = Pf_std / Pf_mean
print('\n***** Average probability of failure iCEred is:', Pf_mean, 'cv=', Pf_cov, '*****')
print('***** Exact probability of failure is:', Pf_ex, '*****\n')

# ============================================================================
plt.figure()
plt.semilogy(Pf, 'r*')
plt.semilogy(Pf_ex*np.ones(NSIM), 'b-')
plt.semilogy(Pf_mean*np.ones(NSIM), 'r--')
plt.xlabel('sims')
plt.ylabel('Pf')
plt.tight_layout()
plt.show()
