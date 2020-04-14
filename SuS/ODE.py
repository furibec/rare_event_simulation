# =============================================================================
# 1D diffusion equation solver and random field settings
# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM
# =============================================================================
# Version 2019-11
# =============================================================================
import numpy as np
import scipy as sp
import scipy.stats as sps
import eigenpairs_solvers as eigenpairs



#============================================================================
#========================diffusion equation parameters=======================
#============================================================================
# domain discretization
nn     = 501
x0, x1 = 0, 1                           # start and end
x      = np.linspace(x0, x1, nn)        # the points of evaluation of solution
h      = x[1]-x[0]                      # partition

# source term 
Ns      = 4
xs      = np.array([0.2, 0.4, 0.6, 0.8])   # source locations
ws      = 0.8                              # source weights/strength
sigma_s = 0.05                             # source standard width
s       = np.zeros(nn)
for i in range(Ns):
    s += ws * sps.norm.pdf(x, loc=xs[i], scale=sigma_s)
s_int = h*np.cumsum(s, dtype=float)        # integrated source

# boundary conditions
# 0. Dirichlet datum
u_r = 1                    # deterministic at x=1

# 1. Neumann flux
mu_F    = -1
sigma_F = np.sqrt(0.2)
T       = lambda u: mu_F + sigma_F*u   # transformation from standard Gaussian u



#============================================================================
#=====================conductivity random field parameters===================
#============================================================================
# Gaussian field
mu_Y    = 1                # mean of the underlying Gaussian
sigma_Y = np.sqrt(0.3)     # std of the underlying Gaussian
l_c     = 0.1              # correlation length [m]

# eigenvalue problem solution
dim_max = 203             # 99% of the RF variability
method  = 0
if (method == 0):
    eigval, eigvec, _ = eigenpairs.analytical_exp_kernel(x, dim_max, l_c)
    eigval            = (sigma_Y**2)*eigval
else:
    N_GP = int(2*dim_max)
    C_nu = lambda x1,x2: (sigma_Y**2) * np.exp( -(abs(x1-x2)/l_c) )  # cov kernel
    #
    eigval, eigvec = eigenpairs.Nystrom(x, C_nu, sigma_Y, dim_max, N_GP)
#
Phi = eigvec @ np.diag(np.sqrt(eigval))



#============================================================================
#======================Solve diffusion ODE "analytically"====================
#============================================================================
def analytical(theta):    
    dim_KL = len(theta)-1

    # RV
    F        = T(theta[0])
    theta_KL = theta[1:]

    # RF
    Y     = mu_Y + Phi[:,:dim_KL] @ theta_KL   # KL expansion
    kappa = np.exp(Y)                          # lognormal RF
    
    # integration constants
    c1 = -F
    c2 = u_r - np.trapz( (c1 - s_int)/kappa, x )

    # "analytical" solution
    u = h*np.cumsum( (c1 - s_int)/kappa, dtype=float ) + c2

    return u