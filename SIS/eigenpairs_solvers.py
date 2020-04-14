# =============================================================================
# Created by:
# Felipe Uribe @ MIT, TUM
# =============================================================================
# contains functions to solve 1D KL eigenvalue problems
# =============================================================================
# References:
# 1."Numerical recipes: the art of scientific computing". Section 19.1
#    Press et al. (2007). 3rd edition. Cambrigde university press.
# 2."Stochastic finite elements: a spectral approach"
#    R. Ghanem and P.D. Spanos. Rev edition (2012). Dover publications.
# =============================================================================
# Version 2019-11
# =============================================================================
import numpy as np
import scipy as sp
from scipy.optimize import fsolve



#=============================================================================
#============================Nystrom method===================================
#=============================================================================
def Nystrom(xnod, C_nu, sigma, M, N_GL):    
    # domain data    
    n = xnod.size
    a = (xnod[-1] - xnod[0])/2     # scale
    
    # compute the Gauss-Legendre abscissas and weights
    xi, w = np.polynomial.legendre.leggauss(N_GL)

    # transform nodes and weights to [0, L]
    xi_s = a*xi + a
    w_s  = a*w
    
    # compute diagonal matrix 
    w_s_sqrt = np.sqrt(w_s)
    D_sqrt   = sp.sparse.spdiags(w_s_sqrt, 0, N_GL, N_GL).toarray()
    S1       = np.tile(w_s_sqrt.reshape(1, N_GL), (N_GL, 1))
    S2       = np.tile(w_s_sqrt.reshape(N_GL, 1), (1, N_GL))
    S        = S1 * S2
    
    # compute covariance matrix 
    Sigma_nu = np.zeros((N_GL, N_GL))
    for i in range(N_GL):
        for j in range(N_GL):
            if i != j:
                Sigma_nu[i,j] = C_nu(xi_s[i], xi_s[j])
            else:
                Sigma_nu[i,j] = sigma**2   # diagonal term
                  
    # solve the eigenvalue problem
    A    = Sigma_nu * S                             # D_sqrt*Sigma_nu*D_sqrt
    L, h = np.linalg.eig(A)                         # solve eigenvalue problem
    # L, h = sp.sparse.linalg.eigsh(A, M, which='LM')   # np.linalg.eig(A)         
    idx  = np.argsort(-np.real(L))                  # index sorting descending
    
    # order the results
    eigval = np.real(L[idx])
    h      = np.real(h[:,idx])
    
    # take the M values
    eigval = eigval[:M]
    h      = h[:,:M]
    
    # replace for the actual eigenvectors
    phi = np.linalg.solve(D_sqrt, h)
    
    # Nystrom's interpolation formula
    # recompute covariance matrix on partition nodes and quadrature nodes
    Sigma_nu = np.zeros((n, N_GL))
    for i in range(n):
        for j in range(N_GL):
            Sigma_nu[i,j] = C_nu(xnod[i], xi_s[j])
    #        
    M1     = Sigma_nu * np.tile(w_s.reshape(N_GL, 1), (1, n)).T
    M2     = phi @ np.diag(1/eigval) 
    eigvec = M1 @ M2            
    #
    return eigval, eigvec



#=============================================================================
#================analytical solution for exponential kernel===================
#=============================================================================
def analytical_exp_kernel(xx, M, l_c):    
    # domain data
    a  = (xx[-1]-xx[0])/2
    T  = (xx[-1]+xx[0])/2   # shift parameter
    xx = xx - T
    
    # definitions
    c     = 1.0/l_c
    fun_o = lambda ww: c  - ww*np.tan(ww*a)
    fun_e = lambda ww: ww + c *np.tan(ww*a)
    
    # constants for indexing the point of search
    j, k   = 0, 0
    nn     = int(np.ceil(M/2))+1
    wn     = np.zeros((M,1))
    eigfun = {}   # as a dictionnary
    eigvec = np.zeros((len(xx),M))
    #
    for i in range(nn+1):
        # odd: compute data associated with equation : c - w*tan(a*w) = 0
        if (i > 0 and 2*i-1 <= M):
            k       = k+1            
            n       = 2*i-1
            w0      = (k-1)*(np.pi/a)+0.01
            wn[n-1] = abs(fsolve(fun_o, w0))
            alpha   = np.sqrt(a + (np.sin(2*wn[n-1]*a)/(2*wn[n-1])))
            #
            eigfun[n-1]   = lambda x: np.cos(wn[n-1]*x)/alpha
            eigvec[:,n-1] = eigfun[n-1](xx)#np.cos(wn[n-1]*xx)/alpha
            
        # even: compute data associated with equation : w + c*tan(a*w)
        if ((2*i+2) <= M):
            j       = j+1
            n       = 2*i+2
            w0      = (j-0.5)*(np.pi/a)+0.01
            wn[n-1] = abs(fsolve(fun_e, w0))
            alpha   = np.sqrt(a - (np.sin(2*wn[n-1]*a)/(2*wn[n-1])))
            #
            eigfun[n-1]   = lambda x: np.sin(wn[n-1]*x)/alpha
            eigvec[:,n-1] = eigfun[n-1](xx)#np.sin(wn[n-1]*xx)/alpha
    #
    eigval = (2*c)/(wn**2 + c**2)  
    eigval = eigval.ravel()    
    #
    return eigval, eigvec, eigfun