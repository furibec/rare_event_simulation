# Rare event simulation algorithms
This repository contains python implementations of four simulation methods used in reliability analysis/rare event simulation:
1. iCE: improved cross-entropy method (single Gaussian and Gaussian mixture biasing densities)
2. SIS: sequential importance sampling (adaptive pCN algorithm as MCMC)
3. SuS: subset simulation (adaptive pCN algorithm as MCMC)
4. iCEred: improved cross-entropy method with failure-informed dimension reduction (single Gaussian). The paper is currently on review; a pre-print can be found in https://arxiv.org/pdf/2006.05496.pdf

For the methods 1,2,3, the target example is a 1D diffusion equation. The conductivity parameter is a log-normal random field which is represented with the KL expansion. The flux is also random and modeled as a Gaussian random variable.
* main_example.py is the running file
* ODE.py defines the problem and solves the diffusion equation
* eigenpairs_solvers.py implements the Nyström method for the solution of the KL eigenvalue problem, and also the analytical solution for the exponential kernel (e.g. Matérn with \nu=0.5)

For the method 4, there are 2 basic examples that are used in the original manuscript.

Any suggestions, corrections or improvements are kindly accepted :-)
