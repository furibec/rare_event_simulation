# rare event simulation algorithms
This repository contains python implementations of three simulation methods used in reliability analysis/rare event simulation:
1. iCE: improved cross-entropy method (single Gaussian and Gaussian mixture biasing densities)
2. SIS: sequential importance sampling (adaptive pCN algorithm as MCMC)
3. SuS: subset simulation (adaptive pCN algorithm as MCMC)

(4. Soon... iCEred: cross-entropy method with failure informed dimension reduction)

The target example is given by a 1D diffusion equation. The conductivity parameter is a log-normal random field which is represented using the KL expansion. The flux is also random and modeled as a Gaussian random variable.

* main_example.py is the running file
* ODE.py defines the problem and solves the diffusion equation
* eigenpairs_solvers.py implements the Nyström method for the solution of the KL eigenvalue problem, and also the analytical solution for the exponential kernel (e.g. Matérn with \nu=0.5)

Any suggestions, corrections or improvements are kindly accepted :-)
