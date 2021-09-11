import numpy as np
import random
import math

class parameters():
    NSteps = 3000
    NTraj = 5000
    dtN = 0.01
    beta = 1.0
    M = 1 # mass of the particle
    nstate =2
    #dirName = "result"
    nb = 3
    lb_n = -(nb-1)/2
    ub_n = (nb-1)/2
    ndof = 1
    #fs_to_au = 41.341 # a.u./fs
    #nskip = 1
    
    # MODEL SPECIFIC
    #ε = 5.0
    #ξ = 4.0
    #β  = 1.0
    #ωc = 2.0
    #Δ  = 1.0 # Non-varied parameter

def Hel0(R):
    
    return  0.5*np.sum(R**2.0) + (1.0/10.0)*np.sum(R**3.0) + (1.0/100.0)*np.sum(R**4.0)

#0.5*sum(x2(:)**2)+(1.0/10.0)*sum(x2(:)**3)+(1.0/100.0)*sum(x2(:)**4))

