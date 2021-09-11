import numpy as np
from model import parameters as param
from model import Hel0
from numpy.random import random
import math


def monte_carlo(param, steps = 3000, dR = 0.5):
    R = np.zeros((param.ndof,param.nb))
    ndof, nb = R.shape
    βn = param.beta/nb 

    #Monte carlo loop
    for i in range(steps):
        rDof  = np.random.choice(range(ndof))
        rBead = np.random.choice(range(nb))

        # Energy before Move
        En0 = ringPolymer(R[rDof,:], param) + Hel0(R[rDof,rBead]) 

         # update a bead -------------
        dR0 = dR * (random() - 0.5)
        R[rDof, rBead] += dR0
        #----------------------------
        # Energy after Move
        En1 = ringPolymer(R[rDof,:], param) + Hel0(R[rDof,rBead]) 

        # Probality of MC
        Pr = np.min([1.0,np.exp(-βn * (En1-En0))])
        # a Random number
        r = random()
        # Accepted
        if r < Pr:
            pass
        # Rejected
        else:
            R[rDof, rBead] -= dR0
    return R


def ringPolymer(R,param):
    """
    Compute Ringpolymer Energy
    E = ∑ 0.5  (m nb^2/β^2)  (Ri-Ri+1)^2
    """
    nb = param.nb
    βn = (param.beta)/nb 
    Ω  = (1 / βn)  
    M  = param.M
    E = 0
    for k in range(-1,nb-1):
        E+= 0.5 * M * Ω**2 * (R[k] - R[k+1])**2
    return E
        
def initP(param):
    nb, ndof = param.nb , param.ndof
    sigp = (param.M * param.nb/param.beta)**0.5
    return np.random.normal(size = (ndof, nb )) * sigp


def force(R,param):
    """ 
        Nuclear Force term : F = -dV/dR for each bead
    """

    nb = param.nb

    F =  np.zeros((R.shape)) # ndof nbead

    #for k in range(nb):
    for m in range(len(F)):
        F[m] = -(R[m] + (3.0/10.0)*R[m]**2 + (4.0/100.0)*R[m]**3)

    #print(F)
    return F

def nm_t(P,R,param):

    nb = param.nb
    ndof = param.ndof
    lb_n = param.lb_n
    ub_n = param.ub_n
    
    cmat = np.zeros((nb,nb))
    pibyn = math.acos(-1.0)/nb


    P_norm = np.zeros((ndof,nb)) #normal modes for momenta
    Q_norm = np.zeros((ndof,nb)) #normal modes for position

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            if l==0:
                cmat[j,l] = 1.0
            elif l >= lb_n and l<0:
                cmat[j,l] = np.sqrt(2.0)*np.sin(2.0*pibyn*(j+1)*l)
            elif l > 0 and l <= ub_n:
                cmat[j,l] = np.sqrt(2.0)*np.cos(2.0*pibyn*(j+1)*l)



    pnew = np.zeros((ndof,nb))
    qnew = np.zeros((ndof,nb))

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            for m in range(ndof):
                pnew[m,l]+= P[m,j]*cmat[j,l]
                qnew[m,l]+= R[m,j]*cmat[j,l]


    P_norm = pnew/nb
    Q_norm = qnew/nb

    return P_norm, Q_norm


def back_nm_t(P_norm,Q_norm,param):

    nb = param.nb
    ndof = param.ndof
    lb_n = param.lb_n
    ub_n = param.ub_n

    cmat = np.zeros((nb,nb))
    pibyn = math.acos(-1.0)/nb

    P = np.zeros((ndof,nb)) # bead representation momenta
    R = np.zeros((ndof,nb)) # bead representation position


    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            if l==0:
                cmat[j,l] = 1.0
            elif l >= lb_n and l<0:
                cmat[j,l] = np.sqrt(2.0)*np.sin(2.0*pibyn*(j+1)*l)
            elif l > 0 and l <= ub_n:
                cmat[j,l] = np.sqrt(2.0)*np.cos(2.0*pibyn*(j+1)*l)

    pnew = np.zeros((ndof,nb))
    qnew = np.zeros((ndof,nb))    

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            for m in range(ndof):
                pnew[m,j]+= P_norm[m,l]*cmat[j,l]
                qnew[m,j]+= Q_norm[m,l]*cmat[j,l]

    P = pnew
    R = qnew

    return P,R
    
def ring(param):

    nb = param.nb
    ndof = param.ndof
    dt = param.dtN
    M = param.M
    beta = param.beta
    lb_n = param.lb_n
    ub_n = param.ub_n

    poly = np.zeros((4,nb))
    
    #Monodromy matrix for free ring-polymer update

    betan = beta/nb
    twown = 2.0/(betan)
    pibyn = math.acos(-1.0)/nb

    for i in range(nb):
        l=(i-int(nb/2))

        if l==0:
            poly[0,0] = 1.0
            poly[1,0] = 0.0
            poly[2,0] = dt/M
            poly[3,0] = 1.0
            
        elif l >= lb_n and l<0:
            poly[0,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            poly[1,l]=-twown*np.sin(l*pibyn)*M*np.sin(twown*np.sin(l*pibyn)*dt)
            poly[2,l]=np.sin(twown*np.sin(l*pibyn)*dt)/(twown*np.sin(l*pibyn)*M)
            poly[3,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            
        elif l > 0 and l <= ub_n:
            poly[0,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            poly[1,l]=-twown*np.sin(l*pibyn)*M*np.sin(twown*np.sin(l*pibyn)*dt)
            poly[2,l]=np.sin(twown*np.sin(l*pibyn)*dt)/(twown*np.sin(l*pibyn)*M)
            poly[3,l]=np.cos(twown*np.sin(l*pibyn)*dt)
   
    

    return poly

def freerp(P,R,param):

    nb = param.nb
    ndof = param.ndof

    P_norm = np.zeros((ndof,nb)) #normal modes for momenta
    Q_norm = np.zeros((ndof,nb)) #normal modes for position
    poly = np.zeros((4,nb))

    poly = ring(param)

    #print("------ before nm_t---------------")
    #print(P[0,0],P[0,1],P[0,2])
    #print(R[0,0],R[0,1],R[0,2])


    P_norm, Q_norm = nm_t(P,R,param) #normal mode obtained

    for k in range(nb):
        for j in range(ndof):
            l=(k-int(nb/2))
            
            pjknew = P_norm[j,l]*poly[0,l] + Q_norm[j,l]*poly[1,l]
            Q_norm[j,l] = P_norm[j,l]*poly[2,l] + Q_norm[j,l]*poly[3,l]
            P_norm[j,l] = pjknew



    P,R = back_nm_t(P_norm,Q_norm,param) # from normal mode to bead



    #print("------ after back_transform---------------")
    #print(P[0,0],P[0,1],P[0,2])
    #print(R[0,0],R[0,1],R[0,2])

    return P,R
    

def run_traj(P,R,param):

    nb = param.nb 
    M = param.M
    dt = param.dtN
    dt2 = 0.5*dt
    #------------------------------
    # ---- velocity verlet algo-----
    for ib in range(nb):
        F = force(R[:,ib],param)
        # propagate P for half step (dt/2)
        # P(t+dt) = P(t) + (dP/dt) * dt 
        P[:,ib] += (1.0/M)*F*dt2
        
    # evolution of ring-polymer
    P,R = freerp(P,R,param) 
       
    for ib in range(nb):
         F = force(R[:,ib],param)
         
         # propagate P for half step (dt/2)
         # P(t+dt) = P(t) + (dP/dt) * dt 
         P[:,ib] += (1.0/M)*F*dt2
 
    return P,R
#----------trajectory loops--------------------

if __name__ == "__main__" :
    
    ndof = param.ndof
    nb = param.nb
    NTraj  = param.NTraj
    nstate = param.nstate
    NSteps = param.NSteps
    dt=param.dtN

    c_xx = np.zeros(NSteps)

    #f = open("R_traj.txt", "w+")
    
    for itraj in range(NTraj):

        R = monte_carlo(param) # initialize R, P
        P = initP(param)

       
       # f.write(f"{itraj} {' '.join(R[0,0:nb].astype(str))} \n")
        


        R0 = np.zeros(ndof)

        R0 = np.sum(R[:,:],axis=1)/nb # taking bead avg position at t=0

        for isteps in range(NSteps):
            
            P,R = run_traj(P,R,param)
            
            #------ calculating correlation functuion------
            Rt = np.zeros(ndof)
            Rt =  np.sum(R[:,:],axis=1)/nb  #taking bead avg positon at finite t
            
            c_xx[isteps] += R0[0]*Rt[0]
    

    c_xx = c_xx/NTraj
    
    #f.close()  

f = open("corr.txt", "w+")
for isteps in range(NSteps):
    f.write(f"{isteps*dt} {c_xx[isteps]} \n")
f.close()  

#----------- adiabatic Ring-Polymer Molecular Dynamics code-------------------------
#------------ (calulaion of correlation function)-----------------------------------
#-------- see [Ian R. Craig and David E. Manolopoulos, JCP, 121, 3368 (2004)]-------







    















#------------ definition of initMap........
#def initMap(param):
"""
initialize Mapping variables
"""
#    q = np.zeros((param.nstate,param.nb))
#    p = np.zeros((param.nstate,param.nb))
#    i0 = param.initState
#    for i in range(param.nstate):
#       for ib in range(param.nb):
#           #l = ib+1
#           #print("ib=",ib,"l=",l)
#            η = np.sqrt(1 + 2*(i==i0))
#            θ = random() * 2 * np.pi
#            q[i,ib] = η * np.cos(θ) 
#            p[i,ib] = η * np.sin(θ) 
#    return q, p
#------------------------------------------------------------

#def pop(q,p,param):
#    nb = param.nb
#    nstate = param.nstate
#    rho = np.zeros((param.nstate,param.nstate))
    
#    for ib in range(nb):
        
#        rho += 0.5*(np.outer(q[:,ib],q[:,ib])+np.outer(p[:,ib],p[:,ib])-np.identity(len(p[:,ib])))
        
#    return rho/nb
#---------------------------------------------------------------------------
    
    
    #q, p = initMap(param)    # initialize q, p

    #    print("q=",q)
    #    print("p=",p)
        
    #    rho_final = pop(q,p,param)
        
     #   print("--------------")
     #   print(rho_final)
     #   print("--------------")

    


