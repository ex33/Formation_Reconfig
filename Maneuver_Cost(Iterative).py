import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered
from Impulsive_Maneuver import Impulsive_Maneuver
from No_Drift_Centered_Conditions import no_drift_centered


def Maneuver_Cost_Iterative(Orbit1,Orbit2,tf, t_step):
    #Get orbit 1 first, off centered
    #Orbit=np.array([C D psi phi w])
    Orbit1_ic=no_drift_centered(Orbit1[0],Orbit1[1],Orbit1[2],Orbit1[3],Orbit1[4])
    
    #Shifts y over to off center it


    #Get orbit 2 intial conditions
    Orbit2_ic=no_drift_centered(Orbit2[0],Orbit2[1],Orbit2[2],Orbit2[3],Orbit2[4])

    Orbit1_propagated=hill_eq(Orbit1_ic,Orbit1[4],tf,t_step)
    
    #Orbit1_propagated[:,2]=Orbit1_propagated[:,2]-50 #shifts y over by n units
    #Cant just shifft over like this casue velocity is with respect to origin, so
    #so the velocity would be all messed up if not changed along with position
    
    r0=Orbit1_propagated[0,:]
    Orbit2_propagated=hill_eq(Orbit2_ic,Orbit2_ic[4],tf,t_step)

    num_rows=Orbit2_propagated.shape[0]
    #Will go through each position in orbit2_propagated, should go through 50
    #Impuslive maneuver prints out delta v now
    #will go through starting poitn of Orbit 1 to whatever position in loop it current is on, 
    #calculate delta v, put into chart,
    #finds minimum value (look into Scipy optimzation libraary)
    for i in range(0,num_rows+num_rows/50,num_rows/50):
        Impulsive_Maneuver(State_i, Orbit2_propagated[i,0:3],tf,t_step,w,deltat)
        



#Conditions for planar orbit
D=0
psi0=0