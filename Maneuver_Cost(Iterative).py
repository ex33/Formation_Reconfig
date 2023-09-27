import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered
from Impulsive_Maneuver import Impulsive_Maneuver
from No_Drift_Centered_Conditions import no_drift_centered
from Hill_Eq import hill_eq

def Maneuver_Cost_Iterative(Orbit1,Orbit2, t_step,delta_t,n):
    """Given two orbit's geometric properties, will calculate the delta_v cost
    to get from Orbit 1 to Orbit 2, where the position of orbit1 will stay constant 
    but the final position on orbit 2 will vary. The minimum delta_v will be found,
    thus returning the 'Optimal' position for the rendevouz.

    See docString for no_drift_centered for information on each parameter of Orbit1/2
    
    :param Orbit1: Contains the geometric properties of orbit 1 (C,D,psi,phi,omega,tf)
    :type Orbit1: np.array (6,), or (6,1) matrix, list of len 6 should also work
    :param Orbit2: Contains the geometric properties of orbit 1 (C,D,psi,phi,omega,tf)
    :type Orbit2: np.array (6,), (6,1) matrix, list of len 6 should also work
    :param t_step: Represents the time step from 0->tf
    :type t_step: float (s)
    :param delta_t: the amount of time desired to get from r0 to rf
    :type delta_t: float (s)


    :return min_cost_value: returns value of minimum cost
    :rtype: float 

    :return min_cost_index: returns index of minimum cost
    :rtype: integer
    """


    'Need to get geometric parameters of orbit so psi can be varied'
    #Orbit=np.array([C ,D ,psi ,phi ,omega ,tf])
    #---Unpack parameters of orbit 1 and 2---
    c1=Orbit1[0]
    d1=Orbit1[1]
    psi1=Orbit1[2]
    phi1=Orbit1[3]
    omega1=Orbit1[4]
    tf1=Orbit1[5]

    c2=Orbit2[0]
    d2=Orbit2[1]
    psi2=Orbit2[2]
    phi2=Orbit2[3]
    omega2=Orbit2[4]
    tf2=Orbit2[5]

    #---Gets the intial states of the two orbit---
    Orbit1_ic=no_drift_centered(c1,d1,psi1,phi1,omega1,tf1,t_step)
    Orbit2_ic=no_drift_centered(c2,d2,psi2,phi2,omega2,tf2,t_step)

    #---Propagate both orbits---
    Orbit1_propagated=hill_eq(Orbit1_ic,omega1,tf1,t_step)
   
    
    #---Shift Orbit 1 over by X in the 6 axis---
    Orbit1_propagated[1,:]=Orbit1_propagated[1,:]+60

    Orbit1_ic_shifted=np.reshape(np.array([Orbit1_propagated[:,0]]),(6,1))
    
    #print(np.shape(Orbit1_ic_shifted)) #Degugging

    
    #---Calculate iterations---
    psi_step=math.pi/n
    psi_span=np.arange(0,math.pi,psi_step)

    #---Initialize vector for delta_v values---
    delta_v_tab=np.empty([1,n])

    #finds minimum value (look into Scipy optimzation libraary)
   
    #---For loop to iterate psi---
    #Calculates intial state from geometric properties of orbit 2. This then gets
    #into the impulsive_maneuver function to calculate the delta v.
    for i in range(n):
        Orbit2_ic=no_drift_centered(c2,d2,psi_span[i],phi2,omega2,tf2,t_step)
        #Orbit2_propagated=hill_eq(Orbit2_ic,omega2,tf2,t_step) #Not necessary, but need to double check is this is useful
            #Probably not since Impulsive-manuever plots the graph as well
        delta_v_tab[0,i]=Impulsive_Maneuver(Orbit1_ic_shifted, Orbit2_ic,tf2,t_step,omega1,delta_t)
    
    #---Find and return minimum cost and its index---
    #Can make it so that it returns the geometric conditions/state/plots the minimum cost trajectory
    min_cost_value=np.min(delta_v_tab)
    min_cost_index=np.argmin(delta_v_tab)
    return min_cost_value, min_cost_index


#Conditions for planar orbit
#---Test Case---
D1=0
D2=0

phi1=0
phi2=0

psi1=0
psi2=0

C1=400
C2=200

a = 6793.137 #km
mu = 398600.5 #km^3/s^2

omega1=(mu/a**3)**(1/2)
omega2=(mu/a**3)**(1/2)

n=1
tf1=n*(2*math.pi)/omega1
tf2=n*(2*math.pi)/omega2
t_step=10
delta_t=3000

Orbit1=np.array([C1, D1, psi1, phi1, omega1,tf1])
Orbit2=np.array([C2, D2, psi2, phi2, omega2,tf2])
n=20
print(np.shape(Orbit1))

test=Maneuver_Cost_Iterative(Orbit1,Orbit2, t_step,delta_t,n)

print(test)