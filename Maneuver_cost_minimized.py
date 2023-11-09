import numpy as np
from No_Drift_Centered_Conditions import no_drift_centered
from Impulsive_Maneuver import Impulsive_Maneuver
import math


def Maneuver_cost_minimized(psi,c2,d2,phi2,Orbit1_ic_shifted,omega1,omega2,delta_t):
    #---Same as function from no_drift_centered, calculates states from geometric property---
    #Orbit2_ic=no_drift_centered(c2,d2,psi,phi2,omega2,tf2,t_step)
    xdot0=omega2*c2*math.cos(psi)
    x0=c2*math.sin(psi)
    y0=2*c2*math.cos(psi)
    ydot0=-2*omega2*c2*math.sin(psi)
    z0=d2*math.cos(phi2)
    zdot0=omega2*d2*math.sin(phi2)
    Orbit2_ic=np.array([[x0],[y0],[z0],[xdot0],[ydot0],[zdot0]])

    ##Impulsive_manuever function

    r0=np.array([Orbit1_ic_shifted[0],Orbit1_ic_shifted[1],Orbit1_ic_shifted[2]])
    v0=np.array([Orbit1_ic_shifted[3],Orbit1_ic_shifted[4],Orbit1_ic_shifted[5]])

    rf=np.array([Orbit2_ic[0],Orbit2_ic[1],Orbit2_ic[2]])
    vf=np.array([Orbit2_ic[3],Orbit2_ic[4],Orbit2_ic[5]])
    
    #---Sets up deltav calculations
    #Goes from orbit 1 to orbit 2 
    #---Transformation matrices setup---
    RR=np.array([[4-3*math.cos(omega1*delta_t), 0, 0],
                 [6*(math.sin(omega1*delta_t)-omega1*delta_t), 1 , 0],
                 [0, 0, math.cos(omega1*delta_t)]])
    RV=np.array([[(1/omega1)*math.sin(omega1*delta_t),(2/omega1)*(1-math.cos(omega1*delta_t)),0],
                 [(2/omega1)*(math.cos(omega1*delta_t)-1), (1/omega1)*(4*math.sin(omega1*delta_t)-3*omega1*delta_t), 0],
                 [0, 0, (1/omega1)*math.sin(omega1*delta_t)]])
    VR=np.array([[(3*omega1)*math.sin(omega1*delta_t),0,0],
                 [(6*omega1)*(math.cos(omega1*delta_t)-1), 0, 0],
                 [0, 0, -omega1*math.sin(omega1*delta_t)]])
    VV=np.array([[math.cos(omega1*delta_t),2*math.sin(omega1*delta_t),0],
                 [-2*math.sin(omega1*delta_t), 4*math.cos(omega1*delta_t)-3, 0],
                 [0, 0, math.cos(omega1*delta_t)]])
    
    v_desired=np.matmul(np.linalg.inv(RV),(rf-np.matmul(RR,r0))) #returns inv(RV)*(rf-RR*r0)
    #print(v0)
    #print(v_desired)
    delta_v1=v_desired-v0
    #print(delta_v1) this is good
    vf_transfer=np.matmul(VR,r0)+np.matmul(VV,v_desired) #returns VR*r0+VV*v_desired
    
    #v_T=v0+delta_v1  #Not used, used previously for plotting purpose
    #print(v_T)
    delta_v2=vf-vf_transfer
    #print(delta_v2)
    #print(delta_v1+delta_v2)
    #delta_v=Impulsive_Maneuver(Orbit1_ic_shifted, Orbit2_ic,tf2,t_step,omega1,delta_t)
    return np.linalg.norm(delta_v1)+np.linalg.norm(delta_v2)





# #---Test Case---
# d1=0
# d2=0
# phi1=0
# phi2=0
# psi1=0
# #psi= -2.982e-03
# psi=-2.982e-03
# c1=400
# c2=200

# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# omega1=(mu/a**3)**(1/2)
# omega2=(mu/a**3)**(1/2)
# n=1
# tf1=n*(2*math.pi)/omega1
# tf2=n*(2*math.pi)/omega2
# t_step=10
# delta_t=3000
# Orbit1_ic_shifted=np.array([[0.00000000e+00],
# [8.60000000e+02],
#  [0.00000000e+00],
#  [4.51048612e-01],
#  [0.00000000e+00],
#  [0.00000000e+00]])
# Orbit2_ic=np.array([[ 0.00000000e+00],
#  [ 4.00000000e+02],
#  [ 0.00000000e+00],
#  [ 2.25524306e-01],
#  [-0.00000000e+00],
#  [ 0.00000000e+00]])
# n=20
# test=Maneuver_cost_minimized(psi,c2,d2,phi2,Orbit1_ic_shifted,omega1,omega2,delta_t)
# print(test)

#Should return 0.5467398743880627


##Used for degugging, value not matching up for some reason
# Orbit2_ic_temp=no_drift_centered(c2,d2,psi,phi2,omega2)
# #print(Orbit2_ic_temp)
#         #Orbit2_propagated=hill_eq(Orbit2_ic,omega2,tf2,t_step) #Not necessary, but need to double check is this is useful
#             #Probably not since Impulsive-manuever plots the graph as well
# delta_v_tab=Impulsive_Maneuver(Orbit1_ic_shifted, Orbit2_ic_temp,tf2,t_step,omega1,delta_t)
# #print(delta_v_tab)