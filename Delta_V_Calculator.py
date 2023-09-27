import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered

def delta_v(r0,rf,v0,delta_t,omega):
    """
    Using the transformation matrix form of the hill equations, can find the total delta_v
    needed to get from intial state (r0,v0) onto a transfer trajectory to get to some final position (rf)
    that takes delta_t time as well as the final velocity at the end of the orbit. Calculates the desired
    velocity (v_desired) that is necessary to go from r0 to rf, then subtracts intial velocity from it to get delta_v.
    Final velocity is found by now assuming delta_v has been applied and thus, intial velocity is v_desired, in which 
    the respective transformation matrices can be applied to find the final velocity.

    :param r0: the intial position of the spacecraft
    :type r0: np.array (3x1) (m)
    :param rf: the desired final position of the spacecraft (typically on another orbit/trajectory)
    :type rf: np.array (3x1) (m)
    :param v0: the intial velocity of the spacecraft (3x1)
    :type v0: np.array (3x1) (m/s)
    :param delta_t: the amount of time desired to get from r0 to rf
    :type delta_t: float (s)
    :param omega: Represents the angular rate, Equation is sqrt(a^3/mu)
    :type omega: float (rad/s)
    
    :return delta_v: The amount of delta_v needed to make transfer (can extract mag. and direction)
    :rtype delta_v: np.array (3x1) (m/s)
    :return vf: Final velocity at end of transfer, useful for other function/calculations
    :rtype vf: np.array (3x1) (m/s)
   
    """
    #---Transformation matrices setup---
    RR=np.array([[4-3*math.cos(omega*delta_t), 0, 0],
                 [6*(math.sin(omega*delta_t)-omega*delta_t), 1 , 0],
                 [0, 0, math.cos(omega*delta_t)]])
    RV=np.array([[(1/omega)*math.sin(omega*delta_t),(2/omega)*(1-math.cos(omega*delta_t)),0],
                 [(2/omega)*(math.cos(omega*delta_t)-1), (1/omega)*(4*math.sin(omega*delta_t)-3*omega*delta_t), 0],
                 [0, 0, (1/omega)*math.sin(omega*delta_t)]])
    VR=np.array([[(3*omega)*math.sin(omega*delta_t),0,0],
                 [(6*omega)*(math.cos(omega*delta_t)-1), 0, 0],
                 [0, 0, -omega*math.sin(omega*delta_t)]])
    VV=np.array([[math.cos(omega*delta_t),2*math.sin(omega*delta_t),0],
                 [-2*math.sin(omega*delta_t), 4*math.cos(omega*delta_t)-3, 0],
                 [0, 0, math.cos(omega*delta_t)]])
    
    #---Calculations---
    v_desired=np.matmul(np.linalg.inv(RV),(rf-np.matmul(RR,r0))) #returns inv(RV)*(rf-RR*r0)
    delta_v=v_desired-v0
    vf=np.matmul(VR,r0)+np.matmul(VV,v_desired) #returns VR*r0+VV*v_desired
    
    return delta_v,vf


##----Testing Function----


# #r0 associated with 
# # C=80 #km
# # D=20 #km
# # psi0= 90
# # phi0= 90
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# #w = (mu/a**3)**(1/2)
# t=5000
# #2*math.pi*w
# r0=np.array([[-7.15197331e+01],[ 7.16917786e+01],[ 8.96147232e+00]])
# v0=np.array([[ 4.04205966e-02],[ 1.61294382e-01],[-2.01617977e-02]])


# #rf and vf associated with
# # C=100 #km
# # D=30 #km
# # psi0= 90
# # phi0= 90
# # a = 6793.137 #km
# # mu = 398600.5 #km^3/s^2
# w = (mu/a**3)**(1/2)

# rf=np.array([[-8.93996664e+01],[ 8.96147232e+01],[ 1.34422085e+01]])
# x,y=delta_v(r0,rf,v0,t,w)
# print(x,y)

# #this is different from vf desired! Still need a second burn to get this velocity, 
# #first burn only gets you into the right position (think of Hohman transfer, need two deltaV)
# #to find the velocity of the craft at rf, can use bottom half of transformation matrix
# #vf=np.array([[ 5.05257457e-02],[ 2.01617977e-01], [-3.02426966e-02]])


