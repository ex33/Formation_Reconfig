import numpy as np
import matplotlib.pyplot as plt
import math
from Hill_Eq import hill_eq

def no_drift_centered(C,D,psi0,phi0,omega):
    """
    Given geometric constraints about the orbit, will return the intial conditions
    that will result in an orbit that is centered in the RST frame with no drift once
    propagated using the closed form hill equations. 

    :param C: Defines the 2-1 Ellipse for intract-radial (Y,X) plane (minor axis is C, major axis is 2C)
    :type C: float (m)
    :param D: Defines the minor axis for ellipse on intract-cross tract (Y,Z) plane. Can think of as the Z offset
    :type D: float (m)
    :param psi0: phase shift for intract-radial (Y,X) plane
    :type psi0: float (radians) 0->pi
    :param phi0: phase shift for Z plane oscilaltions
    :type phi0: float (radians) 0->pi
    :param omega: Represents the angular rate, Equation is sqrt(a^3/mu)
    :type omega: float (rad/s)
    :param tf: Represents the final time to propagate the orbit (typically n*Period) 
    :type tf: float (s)
    :param t_step: Represents the time step from 0->tf
    :type t_step: float (s)

    :return State0: Returns intial state in the form of ([x0],[y0],[z0],[vx0],[vy0],[vz0]) to return specified orbit
    :rtype: np.array (6 x 1)
    """
    
    #Given equations for C,D,psi,phi,yc0,xc0, the closed form hill equations can be
    #represented in terms of C D psi phi related to constraints that yc0=xc0=0, yc0_dot=0

    #---Solved hill eq given constraints---
    #xdot0=math.sqrt((C*C*omega*omega)/(math.tan(psi0)*math.tan(psi0)+1))
    xdot0=omega*C*math.cos(psi0)

    #x0=math.tan(psi0)*xdot0/omega #should maybe be sin of psi
    x0=C*math.sin(psi0)
    #y0=2*xdot0/omega 
    y0=2*C*math.cos(psi0)
    #ydot0=-2*omega*x0
    ydot0=-2*omega*C*math.sin(psi0)
    #z0=math.sqrt((D*D)/(math.tan(phi0)*math.tan(phi0)+1))

    z0=D*math.cos(phi0)
    zdot0=omega*D*math.sin(phi0)
    #zdot0=math.tan(phi0)*omega*z0
    

    #---Form state vector---
    State0=np.array([[x0],[y0],[z0],[xdot0],[ydot0],[zdot0]])
    
    
    #plotter=hill_eq(State0,w,tf,t_step)
    

    
    return(State0) #State0 is vector containing intial conditions




##----Testing Function----

# C=200 #km
# D=20 #km
# psi0= 0 #radians
# phi0= math.pi/4 #raidans
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# w = (mu/a**3)**(1/2)
# n=5 #number of orbits 
# tf=n*(2*math.pi)/w 
# t_step=10

# test_case=no_drift_centered(C,D,psi0,phi0,w,tf,t_step)
# print(test_case)
# x0=test_case[0]
# y0=test_case[1]
# z0=test_case[2]
# xdot0=test_case[3]
# ydot0=test_case[4]
# zdot0=test_case[5]

# yc0=y0-2*xdot0/w
# ycdot=-6*w*x0-3*ydot0
# xc=-2*ycdot/(3*w)
# print(abs(yc0)<10e-7)
# print(abs(ycdot)<10e-7)
# print(abs(xc)<10e-7)
# print(yc0)
# print(ycdot)
# print(xc)

    
