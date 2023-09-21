import numpy as np
import matplotlib.pyplot as plt
import math
from Hill_Eq import hill_eq

def no_drift_centered(C,D,psi0,phi0,w,tf,t_step):
    #Given conditions C,D,pshi0,phi0 where
    #C: Defines the 2-1 ellipse(minor axis is C, major axis is 2C) 
    #of the interceptor motion relative to the target
    #D: 
    #psi:
    #phi:
    #w is omega
    #tf is final time
    #t_step is interval of time 
    #Function returns intial conditions of the interceptor (x0 y0 z0 xdot0 ydot0 zdot0)
    #relative to the target within the RSW frame such that yc_dot=0 (no drift) 
    # and yc0=0, ,xc=0 (centered). These intial conditions will then be plotted according
    #to the closed form solutions to verify with time steps 0:tf:t_step

    xdot0=math.sqrt((C*C*w*w)/(math.tan(psi0)*math.tan(psi0)+1))
    x0=math.tan(psi0)*xdot0/w
    y0=2*xdot0/w
    ydot0=-2*w*x0
    z0=math.sqrt((D*D)/(math.tan(phi0)*math.tan(phi0)+1))
    zdot0=math.tan(phi0)*w*z0
    State0=np.array([[x0],[y0],[z0],[xdot0],[ydot0],[zdot0]])
    
    
    #plotter=hill_eq(State0,w,tf,t_step)
    

    
    return(State0) #State0 is vector containing intial conditions


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

    
