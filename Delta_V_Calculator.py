import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered

def delta_v(r0,rf,v0,t,w):
    #r0 is the initial postion of the spacecraft (3x1)
    #rf is the desired final position of the spacecraft (3x1)
    #v0 is the intial velocity of the spacecraft (3x1)
    #t is the time desired to get from r0 to rf
    #w is the mean orbital motion
    #v_desired is the necessary velocity to get from r0 to rf (3x1)
    #vf is the final velocity given by the transfer orbit (IC being r0 and v_desired)
        #use vf to calculate the 2nd burn needed to stay on an orbit
    #Returns v_desired, delta_v(vector), vf
    RR=np.array([[4-3*math.cos(w*t), 0, 0],[6*(math.sin(w*t)-w*t), 1 , 0],[0, 0, math.cos(w*t)]])
    #print(RR.shape) ##DEBUGGING
    RV=np.array([[(1/w)*math.sin(w*t),(2/w)*(1-math.cos(w*t)),0],[(2/w)*(math.cos(w*t)-1), (1/w)*(4*math.sin(w*t)-3*w*t), 0],[0, 0, (1/w)*math.sin(w*t)]])
    #print(RV.shape) ##DEBUGGING
    #Need matmul to do matrix multiplication
    v_desired=np.matmul(np.linalg.inv(RV),(rf-np.matmul(RR,r0))) #returns inv(RV)*(rf-RR*r0)
    #print(v_desired) #DEBUGGING
    delta_v=v_desired-v0
    #print(delta_v) #DEBUGGING
    
    VR=np.array([[(3*w)*math.sin(w*t),0,0],[(6*w)*(math.cos(w*t)-1), 0, 0],[0, 0, -w*math.sin(w*t)]])
    VV=np.array([[math.cos(w*t),2*math.sin(w*t),0],[-2*math.sin(w*t), 4*math.cos(w*t)-3, 0],[0, 0, math.cos(w*t)]])

    vf=np.matmul(VR,r0)+np.matmul(VV,v_desired)
    #print(vf) #DEBUGGING

    #can also return v_desired and vf
    
    return delta_v,vf


#r0 associated with 
# C=80 #km
# D=20 #km
# psi0= 90
# phi0= 90
a = 6793.137 #km
mu = 398600.5 #km^3/s^2
#w = (mu/a**3)**(1/2)
t=5000
#2*math.pi*w
r0=np.array([[-7.15197331e+01],[ 7.16917786e+01],[ 8.96147232e+00]])
v0=np.array([[ 4.04205966e-02],[ 1.61294382e-01],[-2.01617977e-02]])


#rf and vf associated with
# C=100 #km
# D=30 #km
# psi0= 90
# phi0= 90
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
w = (mu/a**3)**(1/2)

rf=np.array([[-8.93996664e+01],[ 8.96147232e+01],[ 1.34422085e+01]])
x,y=delta_v(r0,rf,v0,t,w)
print(x,y)

#this is different from vf desired! Still need a second burn to get this velocity, 
#first burn only gets you into the right position (think of Hohman transfer, need two deltaV)
#to find the velocity of the craft at rf, can use bottom half of transformation matrix
#vf=np.array([[ 5.05257457e-02],[ 2.01617977e-01], [-3.02426966e-02]])


