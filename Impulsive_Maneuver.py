import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered
from Delta_V_Calculator import delta_v
from Hill_Eq import hill_eq

def Impulsive_Maneuver(State_i, State_f,tf,t_step,w,deltat):
    #Using no_drift_centered, give two relative motion orbits
    #The first orbit's IC will be tabulated in State_i and the IC to stay on the second orbit will be State_f
    #State=[x0 y0 z0 vx0 vy0 vz0]
    #t_final is the time desired to get from the intial orbit to the intial position of the final orbit
    #t_step is the increment of time
    #w is the mean orbital motion
    #deltat is the desired time to for transfer

    #Unpacks intial and final states. Used for delta_v Calcs
    r0=np.array([State_i[0],State_i[1],State_i[2]])
    v0=np.array([State_i[3],State_i[4],State_i[5]])

    rf=np.array([State_f[0],State_f[1],State_f[2]])
    vf=np.array([State_f[3],State_f[4],State_f[5]])
    

    Trajectory_I=hill_eq(State_i,w,tf,t_step)

    #Returns the required intial velocity, delta v to get there, and final velocity once at orbit
    
    delta_v1,vf_T=delta_v(r0,rf,v0,deltat,w) 
    #delta_v1[0]=delta v needed to get to transfer orbit
    #detla_v1[0]=final velocity if transfer orbit
    v_T=v0+delta_v1
    #Form new state for transfer trajectory
    State_T=np.array([State_i[0],State_i[1],State_i[2],v_T[0],v_T[1],v_T[2]])

    #Need to change deltat--> deltat+t_step 
    Trajectory_T=hill_eq(State_T,w,deltat,t_step) 
    #Trajectory_T=hill_eq(State_T,w,deltat+t_step,t_step) 

    Trajectory_F=hill_eq(State_f,w,tf,t_step)

    delta_v2=vf-vf_T




    
    #Create plots
    plt.figure(1)
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    ax.scatter(State_i[0], State_i[1], State_i[2])
    ax.scatter(State_f[0], State_f[1], State_f[2])
    ax.plot3D(Trajectory_I[:,0], Trajectory_I[:,1], Trajectory_I[:,2], label='Intial Trajectory')
    ax.plot3D(Trajectory_T[:,0], Trajectory_T[:,1], Trajectory_T[:,2], label='Transfer Trajectory')
    ax.plot3D(Trajectory_F[:,0], Trajectory_F[:,1], Trajectory_F[:,2], label='Final Trajectory')
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

    return np.linalg.norm(delta_v1+delta_v2)





# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# w = (mu/a**3)**(1/2)
# n=1 #number of orbits 
# tf=n*(2*math.pi)/w 
# t_step=10
# detlat=3600
#Conditions associated with
# C=80 #km
# D=20 #km
# psi0= 90
# phi0= 90
#State_i=np.array([[-7.15197331e+01],[ 7.16917786e+01],[ 8.96147232e+00], [4.04205966e-02],[ 1.61294382e-01],[-2.01617977e-02]])


#Condtitions associated with
# C=100 #km
# D=30 #km
# psi0= 90
# phi0= 90
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# w = (mu/a**3)**(1/2)
#State_f=np.array([[-8.93996664e+01],[ 8.96147232e+01],[ 1.34422085e+01],[ 5.05257457e-02],[ 2.01617977e-01], [-3.02426966e-02]])

#Conditions assocaited with
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
##tate_f=np.array([[ 0.00000000e+00],[ 4.00000000e+02],[ 1.41421356e+01],[ 2.25524306e-01],[-0.00000000e+00],[ 1.59469766e-02]])



#deltat=5600
#deltat=5570
#deltat=5500
# deltat=3000
# test=Impulsive_Maneuver(State_i, State_f,tf,t_step,w,deltat)

