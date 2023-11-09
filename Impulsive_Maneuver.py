import numpy as np
import matplotlib.pyplot as plt
import math
from No_Drift_Centered_Conditions import no_drift_centered
from Delta_V_Calculator import delta_v
from Hill_Eq import hill_eq
from plotter3d import plotter3d

def Impulsive_Maneuver(State_i, State_f,tf,t_step,omega,delta_t):
    """
    Given two states (Such as two different intial states from no_drift_centered),
    the delta_v needed to transfer from one to the other will be calculated and
    the intial, transfer, and final trajectories will be plotted.

    :param State_i: Intial state of vehicle. ([[x0],[y0],[z0],[vx0],[vy0],[vz0]])
    :type State_i: np.array (6x1)
    :param State_f: Final DESIRED state of vehicle. ([[xf],[yf],[zf],[vxf],[vyf],[vzf]])
    :type State_f: np.array (6x1)
    :param tf: Represents the final time to propagate the orbit (typically n*Period) 
    :type tf: float (s)
    :param t_step: Represents the time step from 0->tf
    :type t_step: float (s)
    :param omega: Represents the angular rate, Equation is sqrt(a^3/mu)
    :type omega: float (rad/s)
    :param delta_t: the amount of time desired to get from r0 to rf
    :type delta_t: float (s)

    :return: total deltav (normalized)
    :rtype: float
    """


    #---Unpacks intial and final states---
    r0=np.array([State_i[0],State_i[1],State_i[2]])
    v0=np.array([State_i[3],State_i[4],State_i[5]])

    rf=np.array([State_f[0],State_f[1],State_f[2]])
    vf=np.array([State_f[3],State_f[4],State_f[5]])
    
    #---Propagate the intial and final Trajectory---
    'This will be used for plotting'
    Trajectory_I=hill_eq(State_i,omega,tf,t_step)
    Trajectory_F=hill_eq(State_f,omega,tf,t_step)

    #---Calc first delta-v---
    delta_v1,vf_T=delta_v(r0,rf,v0,delta_t,omega) 
    #---Transfer trajectory properties---
    'This puts you on the transfer orbit'
    v_T=v0+delta_v1

    'Form new state for transfer trajectory'
    State_T=np.array([State_i[0],State_i[1],State_i[2],v_T[0],v_T[1],v_T[2]])

    'Propagate transfer trajectory for plotting'
    Trajectory_T=hill_eq(State_T,omega,delta_t,t_step) 
    
    #---Calc second delta-v---
    delta_v2=vf-vf_T


    
    #Create plots
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    ax.scatter(State_i[0], State_i[1], State_i[2])
    ax.scatter(State_f[0], State_f[1], State_f[2])
    ax.plot3D(Trajectory_I[0,:], Trajectory_I[1,:], Trajectory_I[2,:], label='Intial Trajectory')
    ax.plot3D(Trajectory_T[0,:], Trajectory_T[1,:], Trajectory_T[2,:], label='Transfer Trajectory')
    ax.plot3D(Trajectory_F[0,:], Trajectory_F[1,:], Trajectory_F[2,:], label='Final Trajectory')
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)


    #---Work in progress, trying to make a general plotter function---
    #plotter3d(Trajectory_I[0,:], Trajectory_I[1,:], Trajectory_I[2,:])
    #plotter3d(Trajectory_T[0,:], Trajectory_T[1,:], Trajectory_T[2,:])
    #plotter3d(Trajectory_F[0,:], Trajectory_F[1,:], Trajectory_F[2,:])
    

    #---Returns normalized total delta_v---
    "Will be used for cost calculations"
    return np.linalg.norm(delta_v1)+np.linalg.norm(delta_v2)





# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# omega = (mu/a**3)**(1/2)
# n=1 #number of orbits 
# tf=n*(2*math.pi)/omega 
# t_step=10
# detlat=3600
# #Conditions associated with
# C=80 #km
# D=20 #km
# psi0= 90
# phi0= 90
# State_i=np.array([[-7.15197331e+01],[ 7.16917786e+01],[ 8.96147232e+00], [4.04205966e-02],[ 1.61294382e-01],[-2.01617977e-02]])


# #Condtitions associated with
# C=100 #km
# D=30 #km
# psi0= 90
# phi0= 90
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# w = (mu/a**3)**(1/2)
# State_f=np.array([[-8.93996664e+01],[ 8.96147232e+01],[ 1.34422085e+01],[ 5.05257457e-02],[ 2.01617977e-01], [-3.02426966e-02]])

# #Conditions assocaited with
# # C=200 #km
# # D=20 #km
# # psi0= 0 #radians
# # phi0= math.pi/4 #raidans
# # a = 6793.137 #km
# # mu = 398600.5 #km^3/s^2
# #omega = (mu/a**3)**(1/2)
# # n=5 #number of orbits 
# # tf=n*(2*math.pi)/omega
# # t_step=10
# # State_f=np.array([[ 0.00000000e+00],[ 4.00000000e+02],[ 1.41421356e+01],[ 2.25524306e-01],[-0.00000000e+00],[ 1.59469766e-02]])




# #delta_t=5600
# #delta_t=5570
# delta_t=5500

# test=Impulsive_Maneuver(State_i, State_f,tf,t_step,omega,delta_t)
# print(test)

