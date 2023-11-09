#Group Lasso
from Impulsive_Maneuver import Impulsive_Maneuver
from No_Drift_Centered_Conditions import no_drift_centered
import numpy as np
import math
from Hill_Eq import hill_eq
import matplotlib.pyplot as plt
from Delta_V_Calculator import delta_v
#Simulate test 
#Subroutine 1: Add delta_v to state
#Given a time span, discretize entire range. Assume every range is a potential manuever.
#At an obersvation, all manuevers adds up to the final position
#After observation, no more manuvers will contribue

def manuver(State,delta_v):
    State_temp=np.reshape(State,(6,1))
    return np.array([State_temp[0],State_temp[1],State_temp[2],State_temp[3]+delta_v[0],State_temp[4]+delta_v[1],State_temp[5]+delta_v[2]])

def calc_RV(delta_t):
    RV=np.array([[(1/omega)*math.sin(omega*delta_t),(2/omega)*(1-math.cos(omega*delta_t)),0],
                 [(2/omega)*(math.cos(omega*delta_t)-1), (1/omega)*(4*math.sin(omega*delta_t)-3*omega*delta_t), 0],
                 [0, 0, (1/omega)*math.sin(omega*delta_t)]])
    return RV
def calc_RR(delta_t):
    RR=np.array([[4-3*math.cos(omega*delta_t), 0, 0],
                 [6*(math.sin(omega*delta_t)-omega*delta_t), 1 , 0],
                 [0, 0, math.cos(omega*delta_t)]])
    return RR


def form_A_matrix(tau,t):
    #Tau are manuvers
    #t are observations 
    N=len(tau)
    M=len(t)
    A=[]
    for i in range(M):
        row=[]
        row.append(calc_RR(t[i]))
        row.append(calc_RV(t[i]))
        for j in range(N):
            if t[i]>tau[j]: #if t[i]=t[j], its zero anyways so not t>=tau
                #print(t[i]-tau[j])
                row.append(calc_RV(t[i]-tau[j]))
            else:
                row.append(np.zeros([3,3]))
        if i==0:
            A=np.block(row)
        else:
            A=np.vstack((A, np.block(row)))
       # B=np.concatenate((B,np.block(row)),axis=0)
    return A

def form_B_vector(tau,t,)
            



tau=np.empty([140])
t=np.empty([14])
for i in range(140):
    tau[i]=10*(i+1)
for j in range(14):
    t[j]=100*(j+1)

A=form_A_matrix(tau,t)


# C=100
# D=50
# psi=0
# phi=0
a = 6793.137 #km
mu = 398600.5 #km^3/s^2
omega = (mu/a**3)**(1/2)
# tau=np.zeros([4]) 
# #Step 1: Obtain first state
# State1=no_drift_centered(C,D,psi,phi,omega)


# #Step 2: propagate state for 3 time steps
# t_step=100
# tf=3*t_step
# #tf=0.5*(2*math.pi)/omega
# #t_step=10
# delta_t=2500
# Trajectory1=hill_eq(State1,omega,tf,t_step) #Should be length of 4(Including original state)
# t=np.arange(0,1400,100)
# delta_v=np.array([[1],[1],[1]]) #1m/s in each direction
# #Extracts final state from Trajectory and adds delta_v1

# tau[0]=tf
# State2=manuver(Trajectory1[:,3],delta_v)
# Trajectory2=hill_eq(State2,omega,tf,t_step)

# tau[1]=tf*2-t_step
# State3=manuver(Trajectory2[:,3],delta_v)
# Trajectory3=hill_eq(State3,omega,tf,t_step)

# tau[2]=tf*3-2*t_step
# State4=manuver(Trajectory3[:,3],delta_v)
# Trajectory4=hill_eq(State4,omega,tf,t_step)

# tau[3]=tf*4-3*t_step
# State5=manuver(Trajectory4[:,3],delta_v)
# Trajectory5=hill_eq(State5,omega,tf,t_step)


# Trajectory=np.concatenate((Trajectory1[:,0:3], Trajectory2[:,0:3],Trajectory3[:,0:3],Trajectory4[:,0:3],Trajectory5[:,0:3]),axis=1)
# #B=np.reshape(State1[0:3],(3,1))

# #Form B vector, should contain all position vectors of trajectory at each time step
# B=np.empty([0,1])
# for i in range(15):
#     B=np.concatenate((B,np.reshape(Trajectory[0:3,i],(3,1))),axis=0)

# #Form A Vector
# A=np.empty([0,1])
# for i in range(15):
#     A=np.concatenate()




# #---3-D plots---
# ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
# ax.scatter(State1[0], State1[1], State1[2])
# ax.scatter(State2[0], State2[1], State2[2])
# ax.scatter(State3[0], State3[1], State3[2])
# ax.scatter(State4[0], State4[1], State4[2])
# ax.scatter(State5[0], State5[1], State5[2])
# ax.plot3D(Trajectory[0,:], Trajectory[1,:], Trajectory[2,:], label='Trajectory')
# # ax.plot3D(Trajectory[0,0:4], Trajectory[1,0:4], Trajectory[2,0:4], label='Trajectory12')
# # ax.plot3D(Trajectory[0,3:7], Trajectory[1,3:7], Trajectory[2,3:7], label='Trajectory23')
# # ax.plot3D(Trajectory[0,6:10], Trajectory[1,6:10], Trajectory[2,6:10], label='Trajectory34')
# # ax.plot3D(Trajectory[0,9:13], Trajectory[1,9:13], Trajectory[2,9:13], label='Trajectory45')
# # ax.plot3D(Trajectory[0,12:15], Trajectory[1,12:15], Trajectory[2,12:15], label='Trajectory5')
# ax.set_xlabel('X', fontsize=10)
# ax.set_ylabel('Y', fontsize=10)
# ax.set_zlabel('Z', fontsize=10)
# ax.view_init(elev=20., azim=45) #Attempt to make axis more visable