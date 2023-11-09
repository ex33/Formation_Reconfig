#Quadratic Constraint Optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from Hill_Eq import hill_eq
from numpy.linalg import eig

def Quadratic_constraint(C,omega,delta_t):
    """Using the quadratic constraint on the size of orbit
    Given the constraint in the form of C^2=xfT*A*xf,
    and xf=Bu+Dxi, where xi is assumed to just be zero vector. This is similar to
    xf=delta_t*v+xi in the one d kinematic case. The B matrix is made up of the state
    transformation matrices which encodes this relationship of the velocity to finial position 
    contributions in a given delta t. 

    Constraint for u, which is what we are after will then be:
    1=u^T*D*u_T
    Define D as B^T*(A/C^2)*B
    

    :param C: _description_
    :type C: _type_
    :param omega: _description_
    :type omega: _type_
    :param delta_t: _description_
    :type delta_t: _type_

    :return: deltav1
    :rtype: float
    :return: deltav2 
    :rtype: float
    """
    RV=np.array([[(1/omega)*math.sin(omega*delta_t),(2/omega)*(1-math.cos(omega*delta_t)),0],
                 [(2/omega)*(math.cos(omega*delta_t)-1), (1/omega)*(4*math.sin(omega*delta_t)-3*omega*delta_t), 0],
                 [0, 0, (1/omega)*math.sin(omega*delta_t)]])

    VV=np.array([[math.cos(omega*delta_t),2*math.sin(omega*delta_t),0],
                 [-2*math.sin(omega*delta_t), 4*math.cos(omega*delta_t)-3, 0],
                 [0, 0, math.cos(omega*delta_t)]])
             
    A=np.array([[9, 0, 0, 0, 6/omega, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1/(omega*omega), 0, 0],
                [6/omega, 0, 0, 0, 4/(omega*omega), 0],
                [0, 0, 0, 0, 0, 0]])

    B=np.array([[RV[0,0],RV[0,1],RV[0,2],0,0,0],
                [RV[1,0],RV[1,1],RV[1,2],0,0,0],
                [RV[2,0],RV[2,1],RV[2,2],0,0,0],
                [VV[0,0],VV[0,1],VV[0,2],1,0,0],
                [VV[1,0],VV[1,1],VV[1,2],0,1,0],
                [VV[2,0],VV[2,1],VV[2,2],0,0,1]])
    
    B_T=np.transpose(B)
 
    #Can replace this with the closed solution via matlab symbolic toolbox so no calcs
    #are needed to find D, but that will be an extremely messy matrix
    D=np.matmul(B_T,np.matmul(A/(C*C),B)) #(B_T * A * B)/C^2
    #print(D)
    #Now can optimize using 1=uT*D*u and minimize uTu by solving eigan value problem:
    #lambda*u=A*u, where due to constraint: uTu=1/lambda, so max lambda returns min uT*u

    eig_val, eig_vec=eig(D)
    
    max_eig=np.max(eig_val)
    max_index=np.argmax(eig_val) #Gives the row that the eigven vector belongs to
    
    #print(eig_vec)
    u=math.sqrt(1/max_eig)*eig_vec[:,max_index]
    #print(u)

    x=np.matmul(B,u)
    #bc X=Du, now that we have U, can find x!!!

    #check=np.matmul(np.transpose(u),np.matmul(D,u)) This sound be equal to 1 

    return np.reshape(u,(6,1)), np.reshape(x,(6,1))


a = 6793.137 #km
mu = 398600.5 #km^3/s^2
omega = (mu/(a*a*a))**(1/2)
delta_t=1000


C=500
u,x=Quadratic_constraint(C,omega,delta_t)
State_f=np.reshape(x,(6,1))
t_step=10
t_span=np.arange(0,delta_t,t_step)

y=x[1]
z=x[2]
xdot=x[3]
ydot=x[4]
zdot=x[5]


#Check:
State_i=np.array([[0],[0],[0],u[0],u[1],u[2]])

Trajectory=hill_eq(State_i,omega,delta_t,t_step)
x_t=Trajectory[0,:]
y_t=Trajectory[1,:]
z_t=Trajectory[2,:]

ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.scatter(x_t[0], y_t[0], z_t[0])
ax.plot3D(x_t, y_t, z_t, label='Trajectory')
ax.scatter(x[0], x[1], x[2])
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Z', fontsize=10)
ax.view_init(elev=20., azim=45) #Attempt to make axis more visable
    

C_check=math.sqrt((3*x[0]+2*x[4]/omega)**2+(xdot/omega)**2)
#C_check2=(9*(x[0]*x[0])+(12/omega)*x[0]*ydot+4*ydot**2/omega**2+(xdot*xdot/(omega*omega)))

#Impulsive_Maneuver(State_i, State_f,tf,t_step,omega,delta_t)




    
