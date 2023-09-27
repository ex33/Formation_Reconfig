import numpy as np
import matplotlib.pyplot as plt
import math

def hill_eq(state,omega,tf,t_step):
    """
    Given the intial conditions, angular rate, and final time and step size,
    tabulates position and velocity according to the closed form hill equations 
    and plots this on a 3d axis.
    See below for source:
    https://en.wikipedia.org/wiki/Clohessy%E2%80%93Wiltshire_equations

    OPTIONAL: THERE ARE 2D PLOTS USEFUL FOR VISUALIZATION BUT ARE COMMENTED OUT 
    TO REDUCE CLUTTER WHEN THIS FUNCTION IS CALLED ON.

    :param state: 6x1 Matrix in the form of ([x0],[y0],[z0],[vx0],[vy0],[vz0])
    :type state: np.array (6x1)
    :param omega: Represents the angular rate, Equation is sqrt(a^3/mu)
    :type omega: float (rad/s)
    :param tf: Represents the final time to propagate the orbit (typically n*Period) 
    :type tf: float (s)
    :param t_step: Represents the time step from 0->tf
    :type t_step: float (s)

    :return state_new : This is a 6 x len(t) contains information of propagated states
    :rtype state_new: np.array (6 x len(t))
    """
    
    #---Unpacking State---
    x=state[0]
    y=state[1]
    z=state[2]
    vx=state[3]
    vy=state[4]
    vz=state[5]
    
    
    #---Form time span---
    t=np.arange(0,tf,t_step)

    #---Perform elementwise operation (No for loop needed)---
    x_t=(vx/omega)*np.sin(omega*t)-(3*x+2*vy/omega)*np.cos(omega*t)+(4*x+2*vy/omega)
    y_t=(6*x+4*vy/omega)*np.sin(omega*t)+(2*vx/omega)*np.cos(omega*t)-(6*omega*x+3*vy)*t+(y-2*vx/omega)
    z_t=z*np.cos(omega*t)+(vz/omega)*np.sin(omega*t)
    vx_t=vx*np.cos(omega*t)+(3*omega*x+2*vy)*np.sin(omega*t)
    vy_t=(6*omega*x+4*vy)*np.cos(omega*t)-2*vx*np.sin(omega*t)-(6*omega*x+3*vy)
    vz_t=-z*omega*np.sin(omega*t)+vz*np.cos(omega*t)

    #---Forms Array containing information of state over time span---
    state_new=np.zeros([6,len(t)])
    state_new[0,:]=x_t
    state_new[1,:]=y_t
    state_new[2,:]=z_t
    state_new[3,:]=vx_t
    state_new[4,:]=vy_t
    state_new[5,:]=vz_t

    #---All plotting functions---

    #---Plots position(x y z) vs time on same graph--
    # plt.figure(1)
    # plt.plot(t_span, x_t,'-', label='x' ) #args are x-axis, y-axis, symbol/color, label
    # plt.plot(t_span, y_t,'-', label='y' ) #args are x-axis, y-axis, symbol/color, label
    # plt.plot(t_span, z_t,'-', label='z' ) #args are x-axis, y-axis, symbol/color, label
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.show()

    #---Plots Intract (Y) vs Radial(X)---
    # plt.figure (2)
    # plt.plot(y_t,x_t, '-')
    # plt.xlabel('Intract Axis (Y)')
    # plt.ylabel('Radial Axis (X)')
    # plt.title('Intract/Radial Plane')

    #---Plots Inract (Y) and Cross-tract (Z)
    # plt.figure (3)
    # plt.plot(y_t,z_t, '-')
    # plt.xlabel('Intract Axis (Y)')
    # plt.ylabel('Crosstract Axis (Z)')
    # plt.title('Intract/Crosstract Plane')

    #---Plots Intract Cross Tract (Z) vs Radial (X)
    # plt.figure (4)
    # plt.plot(z_t,x_t, '-')
    # plt.xlabel('Crosstract (Z)')
    # plt.ylabel('Radial Axis (X)')
    # plt.title('Crosstract/ Radial Plane')

    #---3-D plots---
    # ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    # ax.scatter(x, y, z)
    # ax.plot3D(x_t, y_t, z_t, label='Trajectory')
    # ax.set_xlabel('X', fontsize=10)
    # ax.set_ylabel('Y', fontsize=10)
    # ax.set_zlabel('Z', fontsize=10)
    # ax.view_init(elev=20., azim=45) #Attempt to make axis more visable
    

    return state_new
    







##----Testing Function----

# x0 = -0.066538073651029 #km
# y0= 0.186268907590665 #km
# z0 = 0.000003725378152 #km
# dx0= -0.000052436200437 #km/s
# dy0 = 0.000154811363681 #km/s
# dz0 = 0.000210975508077 #km/s
# q0=np.array([[-2.95314044e+00] ,[6.77870988e+04] ,[9.86615711e+01] , [3.82190960e+01],
#   [5.90628088e+00] , [2.25520817e-02]])

# #Constants
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# n = math.sqrt(mu/a**3)


# #print(x0.shape)
# #tf=365*24*60*60 #number of seconds in a year
# tf=24*60*60
# t_step=60 #time step is an hour, propagate every hour
# state=hill_eq(q0,n,tf,t_step)

# print(state[:,1])
