import numpy as np
import matplotlib.pyplot as plt
import math

def hill_eq(state,omega,tf,t_step):
    #state is a vector containing the intial conditions for: x y z vx vy vz
    x=state[0]
    #print(x)
    y=state[1]
    #print(y)
    z=state[2]
    #print(z)
    vx=state[3]
    #print(vx)
    vy=state[4]
    #print(vy)
    vz=state[5]
    
    
    t_span=np.arange(0,tf,t_step)
    x_t=np.empty(len(t_span),float)
    y_t=np.empty(len(t_span),float)
    z_t=np.empty(len(t_span),float)
    vx_t=np.empty(len(t_span),float)
    vy_t=np.empty(len(t_span),float)
    vz_t=np.empty(len(t_span),float)
    state_new=np.empty((len(t_span),6),float)
    i=0
    for t in t_span:
        x_t[i]=(vx/omega)*math.sin(omega*t)-(3*x+2*vy/omega)*math.cos(omega*t)+(4*x+2*vy/omega)
        y_t[i]=(6*x+4*vy/omega)*math.sin(omega*t)+(2*vx/omega)*math.cos(omega*t)-(6*omega*x+3*vy)*t+(y-2*vx/omega)
        z_t[i]=z*math.cos(omega*t)+(vz/omega)*math.sin(omega*t)
        vx_t[i]=vx*math.cos(omega*t)+(3*omega*x+2*vy)*math.sin(omega*t)
        vy_t[i]=(6*omega*x+4*vy)*math.cos(omega*t)-2*vx*math.sin(omega*t)-(6*omega*x+3*vy)
        vz_t[i]=-z*omega*math.sin(omega*t)+vz*math.cos(omega*t)
        state_new[i,:]=[x_t[i],y_t[i],z_t[i],vx_t[i],vy_t[i],vz_t[i] ]
        i=i+1
    
    
    # plt.figure(1)
    # plt.plot(t_span, x_t,'-', label='x' ) #args are x-axis, y-axis, symbol/color, label
    # plt.plot(t_span, y_t,'-', label='y' ) #args are x-axis, y-axis, symbol/color, label
    # plt.plot(t_span, z_t,'-', label='z' ) #args are x-axis, y-axis, symbol/color, label
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.show()

    # plt.figure (2)
    # plt.plot(y_t,x_t, '-')
    # plt.xlabel('Intract Axis (Y)')
    # plt.ylabel('Radial Axis (X)')
    # plt.title('Intract/Radial Plane')

    # plt.figure (3)
    # plt.plot(y_t,z_t, '-')
    # plt.xlabel('Intract Axis (Y)')
    # plt.ylabel('Crosstract Axis (Z)')
    # plt.title('Intract/Crosstract Plane')

    # plt.figure (4)
    # plt.plot(z_t,x_t, '-')
    # plt.xlabel('Crosstract (Z)')
    # plt.ylabel('Radial Axis (X)')
    # plt.title('Crosstract/ Radial Plane')

    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.plot3D(x_t, y_t, z_t, label='Trajectory')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    
    #ax.view_init(elev=20., azim=45)
    ax.dist = 12
    

    return state_new
    


# x0 = -0.066538073651029 #km
# y0= 0.186268907590665 #km
# z0 = 0.000003725378152 #km
# dx0= -0.000052436200437 #km/s
# dy0 = 0.000154811363681 #km/s
# dz0 = 0.000210975508077 #km/s

# #Test case
# q0=np.array([[-2.95314044e+00] ,[6.77870988e+04] ,[9.86615711e+01] , [3.82190960e+01],
#   [5.90628088e+00] , [2.25520817e-02]])

# #Constants
# a = 6793.137 #km
# mu = 398600.5 #km^3/s^2
# n = (mu/a**3)**(1/2)


# #print(x0.shape)
# #tf=365*24*60*60 #number of seconds in a year
# tf=24*60*60
# t_step=60 #time step is an hour, propagate every hour
# state=hill_eq(q0,n,tf,t_step)

# print(state[:,1])
