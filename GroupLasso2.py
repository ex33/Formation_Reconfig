#Group Lasso
from Impulsive_Maneuver import Impulsive_Maneuver
from No_Drift_Centered_Conditions import no_drift_centered
import numpy as np
import math
from Hill_Eq import hill_eq
import matplotlib.pyplot as plt
from Delta_V_Calculator import delta_v

from celer import LassoCV,GroupLassoCV, GroupLasso, Lasso
from celer.plot_utils import configure_plt
#Simulate test 
#Subroutine 1: Add delta_v to state
#Given a time span, discretize entire range. Assume every range is a potential manuever.
#At an obersvation, all manuevers adds up to the final position
#After observation, no more manuvers will contribue


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
    """Given 2 time vectors, where tau represents the entire discretized time span of the trajectory and t is observation times,
    solve for the matrix  that will be used to solve for the impulsive manuvers via least squares method

    :param tau: Entire discretized time span of trajectory. 
    :type tau: 1xN Matrix
    :param t: Observation times
    :type t: 1xM

    :return: Forms A matrix to solve the problem of Ax=B
    :rtype: Returns (N*3) x (6+M*3) Matrix
    """
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

def form_B_vector(tau,t,manuver,r0,v0,del_v):
    """Normally, we would be given the position of the spacecraft at each observation, but this will be artifically made


    :param tau: _description_
    :type tau: _type_
    :param t: _description_
    :type t: _type_
    :param manuver: _description_
    :type manuver: _type_
    """
    N=len(tau)
    M=len(t)
    L=len(manuver)
    B=[]
    for j in range(M):
        row=np.matmul(calc_RR(t[j]),r0)+np.matmul(calc_RV(t[j]),v0)
        for k in range(L):
            if t[j]>manuver[k]:
                row=row+np.matmul(calc_RV(t[j]-manuver[k]),del_v)
        if j==0:
            B=row
            #print(B)
        else:
            if j>12:
                print(j)
                print((row))
            
            B=np.vstack((B,row))
            
            #print(B)
    return B
        
def form_x_vector(State1,tau,manuver):
    x=State1
    for j in range(0,N):
        row=np.zeros([3,1])
        for k in range(L):
            if tau[j]==manuver[k]:
                row=del_v
        x=np.vstack((x,row))
    return x



N=140
M=15

tau=np.empty([N])
t=np.empty([M])
for i in range(N):
    tau[i]=10*(i+1)
for j in range(M):
    t[j]=100*(j+1)


manuver=np.array([200,260,600,750])
L=len(manuver)

# C=100
# D=50
# psi=0
# phi=0
a = 6793.137 #km
mu = 398600.5 #km^3/s^2
omega =math.sqrt(mu/a**3)

# tau=np.zeros([4]) 
# #Step 1: Obtain first state

A=form_A_matrix(tau,t)
State1=no_drift_centered(200,50,0,0,omega)
r0=State1[0:3]
v0=State1[3:6]
del_v=np.array([[0.001],[0.001],[0.001]])

B=form_B_vector(tau,t,manuver,r0,v0,del_v)

A_Control = A[:,6:]
print( (A[:,:6] @ State1))
B_Control = B[:,0] - (A[:,:6] @ State1)[:,0]


x=form_x_vector(State1,tau,manuver)

#print("x")
#print(x)
print("Stuff")
print(np.linalg.norm((A_Control @ x[6:,0]) - B_Control))

print(A_Control.shape)
print(B_Control.shape)

LS=np.linalg.lstsq(A_Control,B_Control)
LS0=LS[0]
plt.plot(LS0)

plt.figure()
plt.plot(x[6:])
print(np.linalg.norm((A_Control @ LS[0]) - B_Control))


#Form group list of list
group=[]
for i in np.arange(0,(N)*3,3):
    group.append([i,i+1,i+2])

print(group)

print("alpha")
alpha=np.max(A_Control.transpose()@B_Control)
print(alpha)
#group_lasso = GroupLassoCV(groups=group)



#compute column norms
print("col norms")
col_norms = np.linalg.norm(A_Control, axis=0)
print(col_norms)

#normalize columns
A_normalized = A_Control / col_norms
print(A_normalized)



#group_lasso = GroupLasso(groups=group, alpha=.000087)
group_lasso = GroupLassoCV(groups=group)
#group_lasso = LassoCV()
#group_lasso = Lasso(alpha=.05)
group_lasso.fit(A_normalized, B_Control)
print("alpha")
print(group_lasso.alpha_)

#print(group_lasso.coef_)
plt.figure()
plt.plot(group_lasso.coef_ / col_norms)
print(group_lasso.coef_.shape)
print(np.linalg.norm((A_Control @ (group_lasso.coef_ / col_norms)) - B_Control))
plt.figure()
plt.plot((A_Control @ (group_lasso.coef_ / col_norms)) - B_Control)


j=0
norm_squared_individual = 0
overall_control_norm = 0
for i in x[6:]:
    norm_squared_individual += i**2
    if j % 3 == 2:
        overall_control_norm += math.sqrt(norm_squared_individual)
        norm_squared_individual = 0
    j+=1

print("true total delta-v")
print(overall_control_norm)

j=0
norm_squared_individual = 0
overall_control_norm = 0
for i in (group_lasso.coef_ / col_norms):
    norm_squared_individual += i**2
    if j % 3 == 2:
        overall_control_norm += math.sqrt(norm_squared_individual)
        norm_squared_individual = 0
    j+=1

print("calculated total delta-v")
print(overall_control_norm)



"""
print("Estimated regularization parameter alpha: %s" % group_lasso.alpha_)

fig = plt.figure(figsize=(6, 3), constrained_layout=True)
plt.semilogx(group_lasso.alphas_, group_lasso.mse_path_, ':')
plt.semilogx(group_lasso.alphas_, group_lasso.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(group_lasso.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square prediction error')
"""
plt.show()


