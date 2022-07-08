import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
ro = 7850
A = 0.05
L = 1
E = 210000000000
mass_matrix_coeff = (ro*A*L)/6
# M = mass_matrix_coeff * np.array([[2,1],[1,2]])
# K = np.array([[131250000, -65625000],[-65625000,-34375000]])
F = np.array([[10000000],[0]])
M = np.array([[6.5416, 1.6354],[1.6354,22.8958]])
K = np.array([[525000000, -262500000],[-262500000,362500000]])


#Chung and Hulbert Optimal parameters

Tmin = 0.0006447
dt = 0.1 * Tmin
tTotal = 0.01
t = np.arange(0, tTotal, dt)
# u0 = np.matmul(inv(K),F)
# print(u0)


def calculateInvKeff(alpha_m,alpha_f,beta,M,K,dt):
    Keff = M*(1-alpha_m)/beta/dt/dt + K*(1-alpha_f)
    KeffInv = np.linalg.inv(Keff)
    return KeffInv


def calculatefeff (alpha_m, alpha_f,beta, M, K, dt, dis, vel, acc):
    res = -np.matmul (K * alpha_f, dis)
    res = res + np.matmul (M,(1.-alpha_m)/beta/dt/dt*dis + (1.-alpha_m)/beta/dt*vel + (1.-alpha_m-2.*beta)/2./beta*acc)
    return res

def performCalculation(alpha_m, alpha_f, gama, beta, M, K, dt, t, u0, ud0, udd0):
    u = np.zeros([2,len(t)]); u[:,0] = 1.*u0
    ud = np.zeros([2,len(t)]); ud[:,0] = 1.*ud0
    udd = np.zeros([2,len(t)]); udd[:,0] = 1.*udd0
    KeffInv = calculateInvKeff(alpha_m,alpha_f,beta,M,K,dt)
    for i in range(len(t)-1):
        feff = calculatefeff(alpha_m,alpha_f,beta,M,K,dt,u[:,i],ud[:,i],udd[:,i])
        u[:,i+1] = np.matmul(KeffInv,feff)
        ud[:,i+1] = gama/beta/dt*(u[:,i+1]-u[:,i]) - (gama-beta)/beta*ud[:,i] - (gama-2.*beta)/2./beta*dt*udd[:,i]
        udd[:,i+1] = 1./beta/dt/dt*(u[:,i+1]-u[:,i]) - 1./beta/dt*ud[:,i] - (1.-2.*beta)/2./beta*udd[:,i]
    return u, ud, udd





u0 = np.array([0.02985843,0.02162162])
ud0 = np.array([0,0])
udd0 = np.matmul(np.linalg.inv(M),-np.matmul(K,u0))
r= 1
alpha_m = (2*r - 1 ) / (r+1)
alpha_f = r / (r+1)
beta = 0.25*(1-alpha_m+alpha_f)**2
gama = 0.5-alpha_m+alpha_f
uCH, udCH, uddCH = performCalculation(alpha_m, alpha_f, gama, beta, M, K, dt, t, u0, ud0,udd0)



plt.rcParams["figure.figsize"]=15,5
plt.plot(t, uCH[1,:], label='displ 2',linewidth=5,color='blue')
plt.plot(t, uCH[0,:], label='displ 1',linewidth=4,color='red')
plt.legend(loc='best')
plt.title('Chung and Hulbert (1993)')
plt.axhline(0, color='black')
plt.show()