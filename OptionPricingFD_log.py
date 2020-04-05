## Projet Option Pricing Finite Difference - log - CN

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm
#-----------------------------------------------------------------------------------------------------------------
K=40
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
zmax=4.6
zmin=-2.3
alpha=0.4 # cond de stabilité : alpha<=0.5 (alpha=dt/(dz*2)) ???
N=131 #nombre de pts spatiaux
dz=(zmax-zmin)/(N-1)
#dt=alpha*dz*dz
dt=T/(N-1)

a=0.25*((sigma/dz)**2+(r-sigma*sigma/2)/dz)
d=1/dt-0.5*((sigma/dz)**2)
c=0.25*((sigma/dz)**2-(r-sigma*sigma/2)/dz)
astar=-a
dstar=1/dt+r+0.5*((sigma/dz)**2)
cstar=-c
#-----------------------------------------------------------------------------------------------------------------
# création de la matrice D
D=np.zeros((N-2,N-2))
for i in range(1,N-3):
        D[i][i]=dstar
        D[i][i-1]=cstar
        D[i][i+1]=astar
D[0][0]=dstar
D[0][1]=astar
D[N-3][N-3]=dstar
D[N-3][N-4]=cstar
#-----------------------------------------------------------------------------------------------------------------
fig=plt.figure()
fig.suptitle('Option pricing / Finite Difference Method / log - CN')
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax2.set_xlim([0,100])
ax2.set_ylim([0,60]) 
ax1.set_ylim([0,60])
ax3.set_zlim([0,60])
ax3.set_xlim([0,100])

ax1.set_title('v as a fct of z')
ax2.set_title('C as a fct of S')
ax3.set_title('C as a fct of S and t')

ax3.set_xlabel('Stock Price')
ax3.set_ylabel('time')
ax3.set_zlabel('Option Price')
ax1.set_xlabel('z')
ax1.set_ylabel('v/C')
ax2.set_xlabel('S')
ax2.set_ylabel('C')
#-----------------------------------------------------------------------------------------------------------------
def v0(z):   #fct pour condition initiale
    if (math.exp(z)-K)>=0:
        return (math.exp(z)-K)
    else:
        return 0        
z=np.linspace(zmin,zmax,N)
v=np.zeros(N) 
for i in range(N):   #on remplit u à maturité puis on reviendra en arr avec le schéma
    v[i]=v0(z[i])
s=np.zeros(N)
for j in range(N):
    s[j]=math.exp(z[j])
ax1.plot(z,v,'b:o',linewidth=1,markersize=4)
ax2.plot(s,v,'b:o',linewidth=1,markersize=4,label='prix à t=T')
ax2.legend()
fig.show()
#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,T,dt)
temps=np.flip(temps,0)
Z=np.zeros((len(temps),len(z)))

Z[0]=np.copy(v)

compteur=1

#plt.clf()
for t in temps[1:]:
    v_plus_frontieres=d*v[1:N-1]+c*v[0:N-2]+a*v[2:N]
    #v_plus_frontieres[0]=v_plus_frontieres[0]
    v_plus_frontieres[N-3]=v_plus_frontieres[N-3]-astar*(math.exp(zmax)-K)
    
    v[1:N-1]=np.linalg.solve(D,v_plus_frontieres)
    v[0]=0
    v[N-1]=math.exp(zmax)-K
    
    Z[compteur]=np.copy(v)
    
    if t==temps[len(temps)-1]:
        ax1.plot(z,v,'r:o',linewidth=1,markersize=4)
        ax2.plot(s,v,'r:o',linewidth=1,markersize=4, label='prix à t=0')
        ax2.legend()
        fig.suptitle(t/T)
        plt.pause(0.01)
        #plt.show()
        fig.show()
    else:
        ax1.plot(z,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2.plot(s,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        fig.suptitle(t/T)
        plt.pause(0.01)    
        #plt.show()
        fig.show()
    
    compteur=compteur+1

X, Y = np.meshgrid(s,temps)
ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig.show()

## Explicit

#-----------------------------------------------------------------------------------------------------------------
K=40
r=0.05
sigma=0.3 
T=1 # unité ?
zmax=4.6
zmin=-2.3
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dz*2)) ???
N=31 #nombre de pts spatiaux
dz=(zmax-zmin)/(N-1)
dt=alpha*dz*dz*dz


cst=1-sigma*sigma*dt/dz-r*dt
cstplus=sigma*sigma*0.5*dt/(dz*dz)+(r-sigma*sigma/2)/2*dt/dz
cstmoins=sigma*sigma*0.5*dt/(dz*dz)-(r-sigma*sigma/2)*0.5*dt/dz

cstnew=1+sigma*sigma*dt/dz+r*dt
cstplusnew=-sigma*sigma*0.5*dt/(dz*dz)-(r-sigma*sigma/2)/2*dt/dz
cstmoinsnew=-sigma*sigma*0.5*dt/(dz*dz)+(r-sigma*sigma/2)*0.5*dt/dz

csthull=1/(1+r*dt)*(1-dt/(dz**2)*sigma**2)
cstplushull=1/(1+r*dt)*(dt/(2*dz)*(r-sigma*sigma*0.5)+dt/(dz**2)*0.5*sigma**2)
cstmoinshull=1/(1+r*dt)*(-dt/(2*dz)*(r-sigma*sigma*0.5)+dt/(dz**2)*0.5*sigma**2)

a=0.25*((sigma/dz)**2+(r-sigma*sigma/2)/dz)
d=1/dt-0.5*((sigma/dz)**2)
c=0.25*((sigma/dz)**2-(r-sigma*sigma/2)/dz)
astar=-a
dstar=1/dt+r+0.5*((sigma/dz)**2)
cstar=-c
#-----------------------------------------------------------------------------------------------------------------
# création de la matrice D
D=np.zeros((N-2,N-2))
for i in range(1,N-3):
        D[i][i]=dstar
        D[i][i-1]=cstar
        D[i][i+1]=astar
D[0][0]=dstar
D[0][1]=astar
D[N-3][N-3]=dstar
D[N-3][N-4]=cstar
#-----------------------------------------------------------------------------------------------------------------
fig=plt.figure()
fig.suptitle('Option pricing / Finite Difference Method / log - CN')
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax2.set_xlim([0,100])
ax2.set_ylim([0,60]) 
ax1.set_ylim([0,60])
ax3.set_zlim([0,60])
ax3.set_xlim([0,100])

ax1.set_title('v/C as a fct of z')
ax2.set_title('C as a fct of S')
ax3.set_title('C as a fct of S and t')

ax3.set_xlabel('Stock Price')
ax3.set_ylabel('time')
ax3.set_zlabel('Option Price')
ax1.set_xlabel('z')
ax1.set_ylabel('v/C')
ax2.set_xlabel('S')
ax2.set_ylabel('C')
#-----------------------------------------------------------------------------------------------------------------
def v0(z):   #fct pour condition initiale
    if (math.exp(z)-K)>=0:
        return (math.exp(z)-K)
    else:
        return 0        
z=np.linspace(zmin,zmax,N)
v=np.zeros(N) 
for i in range(N):   #on remplit u à maturité puis on reviendra en arr avec le schéma
    v[i]=v0(z[i])
s=np.zeros(N)
for j in range(N):
    s[j]=math.exp(z[j])
ax1.plot(z,v,'b:o',linewidth=1,markersize=4)
ax2.plot(s,v,'b:o',linewidth=1,markersize=4,label='prix à t=T')
ax2.legend()
fig.show()
#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,T,dt)
temps=np.flip(temps,0) # <---------flip le tps
Z=np.zeros((len(temps),len(z)))

Z[0]=np.copy(v)

compteur=1

#plt.clf()
for t in temps[1:]:
    v[1:N-1]=csthull*v[1:N-1]+cstmoinshull*v[0:N-2]+cstplushull*v[2:N]
    v[0]=0
    v[N-1]=math.exp(zmax)-K
    
    Z[compteur]=np.copy(v)
    
    if t==temps[len(temps)-1]:
        ax1.plot(z,v,'r:o',linewidth=1,markersize=4)
        ax2.plot(s,v,'r:o',linewidth=1,markersize=4, label='prix à t=0')
        ax2.legend()
        fig.suptitle(t/T)
        plt.pause(0.01)
        #plt.show()
        fig.show()
    else:
        ax1.plot(z,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2.plot(s,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        fig.suptitle(t/T)
        plt.pause(0.01)    
        #plt.show()
        fig.show()
    
    compteur=compteur+1


X, Y = np.meshgrid(s,temps)
ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig.show()

