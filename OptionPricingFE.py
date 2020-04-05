## Projet Option Pricing Finite Element Method

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm

## Avec les lagrangian linear functions (2 nodes)


def BSS(S,K,sigma,r,T):
    d1=(math.log(S/K)+(r+1/2*sigma**2)*(T))/(sigma*math.sqrt(T))      ##########BLACK-SCHOLES call
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(norm.cdf(d1,0,1)*S-K*math.exp(-r*T)*norm.cdf(d2,0,1))

#-----------------------------------------------------------------------------------------------------------------
K=40 # strike, sert uniquement à la fin pour revenir sur C (?)
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
#L=23 # spatial interval (arbitraire...)
Lmax=0.92 # pr avoir 100 (0.92 pour 100)
Lmin=-6 # pr avoir 0.1...
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=101 #nombre de pts spatiaux
#dx=L/(N-1)
dx=(Lmax-Lmin)/(N-1)
dt=alpha*dx*dx
tmax=0.5*sigma*sigma*T
k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
fig_imp=plt.figure()
fig_imp.suptitle('Option pricing / Finite Element Method')
ax1_imp = fig_imp.add_subplot(1, 3, 1)
ax2_imp = fig_imp.add_subplot(1, 3, 2)
ax3_imp = fig_imp.add_subplot(1, 3, 3, projection='3d')

ax2_imp.set_xlim([0,105])
ax2_imp.set_ylim([0,70]) 
ax1_imp.set_ylim([0,1.63])
ax3_imp.set_zlim([0,60])
ax3_imp.set_xlim([0,100])

ax1_imp.set_title('u as a fct of x')
ax2_imp.set_title('C as a fct of S')
ax3_imp.set_title('C as a fct of x and t')

ax3_imp.set_xlabel('Stock Price')
ax3_imp.set_ylabel('tho')
ax3_imp.set_zlabel('Option Price')
ax1_imp.set_xlabel('x')
ax1_imp.set_ylabel('u')
ax2_imp.set_xlabel('S')
ax2_imp.set_ylabel('C')


#-----------------------------------------------------------------------------------------------------------------
x=np.linspace(Lmin,Lmax,N)
# x=np.linspace(0.1,100,N)
# for i in range(N):
#     x[i]=math.log(x[i]/K)    # si on veut adapter le maillage au pb du chgmt de var
    
q=np.zeros(2*N-2)
def q0(x):   #fct pour condition initiale
    if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0  

for i in range(N-1):
    q[2*i]=q0(x[i])
    q[2*i+1]=q0(x[i+1])

v=np.zeros(N)

#on détermine v à partir des coefficients
for i in range(N-1):
    v[i]=q[2*i]
v[N-1]=q[2*N-3]

temps=np.arange(0,tmax,dt)

# création de la matrice C
C_mat=np.zeros((2*N-2,2*N-2))
for i in range(1,2*N-3):
        C_mat[i][i]=dx/3*2
        C_mat[i][i-1]=dx/3*1/2
        C_mat[i][i+1]=dx/3*1/2
C_mat[0][0]=dx/3*1
C_mat[0][1]=dx/3*1/2
C_mat[2*N-3][2*N-3]=dx/3*1
C_mat[2*N-3][2*N-4]=dx/3*1/2


# création de la matrice K_mat
K_mat=np.zeros((2*N-2,2*N-2))
for i in range(1,2*N-3):
        K_mat[i][i]=1/dx*2
        K_mat[i][i-1]=-1/dx*1
        K_mat[i][i+1]=-1/dx*1
K_mat[0][0]=1/dx*1
K_mat[0][1]=-1/dx*1
K_mat[2*N-3][2*N-3]=1/dx*1
K_mat[2*N-3][2*N-4]=-1/dx*1

# création du vecteur F (natural boundary conditions)
F=np.zeros(2*N-2)
#...

#on revient sur C pour voir si ça donne bien le payoff final d une opt
C=np.zeros(N) #N valeurs, dernière indice N-1
for i in range(N):
    C[i]=K*v[i]*math.exp(a*x[i])
s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
ax1_imp.plot(x,v,'b:o',linewidth=1,markersize=2)
ax2_imp.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
#xdata=np.full(N,0)
#ydata=s
#zdata=np.copy(C)
ax2_imp.legend()
fig_imp.show()

Z=np.zeros((len(temps),len(x)))

Z[0]=zdata
compteur=1

for t in temps[1:]:
    
    #F[2*N-3]=(0.5*((k+1)*math.exp(Lmax)-(k-1)*math.exp(-k*t))*math.exp(-a*Lmax+b*t))*dx/2
    #F[2*N-3]=(math.exp(Lmax)-a*math.exp(Lmax)+a)*math.exp(-a*Lmax-b*t)*dx/2
    #F[2*N-3]=(math.exp((1-a)*Lmax-b*t)-a)*dx/2
    F[2*N-3]=((1-a)*math.exp(Lmax)-a)*math.exp(-a*Lmax-b*t)*dx/2 # natural boundary

    #test
    #F[0]=math.exp(Lmin-a*Lmin-B*t)    
    q_plus_frontieres=1/dt*np.dot(C_mat,q)+F
    
    q=np.linalg.solve(1/dt*C_mat+K_mat,q_plus_frontieres)
    #print(q)
    #q[1]=0 # mouai .......
    q[0]=0
    #print(q)
    
    #on détermine v à partir des coefficients
    for i in range(N-1):
        v[i]=q[2*i]
    v[N-1]=q[2*N-3]
    
    
    #on revient sur C
    for i in range(N):
        C[i]=K*v[i]*math.exp(a*x[i]+b*t)
    
    
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_imp.plot(x,v,'r:o',linewidth=1,markersize=2)
        ax2_imp.plot(s,C,'r:o',linewidth=1,markersize=2, label='prix à t=0')
        ax2_imp.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig_imp.show()
    
    else:
        xdata=np.full(N,t)
        ydata=s
        zdata=np.copy(C)
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_imp.plot(x,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2_imp.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        plt.pause(0.1)    
        #plt.show()
        fig_imp.show()
        
    compteur=compteur+1

BSS_liste=np.zeros(N)

for i in range(N):
    BSS_liste[i]=BSS(s[i],K,sigma,r,T)
ax2_imp.plot(s,BSS_liste,color='green',label = 'black scholes')
ax2_imp.legend()
X, Y = np.meshgrid(s, temps)
ax3_imp.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3_imp.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig_imp.show()



## Avec les quadratic functions (3 nodes)


def BSS(S,K,sigma,r,T):
    d1=(math.log(S/K)+(r+1/2*sigma**2)*(T))/(sigma*math.sqrt(T))      #BLACK-SCHOLES call
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(norm.cdf(d1,0,1)*S-K*math.exp(-r*T)*norm.cdf(d2,0,1))

#-----------------------------------------------------------------------------------------------------------------
K=40 # strike, sert uniquement à la fin pour revenir sur C (?)
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
#L=23 # spatial interval (arbitraire...)
Lmax=0.92 # pr avoir 100 (0.92 pour 100)
Lmin=-6 # pr avoir 0.1...
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
# on veut N impair (N-1 pair)
N=201 #nombre de pts spatiaux
#dx=L/(N-1)
dx=(Lmax-Lmin)/(N-1)
dt=alpha*dx*dx
tmax=0.5*sigma*sigma*T
k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
fig_imp=plt.figure()
fig_imp.suptitle('Option pricing / Finite Difference Method / Euler Implicit')
ax1_imp = fig_imp.add_subplot(1, 3, 1)
ax2_imp = fig_imp.add_subplot(1, 3, 2)
ax3_imp = fig_imp.add_subplot(1, 3, 3, projection='3d')

ax2_imp.set_xlim([0,100])
ax2_imp.set_ylim([0,60]) 
ax1_imp.set_ylim([0,1.63])
ax3_imp.set_zlim([0,60])
ax3_imp.set_xlim([0,100])

ax1_imp.set_title('u as a fct of x')
ax2_imp.set_title('C as a fct of S')
ax3_imp.set_title('C as a fct of x and t')

ax3_imp.set_xlabel('Stock Price')
ax3_imp.set_ylabel('tho')
ax3_imp.set_zlabel('Option Price')
ax1_imp.set_xlabel('x')
ax1_imp.set_ylabel('u')
ax2_imp.set_xlabel('S')
ax2_imp.set_ylabel('C')


#-----------------------------------------------------------------------------------------------------------------
x=np.linspace(Lmin,Lmax,N)

q=np.zeros(3*int(1/2*(N-3))+3) # dernier indice 3/2*(N-3)+2

def q0(x):   #fct pour condition initiale
    if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0    

# l indice du premier q du derner "element" est 3/2*(N-3)
for i in range(int((N-3)/2+1)): # N-1 pair !!!
    q[3*i]=q0(x[2*i+2])
    q[3*i+1]=q0(x[2*i+1])
    q[3*i+2]=q0(x[2*i])

v=np.zeros(N)
#on détermine v à partir des coefficients

for i in range(int((N-3)/2+1)): # N-1 pair !!!
    v[2*i]=q[3*i+2]
    v[2*i+1]=q[3*i+1]
    v[2*i+2]=q[3*i]


temps=np.arange(0,tmax,dt)

# création de la matrice C_mat
C_mat=np.zeros((int((N-3)*3/2+3),int((N-3)*3/2+3)))

for i in range(int((N-3)*3/2+1)):
    C_mat[i][i]+=dx/15*4
    C_mat[i][i+1]+=dx/15*2
    C_mat[i][i+2]+=dx/15*(-1)
    C_mat[i+1][i]+=dx/15*2
    C_mat[i+1][i+1]+=dx/15*16
    C_mat[i+1][i+2]+=dx/15*2
    C_mat[i+2][i]+=dx/15*(-1)
    C_mat[i+2][i+1]+=dx/15*2
    C_mat[i+2][i+2]+=dx/15*4
    

# création de la matrice K_mat
K_mat=np.zeros((int((N-3)*3/2+3),int((N-3)*3/2+3)))

for i in range(int((N-3)*3/2+1)):
    K_mat[i][i]+=1/(6*dx)*7
    K_mat[i][i+1]+=1/(6*dx)*(-8)
    K_mat[i][i+2]+=1/(6*dx)*1
    K_mat[i+1][i]+=1/(6*dx)*(-8)
    K_mat[i+1][i+1]+=1/(6*dx)*16
    K_mat[i+1][i+2]+=1/(6*dx)*(-8)
    K_mat[i+2][i]+=1/(6*dx)*1
    K_mat[i+2][i+1]+=1/(6*dx)*(-8)
    K_mat[i+2][i+2]+=1/(6*dx)*7
    

# création du vecteur F (natural boundary conditions)
F=np.zeros(int((N-3)*3/2+3))
#...

#on revient sur C pour voir si ça donne bien le payoff final d une opt
C=np.zeros(N) #N valeurs, dernière indice N-1
for i in range(N):
    C[i]=K*v[i]*math.exp(a*x[i])
s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
ax1_imp.plot(x,v,'b:o',linewidth=1,markersize=2)
ax2_imp.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3_imp.scatter3D(xdata, ydata, zdata)
ax2_imp.legend()
fig_imp.show()

Z=np.zeros((len(temps),len(x)))

Z[0]=zdata
compteur=1

for t in temps[1:]:
    
    #F[2*N-3]=(0.5*((k+1)*math.exp(Lmax)-(k-1)*math.exp(-k*t))*math.exp(-a*Lmax+b*t))*dx/2
    F[int((N-3)*3/2+2)]=(math.exp(Lmax)-a*math.exp(Lmax)+a)*math.exp(-a*Lmax-b*t)*dx
    q_plus_frontieres=1/dt*np.dot(C_mat,q)+F
    
    q=np.linalg.solve(1/dt*C_mat+K_mat,q_plus_frontieres)
    q[1]=0 # mouai .......
    print(q)
    
    #on détermine v à partir des coefficients
    for i in range(int((N-3)/2+1)): # N-1 pair !!!
        v[2*i]=q[3*i+2]
        v[2*i+1]=q[3*i+1]
        v[2*i+2]=q[3*i]
    
    #on revient sur C
    for i in range(N):
        C[i]=K*v[i]*math.exp(a*x[i]+b*t)
    
    
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_imp.plot(x,v,'r:o',linewidth=1,markersize=2)
        ax2_imp.plot(s,C,'r:o',linewidth=1,markersize=2, label='prix à t=0')
        ax2_imp.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig_imp.show()
    
    else:
        xdata=np.full(N,t)
        ydata=s
        zdata=np.copy(C)
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_imp.plot(x,v,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2_imp.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        plt.pause(0.1)    
        #plt.show()
        fig_imp.show()
        
    compteur=compteur+1

BSS_liste=np.zeros(N)

for i in range(N):
    BSS_liste[i]=BSS(s[i],K,sigma,r,T)
ax2_imp.plot(s,BSS_liste,label = 'black scholes')
ax2_imp.legend()
X, Y = np.meshgrid(s, temps)
ax3_imp.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3_imp.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig_imp.show()

    