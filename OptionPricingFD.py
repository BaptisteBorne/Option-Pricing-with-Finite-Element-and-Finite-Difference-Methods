## Projet Option Pricing Finite Difference

## Schéma d Euler explicite

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm


#-----------------------------------------------------------------------------------------------------------------
def BSS(S,K,sigma,r,T):
    d1=(math.log(S/K)+(r+1/2*sigma**2)*(T))/(sigma*math.sqrt(T))      ##########BLACK-SCHOLES call
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(norm.cdf(d1,0,1)*S-K*math.exp(-r*T)*norm.cdf(d2,0,1))
#-----------------------------------------------------------------------------------------------------------------
K=40 # strike, sert uniquement à la fin pour revenir sur C (?)
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
Lmax=math.log(140/40)
Lmin=-6
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=191 #nombre de pts spatiaux
#dx=L/(N-1)
dx=(Lmax-Lmin)/(N-1)
tmax=0.5*sigma*sigma*T
k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Option pricing / Finite Difference Method / Euler Explicit')
# ax1.set_title('u(x)')
# ax2.set_title('C(S)') 
#-----------------------------------------------------------------------------------------------------------------
fig=plt.figure()
fig.suptitle('Option pricing / Finite Difference Method / Euler Explicit')
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax2.set_xlim([0,102])
ax2.set_ylim([0,60]) 
ax1.set_ylim([0,1.75])
ax3.set_zlim([0,60])
ax3.set_xlim([0,102])

ax1.set_title('u as a fct of x')
ax2.set_title('C as a fct of S')
ax3.set_title('C as a fct of x and t')

ax3.set_xlabel('Stock Price')
ax3.set_ylabel('tho')
ax3.set_zlabel('Option Price')
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax2.set_xlabel('S')
ax2.set_ylabel('C')

#-----------------------------------------------------------------------------------------------------------------
def u0(x):   #fct pour condition initiale
    if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0        
#x=np.linspace(-L,L,N) #vecteur des "abscices" spatiales
x=np.linspace(Lmin,Lmax,N)
u=np.zeros(N) #N valeurs, dernière indice N-1 // vecteur qui contiendra les valeurs de u, 
for i in range(N):   #on remplit u à maturité puis on reviendra en arr avec le schéma
    u[i]=u0(x[i])
print(u) # bien N valeurs, derniere indice N-1

#on revient sur C pour voir si ça donne bien le payoff final d une opt
C=np.zeros(N) #N valeurs, dernière indice N-1
for i in range(N):
    C[i]=K*u[i]*math.exp(a*x[i])

s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
ax1.plot(x,u,'b:o',linewidth=1,markersize=2)
ax2.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3.scatter3D(xdata, ydata, zdata)
ax2.legend()
fig.show()

# ça aplatie exponentiellement la courbe en gros
#plt.legend()
#plt.axis([-2,100,-4,100])
#plt.show()
#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,tmax,alpha*dx*dx)
#C=np.zeros(N) #N valeurs, dernière indice N-1
Z=np.zeros((len(temps),len(x)))

print(zdata)
Z[0]=zdata
print(Z[0])
compteur=1

#plt.clf()
for t in temps[1:]:
    u[1:N-1]=u[1:N-1]+alpha*(u[0:N-2]-2*u[1:N-1]+u[2:N])
    u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*t))
    #u[0]=0
    u[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*t))
    
    #on revient sur C
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i]+b*t)
    
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1.plot(x,u,'r:o',linewidth=1,markersize=2)
        ax2.plot(s,C,'r:o',linewidth=1,markersize=2, label='prix à t=0')
        ax2.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig.show()
    else:
        xdata=np.full(N,t)
        ydata=s
        zdata=np.copy(C)
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1.plot(x,u,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        plt.pause(0.1)    
        #plt.show()
        fig.show()
        
    compteur=compteur+1
    
X, Y = np.meshgrid(s, temps)
ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')
BSS_list_exp=np.zeros(N)
for i in range(N):
    BSS_list_exp[i]=BSS(s[i],K,sigma,r,T)
ax2.plot(s,BSS_list_exp)

fig.show()

#-----------------------------------------------------------------------------------------------------------------

# fig2 = plt.figure()
# wf = fig2.add_subplot(1, 2, 1,projection='3d')
# surf = fig2.add_subplot(1, 2, 2,projection='3d')
# #wf = plt.axes(projection="3d")
# X, Y = np.meshgrid(s, temps)
# wf.plot_wireframe(X, Y, Z, color='green')
# 
# wf.set_xlabel('S')
# wf.set_ylabel('t')
# wf.set_zlabel('Price')
# 
# surf.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='winter', edgecolor='none')
# fig2.show()


## Schéma d Euler implicite


K=40 # strike, sert uniquement à la fin pour revenir sur C (?)
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
#L=23 # spatial interval (arbitraire...)
Lmax=0.9
Lmin=-6
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=131 #nombre de pts spatiaux
#dx=L/(N-1)
dx=(Lmax-Lmin)/(N-1)
dt=dx*dx*alpha
tmax=0.5*sigma*sigma*T
k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
# création de la matrice A
A=np.zeros((N-2,N-2))
for i in range(1,N-3):
        A[i][i]=1+2*alpha
        A[i][i-1]=-alpha
        A[i][i+1]=-alpha
A[0][0]=1+2*alpha
A[0][1]=-alpha
A[N-3][N-3]=1+2*alpha
A[N-3][N-4]=-alpha
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
def u0(x):   #fct pour condition initiale
    if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0        
x=np.linspace(Lmin,Lmax,N)
u=np.zeros(N) #N valeurs, dernière indice N-1 // vecteur qui contiendra les valeurs de u, 
for i in range(N):   #on remplit u à maturité puis on reviendra en arr avec le schéma
    u[i]=u0(x[i])
#print(u) # bien N valeurs, derniere indice N-1

#on revient sur C pour voir si ça donne bien le payoff final d une opt
C=np.zeros(N) #N valeurs, dernière indice N-1
for i in range(N):
    C[i]=K*u[i]*math.exp(a*x[i])
s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
ax1_imp.plot(x,u,'b:o',linewidth=1,markersize=2)
ax2_imp.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3_imp.scatter3D(xdata, ydata, zdata)
ax2_imp.legend()
fig_imp.show()

#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,tmax,alpha*dx*dx)
C=np.zeros(N) #N valeurs, dernière indice N-1
Z=np.zeros((len(temps),len(x)))

Z[0]=zdata
compteur=1

#plt.clf()
for t in temps[1:]:
    u_plus_frontieres=np.copy(u[1:N-1])
    u_plus_frontieres[0]=u_plus_frontieres[0]+alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
    u_plus_frontieres[N-3]=u_plus_frontieres[N-3]+alpha*((math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t))))
    
    u[1:N-1]=np.linalg.solve(A,u_plus_frontieres)
    u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
    u[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t)))

    #on revient sur C
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i]+b*t)
        
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_imp.plot(x,u,'r:o',linewidth=1,markersize=2)
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
        ax1_imp.plot(x,u,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2_imp.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        plt.pause(0.1)    
        #plt.show()
        fig_imp.show()
        
    compteur=compteur+1

X, Y = np.meshgrid(s, temps)
ax3_imp.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3_imp.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig_imp.show()


## Schéma de Cranck-Nicolson

K=40 # strike, sert uniquement à la fin pour revenir sur C (?)
r=0.05
sigma=0.3 # volatilité
T=1 #maturité de l option, unité ?
#L=23 # spatial interval (arbitraire...)
Lmax=0.9
Lmin=-6
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=131 #nombre de pts spatiaux
#dx=L/(N-1)
dx=(Lmax-Lmin)/(N-1)
tmax=0.5*sigma*sigma*T
k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
# création de la matrice D
D=np.zeros((N-2,N-2))
for i in range(1,N-3):
        D[i][i]=1+alpha
        D[i][i-1]=-0.5*alpha
        D[i][i+1]=-0.5*alpha
D[0][0]=1+alpha
D[0][1]=-0.5*alpha
D[N-3][N-3]=1+alpha
D[N-3][N-4]=-0.5*alpha
#-----------------------------------------------------------------------------------------------------------------
fig_cn=plt.figure()
fig_cn.suptitle('Option pricing / Finite Difference Method / Cranck-Nicolson')
ax1_cn = fig_cn.add_subplot(1, 3, 1)
ax2_cn = fig_cn.add_subplot(1, 3, 2)
ax3_cn = fig_cn.add_subplot(1, 3, 3, projection='3d')

ax2_cn.set_xlim([0,100])
ax2_cn.set_ylim([0,60]) 
ax1_cn.set_ylim([0,1.63])
ax3_cn.set_zlim([0,60])
ax3_cn.set_xlim([0,100])

ax1_cn.set_title('u as a fct of x')
ax2_cn.set_title('C as a fct of S')
ax3_cn.set_title('C as a fct of x and t')

ax3_cn.set_xlabel('Stock Price')
ax3_cn.set_ylabel('tho')
ax3_cn.set_zlabel('Option Price')
ax1_cn.set_xlabel('x')
ax1_cn.set_ylabel('u')
ax2_cn.set_xlabel('S')
ax2_cn.set_ylabel('C')

#-----------------------------------------------------------------------------------------------------------------
def u0(x):   #fct pour condition initiale
    if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0        
x=np.linspace(Lmin,Lmax,N)
u=np.zeros(N) #N valeurs, dernière indice N-1 // vecteur qui contiendra les valeurs de u, 
for i in range(N):   #on remplit u à maturité puis on reviendra en arr avec le schéma
    u[i]=u0(x[i])
#print(u) # bien N valeurs, derniere indice N-1

#on revient sur C pour voir si ça donne bien le payoff final d une opt
C=np.zeros(N) #N valeurs, dernière indice N-1
for i in range(N):
    C[i]=K*u[i]*math.exp(a*x[i])
s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
ax1_cn.plot(x,u,'b:o',linewidth=1,markersize=2)
ax2_cn.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3_cn.scatter3D(xdata, ydata, zdata)
ax2_cn.legend()
fig_cn.show()

#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,tmax,alpha*dx*dx)
C=np.zeros(N) #N valeurs, dernière indice N-1
Z=np.zeros((len(temps),len(x)))

Z[0]=zdata
compteur=1

#plt.clf()
for t in temps[1:]:
    u_plus_frontieres=(1-alpha)*u[1:N-1]+0.5*alpha*u[0:N-2]+0.5*alpha*u[2:N]
    u_plus_frontieres[0]=u_plus_frontieres[0]+0.5*alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
    u_plus_frontieres[N-3]=u_plus_frontieres[N-3]+0.5*alpha*((math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t))))
    
    u[1:N-1]=np.linalg.solve(D,u_plus_frontieres)
    u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
    u[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t)))
    
    #on revient sur C
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i]+b*t)
        
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_cn.plot(x,u,'r:o',linewidth=1,markersize=2)
        ax2_cn.plot(s,C,'r:o',linewidth=1,markersize=2, label='prix à t=0')
        ax2_cn.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig_cn.show()
    else:
        xdata=np.full(N,t)
        ydata=s
        zdata=np.copy(C)
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1_cn.plot(x,u,color='grey',marker='',linestyle='dashed',linewidth=1)
        ax2_cn.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        plt.pause(0.1)    
        #plt.show()
        fig_cn.show()
        
    compteur=compteur+1

X, Y = np.meshgrid(s, temps)
ax3_cn.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3_cn.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig_cn.show()

## Erreur relative avec les 3 schémas


listeN=np.arange(10,60,1)
err_imp=np.zeros(len(listeN))
err_exp=np.zeros(len(listeN))
err_cn=np.zeros(len(listeN))

compt=0

for elt in listeN:
    K=40 ; r=0.05 ; sigma=0.3 # vol
    T=1 #maturité de l option, unité ?
    Lmax=0.9 ; Lmin=-6 ; alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
    N=elt #nombre de pts spatiaux
    dx=(Lmax-Lmin)/(N-1)
    #dt= alpha*dx*dx
    dt=0.5*((Lmax-Lmin)/(30-1))**2
    tmax=0.5*sigma*sigma*T
    k=r/(0.5*sigma*sigma)
    a=(1-k)/2 ; b=a*a+(k-1)*a-k
    A=np.zeros((N-2,N-2))
    for i in range(1,N-3):
            A[i][i]=1+2*alpha
            A[i][i-1]=-alpha
            A[i][i+1]=-alpha
    A[0][0]=1+2*alpha
    A[0][1]=-alpha
    A[N-3][N-3]=1+2*alpha
    A[N-3][N-4]=-alpha
    D=np.zeros((N-2,N-2))
    for i in range(1,N-3):
            D[i][i]=1+alpha
            D[i][i-1]=-0.5*alpha
            D[i][i+1]=-0.5*alpha
    D[0][0]=1+alpha
    D[0][1]=-0.5*alpha
    D[N-3][N-3]=1+alpha
    D[N-3][N-4]=-0.5*alpha
    def u0(x):   
        if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
            return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
        else:
            return 0        
    x=np.linspace(Lmin,Lmax,N)
    u=np.zeros(N)  
    u_exp=np.zeros(N)
    u_cn=np.zeros(N)
    for i in range(N):   
        u[i]=u0(x[i])
        u_exp[i]=u0(x[i])
        u_cn[i]=u0(x[i])
    C=np.zeros(N)
    C_exp=np.zeros(N)
    C_cn=np.zeros(N)
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i])
        C_exp[i]=K*u_exp[i]*math.exp(a*x[i])
        C_cn[i]=K*u_cn[i]*math.exp(a*x[i])
    s=np.zeros(N)
    for j in range(N):
        s[j]=K*math.exp(x[j])
    temps=np.arange(0,tmax,dt)
    for t in temps[1:]:
        #calcul u pr implicit
        u[1:N-1]=u[1:N-1]+alpha*(u[0:N-2]-2*u[1:N-1]+u[2:N])
        u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*t))
        u[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*t))
        #calcul u pour explicit
        u_exp_plus_frontieres=np.copy(u_exp[1:N-1])
        u_exp_plus_frontieres[0]=u_exp_plus_frontieres[0]+alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
        u_exp_plus_frontieres[N-3]=u_exp_plus_frontieres[N-3]+alpha*((math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t))))
        u_exp[1:N-1]=np.linalg.solve(A,u_exp_plus_frontieres)
        u_exp[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
        u_exp[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t)))
        #calcul u pour Cranck-Nicolson
        u_cn_plus_frontieres=(1-alpha)*u_cn[1:N-1]+0.5*alpha*u_cn[0:N-2]+0.5*alpha*u_cn[2:N]
        u_cn_plus_frontieres[0]=u_cn_plus_frontieres[0]+0.5*alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
        u_cn_plus_frontieres[N-3]=u_cn_plus_frontieres[N-3]+0.5*alpha*((math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t))))
        u_cn[1:N-1]=np.linalg.solve(D,u_cn_plus_frontieres)
        u_cn[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
        u_cn[N-1]=(math.exp(Lmax)-1)*math.exp(-(a*Lmax+b*(t)))
        
        for i in range(N):
            C[i]=K*u[i]*math.exp(a*x[i]+b*t)
            C_exp[i]=K*u_exp[i]*math.exp(a*x[i]+b*t)
            C_cn[i]=K*u_cn[i]*math.exp(a*x[i]+b*t)
            
    # calcul erreur relative pr implicit (pour le nb de coord spatiales de la boucle)
    for l in range(N):
        err_imp[compt]=err_imp[compt]+1/N*(C[l]-(s[l]*norm.cdf((math.log(s[l]/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)-K*math.exp(-r*T)*norm.cdf((math.log(s[l]/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)))**2
        err_exp[compt]=err_exp[compt]+1/N*(C_exp[l]-(s[l]*norm.cdf((math.log(s[l]/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)-K*math.exp(-r*T)*norm.cdf((math.log(s[l]/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)))**2
        err_cn[compt]=err_cn[compt]+1/N*(C_cn[l]-(s[l]*norm.cdf((math.log(s[l]/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)-K*math.exp(-r*T)*norm.cdf((math.log(s[l]/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T)),0,1)))**2
        
    err_imp[compt]=err_imp[compt]/N
    err_exp[compt]=err_exp[compt]/N
    err_cn[compt]=err_cn[compt]/N
    
    compt=compt+1
    
    

print(err_imp)
print(err_exp)
print(err_cn)

plt.plot(listeN,err_imp, label='erreur relative - implicit')
plt.plot(listeN,err_exp, label='erreur relative - explicit')
plt.plot(listeN,err_cn,label='erreur relative - Cranck-Nicolson')
plt.legend()
plt.show()

