## Projet Option Pricing Finite Difference - Option barrière

## Exemple pour une Up-and-In

## Schéma d Euler explicite

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm
#-----------------------------------------------------------------------------------------------------------------
K=40 ; r=0.05 ; T=1 ; sigma=0.3 #vol
Lmax=math.log(70/40) ; Lmin=-6 # pour une barrière à 70
B=70
alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=301 #nombre de pts spatiaux
dx=(Lmax-Lmin)/(N-1)
tmax=0.5*sigma*sigma*T ; k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
fig=plt.figure()
fig.suptitle('Option pricing / FDM / Euler Explicit / Up-And-In Call')
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')

ax2.set_xlim([0,71])
ax2.set_ylim([0,31]) 
ax1.set_ylim([0,0.9])
ax3.set_zlim([0,31])
ax3.set_xlim([0,71])

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
# def u0(x):   #fct pour condition initiale
#     if math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)>=0:
#         return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
#     else:
#         return 0 
def u0(x): 
    if x==Lmax:
        return math.exp(0.5*(k+1)*x)-math.exp(0.5*(k-1)*x)
    else:
        return 0

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
ax1.plot(x,u,'b:o',linewidth=1,markersize=3)
ax2.plot(s,C,'b:o',linewidth=1,markersize=3,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3.scatter3D(xdata, ydata, zdata)
ax2.legend()
fig.show()
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
    #u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*t))
    u[0]=0
    
    d1_B_K=(math.log(B/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
    d2_B_K=(math.log(B/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
    Pcallvanilla_B_K=B*norm.cdf(d1_B_K,0,1)-K*math.exp(-r*T)*norm.cdf(d2_B_K,0,1)
    print("prix BS")
    print(Pcallvanilla_B_K)
    
    u[N-1]=BSS(B,K,sigma,r,T-t)*1/K*math.exp(-(a*Lmax+b*t))
    
    #u[N-1]=0
    
    #on revient sur C
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i]+b*t)
        
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        ax1.plot(x,u,'r:o',linewidth=1,markersize=3)
        ax2.plot(s,C,'r:o',linewidth=1,markersize=3, label='prix à t=0')
        ax2.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig.show()
    elif t==temps[int(len(temps)/2)]:
        ax1.plot(x,u,'k:o',linewidth=1,markersize=3)
        ax2.plot(s,C,'k:o',linewidth=1,markersize=3, label='prix à t=T/2')
        ax2.legend()
        plt.pause(0.1)
    elif t==temps[int(len(temps)/4)]:
        ax1.plot(x,u,'y:o',linewidth=1,markersize=3)
        ax2.plot(s,C,'y:o',linewidth=1,markersize=3, label='prix à t=T/4')
        ax2.legend()
        plt.pause(0.1)
    elif t==temps[int(len(temps)/16)]:
        ax1.plot(x,u,':o',linewidth=1,markersize=3)
        ax2.plot(s,C,':o',linewidth=1,markersize=3, label='prix à t=T/16')
        ax2.legend()
        plt.pause(0.1)
    else:
        xdata=np.full(N,t)
        ydata=s
        zdata=np.copy(C)
        #ax3.scatter3D(xdata, ydata, zdata)
        #ax1.plot(x,u,color='grey',marker='',linestyle='dashed',linewidth=1)
        #ax2.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        #plt.pause(0.1)    
        #plt.show()
        #fig.show()
        
    compteur=compteur+1
    
X, Y = np.meshgrid(s, temps)
ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig.show()

## Erreur relative avec les 3 schémas


listeN=np.arange(100,300,10)
err_imp=np.zeros(len(listeN))
err_exp=np.zeros(len(listeN))
err_cn=np.zeros(len(listeN))

compt=0

for elt in listeN:
    K=40 ; r=0.05 ; sigma=0.3 # vol
    T=1 #maturité de l option, unité ?
    Lmax=math.log(70/40) ; Lmin=-6 ; alpha=0.5 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
    B=70
    N=elt #nombre de pts spatiaux
    dx=(Lmax-Lmin)/(N-1)
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
    temps=np.arange(0,tmax,alpha*dx*dx)
    for t in temps[1:]:
        #calcul u pr implicit
        u[1:N-1]=u[1:N-1]+alpha*(u[0:N-2]-2*u[1:N-1]+u[2:N])
        u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*t))
        u[N-1]=0
        #calcul u pour explicit
        u_exp_plus_frontieres=np.copy(u_exp[1:N-1])
        u_exp_plus_frontieres[0]=u_exp_plus_frontieres[0]+alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
        u_exp_plus_frontieres[N-3]=u_exp_plus_frontieres[N-3]+alpha*0
        u_exp[1:N-1]=np.linalg.solve(A,u_exp_plus_frontieres)
        u_exp[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
        u_exp[N-1]=0
        #calcul u pour Cranck-Nicolson
        u_cn_plus_frontieres=(1-alpha)*u_cn[1:N-1]+0.5*alpha*u_cn[0:N-2]+0.5*alpha*u_cn[2:N]
        u_cn_plus_frontieres[0]=u_cn_plus_frontieres[0]+0.5*alpha*(math.exp(Lmin)*math.exp(-(a*Lmin+b*(t))))
        u_cn_plus_frontieres[N-3]=u_cn_plus_frontieres[N-3]+0.5*alpha*0
        u_cn[1:N-1]=np.linalg.solve(D,u_cn_plus_frontieres)
        u_cn[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*(t)))
        u_cn[N-1]=0
        
        for i in range(N):
            C[i]=K*u[i]*math.exp(a*x[i]+b*t)
            C_exp[i]=K*u_exp[i]*math.exp(a*x[i]+b*t)
            C_cn[i]=K*u_cn[i]*math.exp(a*x[i]+b*t)
            
    # calcul erreur relative pr implicit (pour le nb de coord spatiales de la boucle)
    for l in range(N):
        d1_S_K=(math.log(s[l]/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d2_S_K=(math.log(s[l]/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d1_S_B=(math.log(s[l]/B)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d2_S_B=(math.log(s[l]/B)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d1_B2S_K=(math.log(B*B/s[l]/K)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d2_B2S_K=(math.log(B*B/s[l]/K)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d1_B2S_B=(math.log(B*B/s[l]/B)+(r+sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        d2_B2S_B=(math.log(B*B/s[l]/B)+(r-sigma*sigma*0.5)*T)/(sigma*math.sqrt(T))
        
        Pcallvanilla_S_K=s[l]*norm.cdf(d1_S_K,0,1)-K*math.exp(-r*T)*norm.cdf(d2_S_K,0,1)
        Pcallvanilla_S_B=s[l]*norm.cdf(d1_S_B,0,1)-B*math.exp(-r*T)*norm.cdf(d2_S_B,0,1)
        Pcalldigit_S_B=math.exp(-r*T)*norm.cdf(d2_S_B)
        Pcallvanilla_B2S_K=B*B/s[l]*norm.cdf(d1_B2S_K,0,1)-K*math.exp(-r*T)*norm.cdf(d2_B2S_K,0,1)
        Pcallvanilla_B2S_B=B*B/s[l]*norm.cdf(d1_B2S_B,0,1)-B*math.exp(-r*T)*norm.cdf(d2_B2S_B,0,1)
        Pcalldigit_B2S_B=math.exp(-r*T)*norm.cdf(d2_B2S_B)
        
        Pcall_up_and_out=Pcallvanilla_S_K-Pcallvanilla_S_B-(B-K)*Pcalldigit_S_B-(s[l]/B)**(2*a)*(Pcallvanilla_B2S_K-Pcallvanilla_B2S_B-(B-K)*Pcalldigit_B2S_B)
        #Pcall_up_and_out=Pcallvanilla_S_K-Pcallvanilla_S_B-(B-K)*Pcalldigit_S_B
        
        err_imp[compt]=err_imp[compt]+1/N*(C[l]-Pcall_up_and_out)**2
        err_exp[compt]=err_exp[compt]+1/N*(C_exp[l]-Pcall_up_and_out)**2
        err_cn[compt]=err_cn[compt]+1/N*(C_cn[l]-Pcall_up_and_out)**2
        
        
    compt=compt+1
    
    
print(err_imp)
print(err_exp)
print(err_cn)

plt.plot(listeN,err_imp, label='erreur relative - implicit')
plt.plot(listeN,err_exp, label='erreur relative - explicit')
plt.plot(listeN,err_cn,label='erreur relative - Cranck-Nicolson')
plt.legend()
plt.title('FDM Heat Eqt - Barrier - Error relative to BS price')
plt.xlabel('Number of points in space')
plt.ylabel('Error relative to the BS price')
plt.show()
