## Projet Option Pricing Finite Difference - Option barrière

## Exemple pour une Up-and-Out

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
alpha=0.3 # cond de stabilité : alpha<=0.5 (alpha=dt/(dx*2))
N=451 #nombre de pts spatiaux
dx=(Lmax-Lmin)/(N-1)
tmax=0.5*sigma*sigma*T ; k=r/(0.5*sigma*sigma)
a=(1-k)/2
b=a*a+(k-1)*a-k
#-----------------------------------------------------------------------------------------------------------------
somme_b=np.zeros(N)
def BSS(S,K,sigma,r,T):
    d1=(math.log(S/K)+(r+1/2*sigma**2)*(T))/(sigma*math.sqrt(T))      ##########BLACK-SCHOLES call
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(norm.cdf(d1,0,1)*S-K*math.exp(-r*T)*norm.cdf(d2,0,1))
    
def PBS(S,K,sigma,r,T):
    d1=(math.log(S/K)+(r+1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(-norm.cdf(-d1,0,1)*S+K*math.exp(-r*T)*norm.cdf(-d2,0,1))
    
def blackdigit(S,K,sigma,r,T):
    d2=(math.log(S/K)+(r-1/2*sigma**2)*(T))/(sigma*math.sqrt(T))
    return(math.exp(-r*T)*norm.cdf(d2,0,1))
    
def BlackUpIn(S,K,H,sigma,r,T):
    #petitk=r/(sigma**2)*2
    #a=(1-petitk)/2
    return((H/S)**((2*r-sigma**2)/(sigma**2))*(PBS(H**2/S,K,sigma,r,T)-PBS(H**2/S,H,sigma,r,T)+(H-K)*math.exp(-r*T)*norm.cdf(-((math.log(H/S)+(r-sigma**2/2))/(sigma*math.sqrt(T))),0,1))+BSS(S,H,sigma,r,T)+(H-K)*math.exp(-r*T)*norm.cdf((math.log(S/H)+(r-sigma**2/2))/(sigma*math.sqrt(T)),0,1))


#-----------------------------------------------------------------------------------------------------------------
fig=plt.figure()
fig.suptitle('Option pricing / FDM / Euler Explicit / Barriers')
#ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 1, 1)
#ax3 = fig.add_subplot(1, 3, 3, projection='3d')

#ax2.set_xlim([0,71])
#ax2.set_ylim([0,31]) 
ax2.set_xlim([0,75])
ax2.set_ylim([0,35])


#ax1.set_ylim([0,0.9])
#ax3.set_zlim([0,31])
#ax3.set_xlim([0,71])

#ax1.set_title('u as a fct of x')
#ax2.set_title('C as a fct of S')
#ax3.set_title('C as a fct of x and t')

# ax3.set_xlabel('Stock Price')
# ax3.set_ylabel('tho')
# ax3.set_zlabel('Option Price')
# ax1.set_xlabel('x')
# ax1.set_ylabel('u')
# ax2.set_xlabel('S')
# ax2.set_ylabel('C')

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
    
BSS_list=np.zeros(N)
for i in range(N):
    BSS_list[i]=BSS(s[i],K,sigma,r,T)
    
BlackUpIn_list=np.zeros(N)
for i in range(N):
    BlackUpIn_list[i]=BlackUpOut(s[i],K,70,sigma,r,T)

BlackUpOut_list=np.zeros(N)
for i in range(N):
    BlackUpOut_list[i]=BSS_list[i]-BlackUpIn_list[i]
    
    
    
ax2.plot(s,BSS_list,color='powderblue',label='Exact Sol. B&S (Vanilla)')
ax2.legend()

ax2.plot(s,BlackUpIn_list,'grey', label='Exact Sol. B&S (Up-And-In)')
ax2.legend()

ax2.plot(s,BlackUpOut_list,color='sandybrown',label='Exact Sol. B&S (Up-And-Out)')
ax2.legend()
    
#ax1.plot(x,u,'b:o',linewidth=1,markersize=2)
#ax2.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T (Up-And-Out')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3.scatter3D(xdata, ydata, zdata)
#ax2.legend()
#fig.show()
#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,tmax,alpha*dx*dx)
#C=np.zeros(N) #N valeurs, dernière indice N-1
Z=np.zeros((len(temps),len(x)))

print(zdata)
Z[0]=zdata
#print(Z[0])
compteur=1

#plt.clf()
for t in temps[1:]:
    u[1:N-1]=u[1:N-1]+alpha*(u[0:N-2]-2*u[1:N-1]+u[2:N])
    u[0]=math.exp(Lmin)*math.exp(-(a*Lmin+b*t))
    u[0]=0
    u[N-1]=0
    
    #on revient sur C
    for i in range(N):
        C[i]=K*u[i]*math.exp(a*x[i]+b*t)
    
        
    Z[compteur]=C
    
    if t==temps[len(temps)-1]:
        xdata=np.full(N,t)
        ydata=s
        zdata=C
        #ax3.scatter3D(xdata, ydata, zdata)
        #ax1.plot(x,u,'r:o',linewidth=1,markersize=2)
        ax2.plot(s,C,'ro',linewidth=1,markersize=1.3, label='FDM Sol. (Up-And-Out)')
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
        #ax1.plot(x,u,color='grey',marker='',linestyle='dashed',linewidth=1)
        #ax2.plot(s,C,color='grey',marker='',linestyle='dashed',linewidth=1)
        #fig.suptitle(t/tmax)
        #plt.pause(0.1)    
        #plt.show()
        #fig.show()
        
    compteur=compteur+1
    
for i in range(N):
    somme_b[i]=C[i]
X, Y = np.meshgrid(s, temps)
#ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig.show()



## Projet Option Pricing Finite Difference - Option barrière

## Exemple pour une Up-and-In

## Schéma d Euler explicite


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
print(somme_b)
for i in range(N):
    C[i]=K*u[i]*math.exp(a*x[i])
s=np.zeros(N)
for j in range(N):
    s[j]=K*math.exp(x[j])
#ax1.plot(x,u,'b:o',linewidth=1,markersize=3)
#ax2.plot(s,C,'b:o',linewidth=1,markersize=3,label='prix à t=T (Up-And-In)')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3.scatter3D(xdata, ydata, zdata)
#ax2.legend()
#fig.show()
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
    
    u[N-1]=BSS(B,K,sigma,r,t*2/sigma/sigma)*1/K*math.exp(-(a*Lmax+b*t))
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
        #ax1.plot(x,u,'r:o',linewidth=1,markersize=3)
        ax2.plot(s,C,'go',linewidth=1,markersize=1.3, label='FDM Sol. (Up-And-In)')
        ax2.legend()
        #fig.suptitle(t/tmax)
        plt.pause(0.1)
        #plt.show()
        fig.show()
    # elif t==temps[int(len(temps)/2)]:
    #     #ax1.plot(x,u,'k:o',linewidth=1,markersize=3)
    #     ax2.plot(s,C,'k:o',linewidth=1,markersize=3, label='prix à t=T/2')
    #     ax2.legend()
    #     plt.pause(0.1)
    # elif t==temps[int(len(temps)/4)]:
    #     #ax1.plot(x,u,'y:o',linewidth=1,markersize=3)
    #     ax2.plot(s,C,'y:o',linewidth=1,markersize=3, label='prix à t=T/4')
    #     ax2.legend()
    #     plt.pause(0.1)
    # elif t==temps[int(len(temps)/16)]:
    #     #ax1.plot(x,u,':o',linewidth=1,markersize=3)
    #     ax2.plot(s,C,':o',linewidth=1,markersize=3, label='prix à t=T/16')
    #     ax2.legend()
    #     plt.pause(0.1)
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

for i in range(N):
    somme_b[i]=somme_b[i]+C[i]
ax2.plot(s,somme_b,'ko',markersize=1.3,label='Sum of Barriers FDM Solutions')
ax2.legend()
fig.show()


X, Y = np.meshgrid(s, temps)
#ax3.plot_wireframe(X, Y, Z, color='SeaGreen')
#ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer', edgecolor='none')

fig.show()

## Et on trace la vanille avec la méthode numerique explicite

#on change le Lmax
Lmax=math.log(100/40)

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
#ax1.plot(x,u,'b:o',linewidth=1,markersize=2)
#ax2.plot(s,C,'b:o',linewidth=1,markersize=2,label='prix à t=T')
xdata=np.full(N,0)
ydata=s
zdata=np.copy(C)
#ax3.scatter3D(xdata, ydata, zdata)
#ax2.legend()
#fig.show()

# ça aplatie exponentiellement la courbe en gros
#plt.legend()
#plt.axis([-2,100,-4,100])
#plt.show()
#-----------------------------------------------------------------------------------------------------------------
temps=np.arange(0,tmax,alpha*dx*dx)
#C=np.zeros(N) #N valeurs, dernière indice N-1
Z=np.zeros((len(temps),len(x)))


Z[0]=zdata

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
        #ax1.plot(x,u,'r:o',linewidth=1,markersize=2)
        #ax2.plot(s,C,'r:o',linewidth=1,markersize=2, label='prix à t=0 de vanilla')
        #ax2.legend()
        #fig.suptitle(t/tmax)
        #plt.pause(0.1)
        #plt.show()
        #fig.show()
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
        fig.show()
        
    compteur=compteur+1
    


fig.show()

