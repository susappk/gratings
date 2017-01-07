# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:15:43 2016

@author: teddy
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
p=1#the number of P
lambda0=1
d=1.41*lambda0
theta=20/180*np.pi+0j
mode='tm'
#################### #periodic structure
h=np.array([0.1*d,0.1*d])
epsilon=np.array([1.45**2,1.6**2,1.0**2])
medianumber=np.size(epsilon)
media=np.arange((medianumber))
ep=np.array([0,0.25*lambda0])
layernumber=np.size(ep)
layer=np.arange((layernumber))
yp=np.array([0,0.25*lambda0])
#####################
M=31
yitaT_2=np.arange((M),dtype=complex)
yitaT_1=np.arange((M),dtype=complex)
yitaT_0=np.arange((M),dtype=complex)
yitaT1=np.arange((M),dtype=complex)
yitaR_1=np.arange((M),dtype=complex)
yitaR_0=np.arange((M),dtype=complex)
x1=np.arange(M)
for x in np.arange(M):#truncation parameter
    N=5+x*2
    order=np.linspace(-N,N,2*N+1)
    #####################
    k0=2*np.pi/lambda0
    alpha0=k0*np.sin(theta)+0j
    k=k0*np.sqrt(epsilon)+0j#epsilon(p=0,1,2)
    alpha=alpha0+order*2*np.pi/d
    #theta orders
    thetaR=np.arcsin(alpha/k[2])/np.pi*180
    thetaT=np.arcsin(alpha/k[0])/np.pi*180
    #matrix
    beta=np.zeros((medianumber,2*N+1),dtype=complex)
    phi=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)#phi matrix(3D(p,2Dphimatrix))
    w=np.zeros((medianumber,layernumber,4*N+2,4*N+2),dtype=complex)#w matrix(4D(p,q,2dwmatrix))
    t=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)
    deltan0=np.zeros((2*N+1),dtype=complex)
    #function of L
    def L(m,X,hp):
        return np.exp(1j*X*hp/2)*special.iv(m,-1j*X*hp/2)
    #end 
    #beta matrix
    for i in media:
        for n in order:
            beta[i,n+N]=np.sqrt(k[i]**2-alpha[n+N]**2)
    #phi matrix
    for j in layer:
        for n in order: 
            phi[j,n+N,n+N]=np.exp(1j*beta[j,n+N]*ep[j]);
            phi[j,n+3*N+1,n+3*N+1]=np.exp(-1j*beta[j,n+N]*ep[j]);
    #w matrix
    for i in media:
        for j in layer:        
            for n in order:            
                for m in np.linspace(-N,N,2*N+1):
                    #modules11
                    w[i,j,m+N,n+N]=L(m-n,beta[i,n+N],h[j])
                    #modules12
                    w[i,j,m+N,n+3*N+1]=L(m-n,-beta[i,n+N],h[j])
                    #modules21
                    if mode == 'tm' :
                        w[i,j,m+3*N+1,n+N]=(k[i]**2-alpha[m+N]*alpha[n+N])/epsilon[i]/beta[i,n+N]*L(m-n,beta[i,n+N],h[j])
                    elif mode == 'te' :
                        w[i,j,m+3*N+1,n+N]=(k[i]**2-alpha[m+N]*alpha[n+N])/beta[i,n+N]*L(m-n,beta[i,n+N],h[j])
                    #modules22
                    if mode == 'tm' :
                        w[i,j,m+3*N+1,n+3*N+1]=-(k[i]**2-alpha[m+N]*alpha[n+N])/epsilon[i]/beta[i,n+N]*L(m-n,-beta[i,n+N],h[j])
                    elif mode == 'te' :
                        w[i,j,m+3*N+1,n+3*N+1]=-(k[i]**2-alpha[m+N]*alpha[n+N])/beta[i,n+N]*L(m-n,-beta[i,n+N],h[j])
    #T matrix
    for j in layer:
        t[j,:,:]=np.dot(np.linalg.inv(w[j+1,j,:,:]),w[j,j,:,:])
        t[j,:,:]=np.dot(t[j,:,:],phi[j,:,:])
    T=np.dot(t[1,:,:],t[0,:,:])
    #end T matrix
    deltan0[N]=1
    T22=np.linalg.inv(T[(2*N+1):,(2*N+1):])
    HT=np.dot(T22,np.exp(-1j*beta[p+1,N]*yp[1])*deltan0)
    T12=(T[:(2*N+1),(2*N+1):])
    HR=np.dot(T12,T22)
    HR=np.exp(-1j*beta[p+1,:]*yp[1])*np.dot(HR,np.exp(-1j*beta[p+1,N]*yp[1])*deltan0)
    if mode == 'tm':
        yitaT=np.abs(HT)**2*(epsilon[p+1]*beta[0,:]/epsilon[0]/beta[p+1,N])
        yitaR=np.abs(HR)**2*(beta[p+1,:]/beta[p+1,N])
    elif mode == 'te':
        yitaT=np.abs(HT)**2*(beta[0,:]/beta[p+1,N])
        yitaR=np.abs(HR)**2*(beta[p+1,:]/beta[p+1,N])
    yitaT_2[x]=yitaT[N-2]
    yitaT_1[x]=yitaT[N-1]
    yitaT_0[x]=yitaT[N]
    yitaT1[x]=yitaT[N+1]
    yitaR_1[x]=yitaR[N-1]
    yitaR_0[x]=yitaR[N]
plt.figure(1,figsize=(10,12))
ax1 = plt.subplot(321) # 在图表2中创建子图1
plt.sca(ax1)   # 选择图表2的子图1
plt.plot(5+x1*2,np.real(yitaT_2),label="$ -2 order$",color="red",linewidth=2)
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM transmistted")
plt.legend(loc = 'upper left')
ax2 = plt.subplot(322) # 在图表2中创建子图2
plt.sca(ax2)  # 选择图表2的子图2
plt.plot(5+x1*2,np.real(yitaT_1),"b--",label="$-1 order$")
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM transmistted")
plt.legend(loc = 'upper left')
ax3 = plt.subplot(323) # 在图表2中创建子图2
plt.sca(ax3)  # 选择图表2的子图2
plt.plot(5+x1*2,np.real(yitaT_0),"b--",label="$0 order$")
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM transmistted")
plt.legend(loc = 'upper left')
ax4 = plt.subplot(324) # 在图表2中创建子图2
plt.sca(ax4)  # 选择图表2的子图2
plt.plot(5+x1*2,np.real(yitaT1),"b--",label="$1 order$")
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM transmistted")
plt.legend(loc = 'upper left')
#####
ax5 = plt.subplot(325) # 在图表2中创建子图1
plt.sca(ax5)   # 选择图表2的子图1
plt.plot(5+x1*2,np.real(yitaR_1),label="$ 0 order$",color="red",linewidth=2)
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM Reflacted")
plt.legend(loc = 'upper left')
ax6 = plt.subplot(326) # 在图表2中创建子图2
plt.sca(ax6)  # 选择图表2的子图2
plt.plot(5+x1*2,np.real(yitaR_0),"b--",label="$-1 order$")
plt.xlabel("N")
plt.ylabel("\yita")
plt.title("TM Reflacted")
plt.legend(loc = 'upper left')
plt.show()