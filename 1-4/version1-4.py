# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:15:43 2016

@author: teddy
"""
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
def f(aa):     
    #import matplotlib.pyplot as plt
    p=1#the number of P
    lambda0=1
    d=1.41*lambda0
    theta=aa/180*np.pi+0j
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
    N=15 #truncation parameter
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
    def L(m,x,hp):
        return np.exp(1j*x*hp/2)*special.iv(m,-1j*x*hp/2)
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
    deltan0[15]=1
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
    return (yitaT,yitaR)
    
kk=150
thetakk=np.linspace(-89,89,kk)
yitaTt=np.zeros((kk,31),dtype=complex)
yitaRr=np.zeros((kk,31),dtype=complex)
for i in np.arange(kk):
    yitaTt[i,:]=f(thetakk[i])[0]
    yitaRr[i,:]=f(thetakk[i])[1]
plt.figure(figsize=(12,12))
thetakk=thetakk/180*np.pi
#plt.plot(np.sin(thetakk),np.real(yitaTt[:,15]),label="$T-0$",color="b")
#plt.plot(np.sin(thetakk),np.real(yitaRr[:,15]),label="$R-0$",color="g")
#plt.plot(np.sin(thetakk),np.real(yitaRr[:,14]),label="$R-1$",color="r")
plt.plot(np.sin(thetakk),np.real(yitaTt[:,14]),label="$T-1$",color="c")
#plt.plot(np.sin(thetakk),np.real(yitaTt[:,13]),label="$T-2$",color="m")
#plt.plot(np.sin(thetakk),np.real(yitaTt[:,16]),label="$T1$",color="y")
plt.xlabel("sin(theta)")
plt.ylabel("yita")
plt.title("TM")
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    