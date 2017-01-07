# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:15:43 2016

@author: teddy
"""
import numpy as np
from scipy import special
from scipy import linalg
#import matplotlib.pyplot as plt
p=1#the number of P
lambda0=1
d=1.3*lambda0
theta=30/180*np.pi+0j
mode='te'
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
lphi=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)#phi matrix(3D(p,2Dphimatrix))
rphi=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)
w=np.zeros((medianumber,layernumber,4*N+2,4*N+2),dtype=complex)#w matrix(4D(p,q,2dwmatrix))
t=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)
s=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)
S=np.zeros((layernumber,4*N+2,4*N+2),dtype=complex)
deltan0=np.zeros((2*N+1),dtype=complex)
#function of L
def L(m,x,hp):
    return np.exp(1j*x*hp/2)*special.iv(m,-1j*x*hp/2)
#end 
#beta matrix
for i in media:
    for n in order:
        beta[i,n+N]=np.sqrt(k[i]**2-alpha[n+N]**2)
#lphi &rphi matrix
for j in layer:
    for n in order: 
        lphi[j,n+N,n+N]=1
        lphi[j,n+3*N+1,n+3*N+1]=np.exp(-1j*beta[j,n+N]*ep[j])
        rphi[j,n+N,n+N]=np.exp(1j*beta[j,n+N]*ep[j])
        rphi[j,n+3*N+1,n+3*N+1]=1
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
#t matrix
for j in layer:
    t[j,:,:]=linalg.inv(w[j+1,j,:,:]).dot(w[j,j,:,:])
#t transform s
M=2*N+1
for j in layer:
    s[j,:M,:M]=t[j,:M,:M]-t[j,:M,M:].dot(linalg.inv(t[j,M:,M:]).dot(t[j,M:,:M]))#11
    s[j,:M,M:]=t[j,:M,M:].dot(linalg.inv(t[j,M:,M:]))#12
    s[j,M:,:M]=-linalg.inv(t[j,M:,M:]).dot(t[j,M:,:M])#21
    s[j,M:,M:]=linalg.inv(t[j,M:,M:])#22
    s[j,:,:]=lphi[j,:,:].dot(s[j,:,:]).dot(rphi[j,:,:])
#s transform S
S[0,:,:]=s[0,:,:]
S[1,:M,:M]=s[1,:M,:M].dot(linalg.inv(1-linalg.inv(s[0,:M,M:]).dot(s[1,M:,:M]))).dot(s[0,:M,:M])
S[1,:M,M:]=s[1,:M,M:]+s[1,:M,:M].dot(s[0,:M,M:]).dot(linalg.inv(1-s[1,M:,:M].dot(s[0,:M,M:]))).dot(s[1,M:,M:])
S[1,M:,:M]=s[0,M:,:M]+s[0,M:,M:].dot(s[1,M:,:M]).dot(linalg.inv(1-linalg.inv(S[0,:M,M:]).dot(s[1,M:,:M]))).dot(s[0,:M,:M])
S[1,M:,M:]=s[0,M:,M:].dot(linalg.inv(1-s[1,M:,:M].dot(s[0,:M,M:]))).dot(s[1,M:,M:])
#end T matrix
deltan0[N]=1
ht=S[1,M:,M:].dot(np.exp(-1j*beta[p+1,N]*yp[1])*deltan0)
hr=np.exp(-1j*beta[p+1,:]*yp[1])*S[1,:M,M:].dot(np.exp(-1j*beta[p+1,N]*yp[1])*deltan0)
if mode == 'tm':
    yitaT=np.abs(ht)**2*(epsilon[p+1]*beta[0,:]/epsilon[0]/beta[p+1,N])
    yitaR=np.abs(hr)**2*(beta[p+1,:]/beta[p+1,N])
elif mode == 'te':
    yitaT=np.abs(ht)**2*(beta[0,:]/beta[p+1,N])
    yitaR=np.abs(hr)**2*(beta[p+1,:]/beta[p+1,N])