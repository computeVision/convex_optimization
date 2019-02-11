#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)

#plt.clf()
plt.close('all')
PLOT = False
LOG = True

B = np.load('mnist_training_images.npy') / 255.
D = np.load('D16.npy'); Dim = "16"
D = np.load('D32.npy'); Dim = "32"
D = np.load('D64.npy'); Dim = "64"
D = np.load('D128.npy'); Dim = "128"

if PLOT:
    if Dim == "16":
        C_I = np.load('C_ISTA_16.npy')
        C = np.load('C_FISTA_16.npy')
    elif Dim == "32":
        C_I = np.load('C_ISTA_32.npy')
        C = np.load('C_FISTA_32.npy')
    elif Dim == "64":
        C_I = np.load('C_ISTA_64.npy')
        C = np.load('C_FISTA_64.npy')
    elif Dim == "128":
        C_I = np.load('C_ISTA_128.npy')
        C = np.load('C_FISTA_128.npy')
    reconstr_I = D.dot(C_I)
    reconstr = D.dot(C)
    
#    plt.figure(20)
#    plt.imshow(reconstr[:,0].reshape(28,28))
    
    for i in xrange(5):
        plt.imsave("ass3_description/reconstr_fista_"+str(i)+"_"+Dim+".png",reconstr[:,i].reshape(28,28))
        plt.imsave("ass3_description/reconstr_ista_"+str(i)+"_"+Dim+".png",reconstr_I[:,i].reshape(28,28))
        # plt.imsave("ass3_description/orig_"+str(i)+"_"+Dim+".png",B[:,i].reshape(28,28))
        
    sys.exit()

     
energy_ista = []
energy_fista = []
L = np.linalg.norm(D.T.dot(D)) # Literature: https://arxiv.org/pdf/1501.02888.pdf
lam = 0.1
x0 = rng.rand(D.shape[1])
iters = 350



def E(C):
    return 0.5*np.linalg.norm((D.dot(C)-B).ravel())**2 + lam*np.linalg.norm(C.ravel(), 1)

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x)-lam/L, 0.0)

###############################################################################
# ISTA
C = rng.randn(D.shape[1], B.shape[1])
for _ in xrange(iters):
    grad_f = D.T.dot(D.dot(C) - B) / L
    C = soft_threshold(C - grad_f/L, lam)
    
    energy_ista.append(E(C))
    print len(energy_ista)
    
    # if LOG and len(energy_ista) >=2:
    #     print (energy_ista[-1] - energy_ista[-2])
    # if len(energy_ista) >= 2 and  np.abs(energy_ista[-1] - energy_ista[-2]) < 1e2:
    #     print "early stopping"
    #     break
    
np.save("C_ISTA_" + Dim +".npy", C)
np.save("E_ISTA_" + Dim +".npy", energy_ista)

###############################################################################
## FISTA
C = rng.randn(D.shape[1], B.shape[1])
x = np.zeros_like(C)
t = 1
for _ in xrange(iters):
    x_prev = x.copy()
    grad_f = D.T.dot(D.dot(C)-B)/L
    x = soft_threshold(C-grad_f/L, lam)
    t0 = t
    t = (1.+np.sqrt(1+4.*t**2))/2.
    C = x+((t0-1.)/t)*(x-x_prev)
    energy_fista.append(E(C))
    print len(energy_ista)
    # print "energy: ", energy_fista[-1]
    
    # if LOG and len(energy_fista) >=2:
    #     print (energy_fista[-1] - energy_fista[-2])
    
    # if len(energy_fista) >= 2 and (energy_fista[-1] - energy_fista[-2]) > 1e1:
    #     print "early stopping"
    #     break

np.save("C_FISTA_ " + Dim +".npy", C)
np.save("E_FISTA_ " + Dim +".npy", energy_fista)

plt.figure(10)
plt.title("Energy: E(C), dim="+Dim)
plt.ylabel("Energy")
plt.xlabel("Epochs")
plt.grid()
plt.plot(energy_ista, label="ISTA")
plt.plot(energy_fista, label="FISTA")
plt.legend()
plt.savefig('ass3_description/(F)ISTA_'+Dim+'.png')
plt.show()


#orig = B[:,0].reshape((28,28))

