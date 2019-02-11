#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.clf()
plt.close('all')

y = np.linspace(-4,4,1000)

lam0 = np.float(1e-1)
lam1 = np.float(1)
lam2 = np.float(1e1)

###############################################################################
f1 = np.max([np.zeros_like(y), 1.-y], axis=0)

def fy1(lam):
    fy1_lam = np.zeros_like(y)
    for i, yi in enumerate(y):
        if yi < 1-lam:
            fy1_lam[i] = 1. - yi - 0.5*lam
        if yi >= 1-lam and yi < 1:
            fy1_lam[i] = 1./(lam*2.)*(yi-1)**2
        if yi >= 1:       
            fy1_lam[i] = 0
    
    return fy1_lam

fy1_lam0 = fy1(lam0)    
fy1_lam1 = fy1(lam1) 
fy1_lam2 = fy1(lam2) 

###############################################################################
f2 = np.abs(y) + (np.abs(y)**3) / 3. 

def fy2(lam):
    
    fy2_lam = np.zeros_like(y)

    def tmp_plus(yii):
        return -(1.0/lam)/2.0 + (((1.0/lam)/2.)**2. - (1.0 - (yii/lam)))**0.5

    def tmp_minus(yii):
        return -(-1.0/lam)/2. - (((-1.0/lam)/2.)**2. - (1.0 + (yi/lam)))**0.5

    for i, yi in enumerate(y):
        if yi > lam:
            y1 = tmp_plus(yi)
            fy2_lam[i] = y1 + (y1**3.0)/3.0 + (np.linalg.norm(yi - y1)**2.)/(2.*lam)
        elif -lam <= yi and yi <= lam:
            fy2_lam[i] = (yi**2.)/(2.*lam)
        elif yi < lam:
            y1 = tmp_minus(yi)
            fy2_lam[i] = -y1 + ((-y1)**3.0)/3.0 + (np.linalg.norm(yi - y1)**2.)/(2.*lam)
            
    return fy2_lam

fy2_lam0 = fy2(lam0)
fy2_lam1 = fy2(lam1)
fy2_lam2 = fy2(lam2)

###############################################################################
f3 = np.zeros_like(y)
inf = 20.
a = -2.
b = 2.
for i, yi in enumerate(y):
  if yi >= a and yi <= b:
    f3[i] = 0
  else:
    f3[i] = inf
  
def fy3(lam):
    fy3_lam = np.zeros_like(y)
    
    frac_l = 1./(2.*lam)
    for i, yi in enumerate(y):
        if yi < a:
            fy3_lam[i] = frac_l * np.linalg.norm(yi-a)**2
        if yi >= a and yi <= b:
            fy3_lam[i] = 0
        if yi > b:      
            fy3_lam[i] = frac_l * np.linalg.norm(yi-b)**2
        
    return fy3_lam

fy3_lam0 = fy3(lam0)
fy3_lam1 = fy3(lam1)
fy3_lam2 = fy3(lam2)


###############################################################################
f4 = np.max([np.abs(y), (np.abs(y)**2)],axis=0) 
def fy4(lam):
    fy4_lam = np.zeros_like(y)
    
    lam2 =  (2*lam)
    def f(yi, lam):
        (yi/(2*lam+1))**2 + np.linalg.norm(y[i]*(1-(1/(2*lam+1))))**2 / lam2
    
    
    for i, yi in enumerate(y):
        if yi >= 2.*lam + 1:
            fy4_lam[i] = f(yi,lam)
        if yi <= -2*lam - 1:
            fy4_lam[i] = f(yi,lam)
        if yi < 1 + 2*lam and yi >= 1+lam:      
            fy4_lam[i] = 1 + np.linalg.norm(yi - 1)**2 / lam2
        if yi > -1 - 2*lam and yi <= -1-lam:      
            fy4_lam[i] = 1 + np.linalg.norm(yi + 1)**2 / lam2
        if yi < 1+ lam and yi >= lam:
            fy4_lam[i] = np.abs(y[i]-lam) + lam/2.
        if yi > -1- lam and yi <= -lam:
            fy4_lam[i] = np.abs(y[i]+lam) + lam/2.
        if yi < lam and yi > -lam:
            fy4_lam[i] = np.linalg.norm(yi)**2 / lam2
    
    return fy4_lam

fy4_lam0 = fy4(lam0)
fy4_lam1 = fy4(lam1)
fy4_lam2 = fy4(lam2)


def plot_functions(fyx_lambda_x, fct_name):
    plt.figure()
    plt.title(fct_name)
    plt.plot(y, fyx_lambda_x[0], label=fct_name)
    plt.plot(y, fyx_lambda_x[1], '--', label='lambda=0.1')
    plt.plot(y, fyx_lambda_x[2], '--', label='lambda=1')
    plt.plot(y, fyx_lambda_x[3], '--', label='lambda=10')
    plt.xlabel("y")
    plt.grid()
    plt.legend()
    plt.savefig('ass3_description/'+fct_name+'.png')
    plt.show()
    
    
plot_functions([f1, fy1_lam0, fy1_lam1, fy1_lam2], 'f1(y)')
plot_functions([f2, fy2_lam0, fy2_lam1, fy2_lam2], 'f2(y)')
plot_functions([f3, fy3_lam0, fy3_lam1, fy3_lam2], 'f3(y)')
plot_functions([f4, fy4_lam0, fy4_lam1, fy4_lam2], 'f4(y)')
