#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.clf()
plt.close('all')

x = np.linspace(-4,4,1000)
inf = 20
lam = 1

###############################################################################
# d

fd = np.zeros_like(x)
a = 3
tmp = 4*a*lam*np.ones_like(x)
fd_prox = (x+np.sqrt(x**2 + 2*tmp))/(2*(np.ones_like(x)+lam*np.ones_like(x)))

for i, xi in enumerate(x):
  if xi > 0:
    fd[i] = -a*np.log(xi) + 0.5*xi**2
  else:
    fd[i] = inf

###############################################################################
# e
a = 1.
fe = np.max([np.abs(x) - a, np.zeros_like(x)], axis=0) 
fe_prox = np.zeros_like(x)

for i, xi in enumerate(x):
    if xi <= -a-lam:
        fe_prox[i] = xi+lam
    if xi >= a+lam:
        fe_prox[i] = xi-lam
    if xi >= -a and xi <= a:
        fe_prox[i] = xi
    if xi < -a and xi > -a -lam:
        fe_prox[i] = -a
    if xi > a and xi < a + lam:
        fe_prox[i] = a
    
###############################################################################
# f
a = 1.0
ff = np.zeros_like(x)
ff_prox = np.zeros_like(x)

for i, xi in enumerate(x):
    if np.abs(xi) < a:
        ff[i] = np.log(a) - np.log(a - np.abs(xi))
    else:
        ff[i] = inf
    
for i, xi in enumerate(x):
    if xi > (lam / a):
        tmp_i = (a+xi)/2.0
        tmp_j = a*xi - lam
        ff_prox[i] = tmp_i -(tmp_i**2.0 - tmp_j)**0.5
    elif xi < (-lam / a):
        tmp_i = (a-xi)/2.0
        tmp_j = a*xi + lam
        ff_prox[i] = -tmp_i + (tmp_i**2.0 + tmp_j)**0.5
    elif (- lam / a) <= xi and xi <= (lam / a):
        ff_prox[i] = 0.

###############################################################################
# plots 
def plot_functions(f, fct_name):
    plt.figure()
    plt.title(fct_name)
    plt.plot(x, f[0], label=fct_name)
    plt.plot(x, f[1], '--', label='prox')
    plt.xlabel("x")
    plt.grid()
    plt.legend()
    plt.savefig('ass3_description/'+fct_name+'.png')
    plt.show()
    
plot_functions([fd, fd_prox], 'f_d(x)')
plot_functions([fe, fe_prox], 'f_e(x)')
plot_functions([ff, ff_prox], 'f_f(x)')
