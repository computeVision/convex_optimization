#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Fri May 11 19:00:56 2018

@author: jester
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

###############################################################################
# Gradient

interval1 = 5 ** -0.5
interval1_n = -interval1
interval2 = 2 * interval1
interval2_n = -interval2


def ax_plus_b(a, xk, b=-3):
    return a.dot(xk) + b


def initial_a():
    a1 = np.array([1, 0])
    a2 = np.array([interval1_n, interval2])
    a3 = np.array([interval1_n, interval2_n])
    return (a1, a2, a3)


def f1(xk):
    a = np.array([1, 0])
    return ax_plus_b(a, xk)


def f2(xk):
    a = np.array([interval1_n, interval2])
    return ax_plus_b(a, xk)


def f3(xk):
    a = np.array([interval1_n, interval2_n])
    return ax_plus_b(a, xk)


def f(xk):
    return np.max([f1(xk), f2(xk), f3(xk)], axis=0)


def g(xk):
    (a1, a2, a3) = initial_a()

    is_f1 = False
    is_f2 = False
    is_f3 = False

    if f(xk) == f1(xk):
        is_f1 = True

    if f(xk) == f2(xk):
        is_f2 = True

    if f(xk) == f3(xk):
        is_f3 = True

    if is_f1 and is_f2 and is_f3:
        return np.array([0, 0])
        # return np.array([1,interval2]) # choice 2
        # return np.array([interval1,interval2]) # choice 3

    if is_f1 and is_f2:
        return np.array([0, 0])
        # return np.array([1,interval2]) # choice 2
        # return np.array([interval1_n,interval2]) # choice 3

    if is_f1 and is_f3:
        return np.array([0, 0])
        # return np.array([1,0]) # choice 2
        # return np.array([interval1_n,interval2]) # choice 3

    if is_f2 and is_f3:
        return np.array([interval1_n, 0])
        # return np.array([interval1_n,interval2]) # choice 2
        # return np.array([interval1_n,0]) # choice 3

    if is_f3:
        return a3

    if is_f2:
        return a2

    if is_f1:
        return a1
    else:
        raw_input('should not be reached')


###############################################################################
# Evaluation

##### setup
epochs = int(1e5 * 2.5)
xk = np.array([8, 8])  # initial value
xk_list = np.array([xk])

# constant stepsize
tk = 1e-3

# diminisching stepsize
beta = 5
gamma = np.sqrt(beta)

# dynamic stepsize
f_star = -3
f_star_list = np.array([])

print ('initial value: ', xk)
for k in range(epochs):
    tmp = f(xk) - f_star

    #diminishing stepsize
    # tk = beta/(k+gamma)

    #dynamic stepsize
    tk = tmp / np.linalg.norm(g(xk) ** 2 + 1e-3)

    xk_new = xk - tk * g(xk)
    xk_list = np.append(xk_list, xk_new)
    xk = xk_new

    if tmp < 1e-3:
        print 'stopping criteria'
        epochs = k
        break
    f_star_list = np.append(f_star_list, tmp)


###############################################################################
# Plot Contour
# idea from:  https://plot.ly/matlab/contour-plots/ and
# https://matplotlib.org/examples/pylab_examples/contour_demo.html

print 'Plot Contour ...'

nr_points = 500
for_f1 = np.zeros((nr_points, nr_points))
for_f2 = np.zeros((nr_points, nr_points))
for_f3 = np.zeros((nr_points, nr_points))

x1 = np.linspace(-10, 10, nr_points)
x2 = np.linspace(-10, 10, nr_points)
(a1, a2, a3) = initial_a()
b = -3.

for (i, i_val) in enumerate(x1):
    for (j, j_val) in enumerate(x2):
        tmp = np.array([i_val, j_val])
        for_f1[j, i] = a1.dot(tmp) + b
        for_f2[j, i] = a2.dot(tmp) + b
        for_f3[j, i] = a3.dot(tmp) + b

f_max = np.max([for_f1, for_f2, for_f3], axis=0)

print '... finished'


plt.figure(10)
plt.plot(xk_list[::2], xk_list[1::2], 'b.-')
(X1, X2) = np.meshgrid(x1, x2)
plt.clabel(plt.contour(X1, X2, f_max), inline=True, fontsize=6)
plt.title('f(x)')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.colorbar()
plt.show()

print ('reached value: ', xk)

###############################################################################
# Plot Convergence

plt.figure(20)
plt.plot(np.arange(epochs), f_star_list, 'b')
plt.xlabel('Epochs')
plt.ylabel('f-f*')
plt.title('Convergence')
plt.show()
