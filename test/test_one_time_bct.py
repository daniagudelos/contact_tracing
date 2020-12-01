#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:38:14 2020

@author: saitel
"""
import numpy as np
from parameters.constant_parameters import h
from matplotlib import pyplot as plt
from matplotlib import cm
from periodic.backward_tracing import one_time as bct

a_max = 5

t_0 = np.arange(0, a_max + h(), h())
a = np.arange(0, a_max + h(), h())

kappa = bct.calculate_kappa(a, t_0)

X, Y = np.meshgrid(t_0, a)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel('time of infection')
ax.set_ylabel('age of infection')
ax.set_zlabel('probability of infection')

ax.plot_surface(X, Y, kappa, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.savefig('prob_infection_one_time_bct_constant_pars.png')
