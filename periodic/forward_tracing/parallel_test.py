#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:08:24 2021

@author: saitel
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


b = np.zeros((10, 10))
for t_0_index in range(0, 10):
    for a_index in range(0, 10):
        b[t_0_index, a_index] = t_0_index * a_index


def my_func(t_0_index):
    b2 = np.zeros((10))
    for a_index in range(0, 10):
        b2[a_index] = t_0_index * a_index
    return b2

b3 = np.zeros((10, 10))
# res =  my_func(5)
b3 = np.asarray(Parallel(n_jobs=4)(delayed(my_func)(i) for i in range(0, 10)))
