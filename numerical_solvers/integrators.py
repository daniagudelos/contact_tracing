#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:38:17 2020

@author: saitel
"""
import numpy as np


def trapezoidalRule(f, a, t_0):
    """

    Parameters
    ----------
    f : function.
    a : last step calculated.

    Returns
    -------
    float
        a step of the trapezodial rule

    """
    if a == 0:
        return 0
    h = 0.05
    n = (a) // h
    s = 0.5 * (f(a, a, t_0) + f(a, 0, t_0))
    for i in np.arange(0, n + h, 1):
        s = s + f(a, a + i * h, t_0)
    return h * s