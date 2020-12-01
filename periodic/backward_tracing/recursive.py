#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:54:35 2020

@author: saitel
"""


def to_integrate(a, b, t_0):
    if a == 0:
        return 0
    return beta(a - b, t_0 + a - b) * (dkappa(b, t_0 + a - b) -
                                       kappa(b, t_0 + a - b))

def dkappa(a, t_0, previous_kappa):
        """
    
        Parameters
        ----------
        next_a : step to be calculated.
        t_0 : time of infection for the cohort.
    
        Returns
        -------
        float
            the probability of infection at age of infection next_a.
    
        """
    
       # a = next_a - h()
    
       # if next_a == 0:
       #     return mu(a) + sigma(a, t_0 + a)
    
        integral_value = trapezoidalRule(to_integrate, a, t_0)
        return -previous_kappa * (mu(a) + sigma(a, t_0 + a) + p() * integral_value)