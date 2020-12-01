#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:54:35 2020

@author: saitel
"""
import numpy as np
from numerical_solvers.integrators import trapezoidalRule


class OneTimeBCT:
    def __init__(self, parameters):
        self.beta = parameters.beta
        self.mu = parameters.mu
        self.sigma = parameters.sigma
        self.p = parameters.p
        self.h = parameters.h

    def to_integrate(self, a, b, t_0):
        return self.beta(a - b, t_0 + a - b) * self.sigma(b, t_0 + a)

    def dkappa(self, a, t_0, previous_kappa):
        """
        Parameters
        ----------
        a : last step calculated
        t_0 : time of infection for the cohort.
    
        Returns
        -------
        float
            the derivative of the probability of infection at the next age of 
            infection.
    
        """
        integral_value = trapezoidalRule(self.to_integrate, a, t_0)
        return -previous_kappa * (self.mu(a) + self.sigma(a, t_0 + a) +
                                  self.p() * integral_value)

    def calculate_kappa(self, a, t_0):
    
        kappa = np.zeros((len(t_0), len(a)))
        count = 0
        total = len(t_0) * len(a)
        for j in range(0, len(t_0)):
            kappa[j][0] = 1
            for i in range(1, len(a)):
                count = count + 1
                print("Progress ", count / total * 100)
                kappa[j][i] = kappa[j][i-1] + self.h() * self.dkappa(a[i-1], t_0[j],
                                                           kappa[j][i-1])
        return kappa
