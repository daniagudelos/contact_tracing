#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:49:44 2020

@author: saitel
"""
from math import sin, pi


class Parameters:
    pass


class ConstantParameters(Parameters):

    def __init__(self, beta=2.2, mu=1/3.47, sigma=1/3.47, p=0.5, h=0.005):
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.h = h

    def get_beta(self, a, t):
        return self.beta

    def get_dbeta(self, a, t):
        return 0

    def get_mu(self, a):
        return self.mu

    def get_sigma(self, a, t):
        return self.sigma

    def get_p(self):
        return self.p

    def get_h(self):
        return self.h


class VariableParameters(Parameters):
    def __init__(self, p=0.5, h=0.005):
        self.p = p
        self.h = h

    def get_beta1(self, a):
        """
        beta1 depends on a, and reflect the viral load behavior.

        Parameters
        ----------
        a : float.

        Returns
        -------
        float.

        """

        return  max(0, a/4 * (6-a))

    def get_beta2(self, t):
        """
        beta2 depends only on time periodically.
        
        Period: 1 day

        Parameters
        ----------
        t : float.

        Returns
        -------
        TYPE
            float.

        """

        return sin(2 * pi * t) + 1

    def get_beta(self, a, t):
        return self.get_beta1(a) * self.get_beta2(t)

    def get_dbeta(self, a, t):
        """
        Derivative of beta wrt a
        dbeta/da = dbeta1/da * beta2 + dbeta2/da * beta1
        dbeta/da = dbeta1/da * beta2 + 0

        Parameters
        ----------
        a : float.
        t : float.

        Returns
        -------
        TYPE
            float.

        """

        return max(0, (3 / 2 - a / 2) * sin(2 * pi * t) + 1)

    def get_mu(self, a):
        return max(0, a / 5 * (7 - a))

    def get_sigma1(self, a):
        return max(0, a / 4 * (7 - a))

    def get_sigma2(self, t):
        return sin(2 * pi * t) + 1

    def get_sigma(self, a, t):
        return self.get_sigma1(a) * self.get_sigma2(t)

    def get_p(self):
        return self.p

    def get_h(self):
        return self.h
