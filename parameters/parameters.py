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

    def __init__(self, beta=1.0, mu=0.3, sigma=0.5, p=0.5, h=0.05):
        """
        Initialize the set of constant parameters

        Parameters
        ----------
        beta : DOUBLE, optional
            DESCRIPTION. Contact rate. The default is 1.0 / unit of time.
            The value is multiplied by h to return 1.0 * h / fraction of time
        mu : DOUBLE, optional
            DESCRIPTION. Spontaneous recovery rate. The default is
            0.3 / unit of time. The value is multiplied by h to return
            0.3 * h / fraction of time
        sigma : DOUBLE, optional
            DESCRIPTION. Observed recovery. The default is 0.5 / unit of time.
            The value is multiplied by h to return 0.5 * h / fraction of time
        p : TYPE, optional
            DESCRIPTION. Probability of success for contact tracing. The
            default is 0.5.
        h : DOUBLE, optional
            DESCRIPTION. Discretization step (time). The default is 0.05.

        Returns
        -------
        None.

        """
        self.h = h
        self.beta = beta * self.h
        self.mu = mu * self.h
        self.sigma = sigma * self.h
        self.p = p

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
    def __init__(self, p=0.5, h=0.005, period=1):
        self.p = p
        self.h = h
        self.period = period
        self.period_length = int(round(period / h, 1))

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

        return max(0, a/4 * (6-a))

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

    def get_period(self):
        return self.period

    def get_period_length(self):
        return self.period_length


class TestParameters1(Parameters):
    #  beta1(a) = const.,  dbeta1/da = 0
    def __init__(self, beta2, p=0.5, h=0.05, period_time=1, beta1=1.0,
                 mu=0.3, sigma=0.5):
        self.p = p
        self.h = h
        self.period = period_time
        self.period_length = int(round(period_time / h, 1))
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.sigma = sigma

    def get_beta1(self, a):
        return self.beta1

    def get_beta2(self, t):
        index = int(t / self.h) % self.period_length
        return self.beta2[index]

    def get_beta(self, a, t):
        return self.get_beta1(a) * self.get_beta2(t)

    def get_dbeta(self, a, t):
        """
        Derivative of beta wrt a
        """
        return 0

    def get_mu(self, a):
        return self.mu

    def get_sigma1(self, a):
        return self.sigma1

    def get_sigma2(self, t):
        return self.sigma2

    def get_sigma(self, a, t):
        return self.sigma

    def get_p(self):
        return self.p

    def get_h(self):
        return self.h

    def get_period(self):
        return self.period

    def get_period_length(self):
        return self.period_length
