#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:18:07 2021

@author: saitel
"""
import numpy as np
import scipy.integrate as integrate
from math import exp


class NoCT:
    def __init__(self, parameters, a_max, t_0_max):
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.cycle = 0
        self.cycles = 0
        self.progress = 0
        self.interrupted = False
        self.t_0 = np.arange(0, t_0_max + a_max + self.h(), self.h())
        self.a = np.arange(0, a_max + self.h(), self.h())
        self.kappa_hat = np.zeros((len(self.a), len(self.t_0)))
        self.dkappa_hat = np.zeros((len(self.a), len(self.t_0)))

    def calculate_kappa_hat(self, a_max, t_0_max):
        a = self.a
        t_0 = self.t_0

        for i in range(0, len(a)):
            for j in range(0, len(t_0)):
                self.kappa_hat[i, j] = self.calculate_kappa_hat_point(a[i],
                                                                      t_0[j])

        return t_0, a, self.kappa_hat

    def calculate_kappa_hat_point(self, a_v, t_0_v):
        """
        Returns kappa_hat(a;t_0)

        Parameters
        ----------
        a_v : Float. The value of current a.
        t_0_v : Float. The value of current t_0.

        Returns
        -------
        Float. Integral value

        """
        result = integrate.quad(lambda a: self.mu(a) +
                                self.sigma(a, t_0_v + a), 0, a_v)
        return exp(- result[0])

    def get_kappa_hat_at(self, a_index, t_0_index):
        return self.kappa_hat[a_index, t_0_index]

    def calculate_dkappa_hat(self, a_max, t_0_max):
        t_0 = self.t_0
        a = self.a

        for i in range(0, len(a)):
            for j in range(0, len(t_0)):
                self.dkappa_hat[i, j] = self.get_kappa_hat(i, j) * \
                    (-self.mu(a[i]) - self.sigma(a[i], t_0[j] + a[i]))

    def get_dkappa_hat(self, a_index, t_0_index):
        return self.dkappa_hat[a_index, t_0_index]
