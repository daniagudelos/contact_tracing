#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:28 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time import OneTimeBCT
from math import exp
import numpy as np


class OneTimeFCT:
    def __init__(self, parameters, n_gen):
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.nct = NoCT(parameters)
        self.bct = OneTimeBCT(parameters)
        self.n_gen = n_gen

    def calculate_f_zero(self, a_max, t_0_max):
        _, _, kappa_minus, _ = self.bct.calculate_kappa(a_max, t_0_max)
        kappa_hat = self.nct.calculate_kappa_hat_point(a_max, t_0_max)
        return kappa_minus[-1, -1] / kappa_hat

    def calculate_kappa_plus_point(self, a_v, t_0_v, i):
        n = 60
        m = a_v
        b, w = np.polynomial.laguerre.laggauss(n)

        a = np.arange(0, a_v + self.h(), self.h())
        f = np.zeros((self.n_gen, len(a) + len(b) + 2, len(b) + 1))

        f[0] = self.calculate_f_zero(a_v, t_0_v)

        for i in range(1, self.n_gen):
            sum1 = 0
            sum2 = 0
            d = 0
            for j in range(0, n + 1):
                d += w[j] * self.beta(b[j], t_0_v) * \
                    self.nct.calculate_kappa_hat_point(b[j], t_0_v - b[j]) * \
                    self.f[i, 1, 1] * exp(b[j])

            for j in range(0, n + 1):
                sum1 = (self.sigma(b[j], t_0_v) *
                        self.nct.calculate_kappa_hat_point(b[j], t_0_v - b[j])
                        * f[i - 1, b[j], t_0_v - b[j]] +
                        self.sigma(a[m] + b[j], t_0_v + a[m]) *
                        self.nct.calculate_kappa_hat_point(a[m] + b[j],
                                                           t_0_v - b[j])
                        * f[i - 1, a[m] + b[j], t_0_v - b[j]])
                for k in range(1, m):
                    sum1 += self.sigma(a[k] + b[j], t_0_v + a[k]) * \
                        self.nct.calculate_kappa_hat_point(a[k] + b[j],
                                                           t_0_v - b[j])
            sum2 += w[j] * self.h / 2 * self.beta(b[j], t_0_v) / d * sum1 * \
                exp(b[j])
            f[i] = 1 - self.p * sum2

# Problema: como traer los valores de kappa hat, fi-1, si para traerlos necesito 
# el indice del array y no el valor de a o to?