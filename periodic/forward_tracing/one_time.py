#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:28 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time import OneTimeBCT
import numpy as np


class OneTimeFCT:
    def __init__(self, parameters, n_gen, trunc):
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.nct = NoCT(parameters)
        self.bct = OneTimeBCT(parameters)
        self.n_gen = n_gen
        self.trunc = trunc  # truncation for \int f_i-1 in the last generation

    def calculate_f_zero(self, a_max, t_0_max):
        _, _, kappa_minus, _ = self.bct.calculate_kappa(a_max, t_0_max)
        _, _, kappa_hat = self.nct.calculate_kappa_hat(a_max, t_0_max)
        return kappa_minus / kappa_hat

    def calculate_kappa_plus(self, a_max, t_0_max):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.nct.calculate_kappa_hat(a_max, t_0_max)
        f_plus = self.calculate_f_plus(a_max, t_0_max)
        return kappa_hat * f_plus

    def calculate_d(self, f_old, t_0, t_0_index):
        L = self.trunc   # ej. 60
        b = np.arange(0, L + self.h(), self.h())  # points: L / h + 1

        d = (self.beta(b[0], t_0[t_0_index]) *
             self.nct.get_kappa_hat_at(b[0], t_0[t_0_index] - b[0]) *
             f_old[b[0], t_0[t_0_index] - b[0]] +
             self.beta(b[-1], t_0[t_0_index]) *
             self.nct.get_kappa_hat_at(b[-1], t_0[t_0_index] - b[-1]) *
             f_old[b[-1], t_0[t_0_index] - b[-1]]) * 0.5

        # avoid first and last index:
        for j in range(1, t_0_index + 1):
            d += (self.beta(b[j], t_0[t_0_index]) *
                  self.nct.get_kappa_hat_at(b[j], t_0[t_0_index] - b[j]) *
                  f_old[b[j], t_0[t_0_index] - b[j]])

        for j in range(t_0_index + 1, len(b) - 1):
            # adjust b_index so that it is positive
            b_index = j % t_0_index  # Only works if t_0 covers X whole periods
            d += (self.beta(b[j], t_0[t_0_index]) *
                  self.nct.get_kappa_hat_at(j, t_0_index - b_index) *
                  f_old[j, t_0_index - b_index])
        return self.h * d

    def calculate_f_plus(self, a_max, t_0_max):
        # Calcuates f_plus_infinity: after convergence

        # sum points
        L_max = self.trunc * 2 ^ (self.n_gen - 1)  # ej. 60
        f = np.zeros((self.n_gen, L_max + a_max + 1, t_0_max + 1))

        # Calculate first generation
        f[0] = self.calculate_f_zero(L_max + a_max, t_0_max)  # a + b, t_0 -b

        # initialize cycle for a = 0
        for i in range(1, self.n_gen):  # from 1 to n_gen - 1
            for j in range(0, t_0_max + 1):  # from 0 to t_0_max
                f[i, 0, j] = 1

        for i in range(1, self.n_gen):  # from 1 to n_gen - 1
            L = self.trunc * 2 ^ (self.n_gen - 1 - i)
            b = np.arange(0, L + self.h(), self.h())  # points: L / h + 1
            a = np.arange(0, a_max + self.h(), self.h())
            t_0 = np.arange(0, t_0_max + a_max + self.h(), self.h())
            for j in range(1, a_max + 1):  # from 1 to a_max
                for k in range(0, t_0_max + 1):  # from 0 to t_0_max
                    f[i, j, k] = self.calculate_f_plus_point(a, j, t_0, k, b,
                                                             f[i-1])
        return f[self.n_gen - 1]

    def calculate_f_plus_point(self, a, a_index, t_0, t_0_index, b, f_old):
        # Input: vector a, index a, vector t_0, index t_0, matrix f_i-1

        outer = 0

        # Calculate d for t0[t_0_index]
        d = self.calculate_d(f_old, t_0, t_0_index)

        # b = 0 and b = N
        N = len(b) - 1  # outer: b is from 0 to len(b) - 1 (len(b) points)
        outer = (self.calculate_phi(a, a_index, t_0, t_0_index, b, 0, f_old, d)
                 + self.calculate_phi(a, a_index, t_0, t_0_index, b, N, f_old,
                                      d)) * 0.5

        for j in range(1, N):  # from 1 to N - 1
            outer += (self.calculate_phi(a, a_index, t_0, t_0_index, b, j,
                                         f_old, d))
        outer = outer * self.h
        return 1 - self.p * outer

    def calculate_phi(self, a, a_index, t_0, t_0_index, b, b_index, f_old, d):
        # Calculates Phi for a given value of b and a

        M = int(a_index / self.h)  # inner

        # b_index for M
        if b_index > t_0_index:
            b_index_fixed = b_index % t_0_index
        else:
            b_index_fixed = b_index

        inner = (self.sigma(a[0] + b[b_index], t_0[t_0_index] + a[0]) *
                 self.nct.get_kappa_hat_at(b_index, t_0_index - b_index_fixed)
                 * f_old[b_index, t_0_index - b_index_fixed] +
                 self.sigma(a[M] + b[b_index], t_0[t_0_index] + a[M]) *
                 self.nct.get_kappa_hat_at(M + b_index, t_0_index -
                                           b_index_fixed) *
                 f_old[M + b_index, t_0_index - b_index_fixed]) * 0.5

        for k in range(1, M):  # 1 to M - 1
            inner += (self.sigma(a[k] + b[b_index], t_0[t_0_index] + a[k])
                      * self.nct.get_kappa_hat_at(k + b_index, t_0_index -
                                                  b_index_fixed) *
                      f_old[k + b_index, t_0_index - b_index_fixed])
        inner = inner * self.h
        return self.beta(b[b_index], t_0[t_0_index]) * inner / d
