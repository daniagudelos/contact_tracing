#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:28 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time_bct import OneTimeBCT
from parameters.parameters import ConstantParameters, VariableParameters
import numpy as np
from helper.plotter import Plotter
from scipy.integrate import trapz


class OneTimeFCT:
    def __init__(self, parameters, n_gen, trunc, a_max, t_0_max):
        # trunc: truncation for \int f_i-1 in the last generation (f_gen_max)
        self.parameters = parameters
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.gen_max = n_gen - 1
        self.a_max = a_max
        self.t_0_max = t_0_max
        self.trunc = trunc
        self.nct = NoCT(self.parameters, self.trunc * self.gen_max +
                        self.a_max, self.t_0_max)
        self.bct = OneTimeBCT(self.parameters, self.trunc * self.gen_max +
                              self.a_max, self.t_0_max)
        _, _, kappa_minus = self.bct.calculate_kappa_minus()
        _, _, self.kappa_hat = self.nct.calculate_kappa_hat()
        self.f = []
        self.f.append(kappa_minus / self.kappa_hat)
        self.t_0_length = int(round(t_0_max / self.h(), 1))
        self.a_length = int(round(a_max / self.h(), 1))

    def calculate_kappa_plus(self):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.kappa_hat
        f_plus = self.calculate_f_plus()
        return kappa_hat * f_plus

    def calculate_d(self, f_old, t_0_array, t_0_index, a_array, b_length):
        f_old = self.f[-1]
        b_array = a_array[0:b_length + 1]
        temp = np.zeros_like(b_array)
        for i in range(0, t_0_index + 1):
            temp[i] = (self.beta(b_array[i], t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index - i, i] *
                       f_old[t_0_index - i, i])
        for i in range(t_0_index + 1, b_length + 1):
            if t_0_index != 0:
                b_index_fixed = i % t_0_index
            else:
                b_index_fixed = i
            temp[i] = (self.beta(b_array[i], t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index - b_index_fixed, i] *
                       f_old[t_0_index - b_index_fixed, i])

        return trapz(temp)

    def calculate_d2(self, f_old, t_0, t_0_index):
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

    def calculate_f_plus(self):
        # Calcuates f_plus_infinity: after convergence

        # Calculate first generation
        for i in range(1, self.gen_max + 1):  # from 1 to gen_max
            b_max = self.trunc * i
            b_length = int(round(b_max / self.h(), 1))
            # b_array = np.linspace(0.0, self.b_max + self.a_max,
            #                b_length + self.a_length + 1)
            a_array = np.linspace(0.0, self.a_max + b_max, self.a_length +
                                  b_length + 1)
            t_0_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)

            for j in range(0, self.t_0_max + 1):  # from 0 to t_0_max
                f_plus = np.ones((self.t_0_length + 1, self.a_length +
                                  b_length + 1))
                for k in range(1, len(a_array) + 1):  # from 1 to a_max + b_max
                    f_plus[j, k] = self.calculate_f_plus_point(a_array, k,
                                                               t_0_array, j,
                                                               self.f[i-1],
                                                               b_length)
                self.f.append(f_plus)
        return self.f  # [self.gen_max]

    def calculate_f_plus_point(self, a_array, a_index, t_0_array, t_0_index,
                               f_old, b_length):
        # Input: vector a, index a, vector t_0, index t_0, matrix f_i-1

        outer = 0

        # Calculate d for t0[t_0_index]
        d = self.calculate_d(f_old, t_0_array, t_0_index, a_array, b_length)

        # b = 0 and b = N
        # N = len(b) - 1  # outer: b is from 0 to len(b) - 1 (len(b) points)
        # outer = (self.calculate_phi(a_array, a_index, t_0_array, t_0_index, b, 0, f_old, d)
        #         + self.calculate_phi(a_array, a_index, t_0_array, t_0_index, b, N, f_old,
        #                              d)) * 0.5

        # for j in range(1, N):  # from 1 to N - 1
        #    outer += (self.calculate_phi(a_array, a_index, t_0, t_0_index, b, j,
        #                                 f_old, d))
        # outer = outer * self.h
        # return 1 - self.p * outer

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


def one_time_fct_test(pars, filename, a_max=2, t_0_max=6):
    otfct = OneTimeFCT(pars, n_gen=2, trunc=10, a_max=a_max, t_0_max=t_0_max)
    kappa_plus = otfct.calculate_kappa_plus()
    # a, t_0 = np.meshgrid(a_array, t_0_array)
    # Plotter.plot_3D(t_0, a, kappa_minus, filename + '_60_10', my=0.5)
    # Plotter.plot_3D(t_0, a, kappa_minus, filename + '_n60_10', azim=-60,
    #                my=0.5)
    return kappa_plus


def main2():
    kappa_plus = one_time_fct_test(VariableParameters(p=1/3, h=0.5),
                                   '../figures/periodic/fct_ot_constant_p03',
                                   a_max=2, t_0_max=2)
    return kappa_plus


if __name__ == '__main__':
    kappa_plus = main2()
