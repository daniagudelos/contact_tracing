#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:05:06 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.recursive_bct import RecursiveBCT
from parameters.parameters import ConstantParameters, VariableParameters, TestParameters1
import numpy as np
from helper.plotter import Plotter


class RecursiveFCT:
    def __init__(self, parameters, n_gen, trunc, a_max, t_0_max):
        self.period = parameters.get_period()  # Period in days
        self.period_length = parameters.get_period_length()
        self.parameters = parameters
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.n_gen = n_gen
        self.a_max = a_max
        self.a_length = int(round(self.a_max / self.h(), 1))
        self.t_0_max = t_0_max
        self.t_0_length = int(round(self.t_0_max / self.h(), 1))
        self.N_max = trunc  # N in outer integral
        self.N_length = int(round(self.N_max / self.h(), 1))
        self.nct = NoCT(self.parameters, self.N_max * (self.n_gen - 1) +
                        self.a_max, self.t_0_max)
        self.bct = RecursiveBCT(self.parameters, self.N_max * (self.n_gen - 1)
                                + self.a_max, self.t_0_max)
        self.t_0_array, self.a_array, self.kappa_hat = self.nct.calculate_kappa_hat()
        self.f = []

    def calculate_f_0(self):
        _, _, kappa_minus = self.bct.calculate_kappa_minus()
        f_0 = kappa_minus / self.kappa_hat
        f_0 = np.where(f_0 > 1, 1, f_0)  # truncate ratio
        f_0 = np.where(f_0 == 0, 1, f_0)  # 0 / 0 ~ 1
        return f_0

    def calculate_kappa_plus(self):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.kappa_hat[:, 0:(self.a_length + 1)]
        f_plus = self.calculate_f_plus()
        return (self.t_0_array, self.a_array[0:(self.a_length + 1)],
                kappa_hat * f_plus)

    def calculate_d(self, t_0_index):
        f_old = self.f[-1]
        temp = np.zeros(self.N_length + 1)
        for i in range(0, min(t_0_index + 1, self.N_length)):
            temp[i] = (self.beta(self.a_array[i], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index - i, i] *
                       f_old[t_0_index - i, i])
        for i in range(min(t_0_index + 1, self.N_length), self.N_length + 1):
            if t_0_index != 0:
                t_0_index_fixed = t_0_index
                b_index_fixed = i % t_0_index
            else:  # move t_0 one period forward
                t_0_index_fixed = t_0_index + self.period_length
                b_index_fixed = int(i % t_0_index_fixed)
            temp[i] = (self.beta(self.a_array[i], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index_fixed - b_index_fixed, i] *
                       f_old[t_0_index_fixed - b_index_fixed, i])

        return np.trapz(temp)

    def calculate_f_plus(self):
        # Calculates f_plus_infinity: after convergence

        self.f.append(self.calculate_f_0())
        # Calculate first generation: sum_{b=0}^{N} sum_{a=0}^{M}
        for i in range(1, self.n_gen):  # from 1 to gen_max

            # hasta donde calculo en gen i: f_i(a; t_0)
            M_max = self.a_max + (self.n_gen - i - 1) * self.N_max
            M_length = self.a_length + (self.n_gen - i - 1) * self.N_length
            f_plus = np.ones((self.t_0_length + 1, M_length + 1))

            for j in range(0, self.t_0_length + 1):  # from 0 to t_0_max
                # Calculate d for t0[t_0_index]
                d = self.calculate_d(j)
                for k in range(1, M_length + 1):  # from 1 to a_max + b_max
                    f_plus[j, k] = self.calculate_f_plus_point(j, k, M_max,
                                                               M_length, d)
            self.f.append(f_plus)
        return self.f[-1]

    def calculate_f_plus_point(self, t_0_index, a_index, M_max, M_length, d):
        # Input: index in t_0, index in a, upper bound of inner
        # summation, number of points of inner summation

        # integral 1
        outer1 = self.calculate_outer1(t_0_index, a_index)
        # integral 2
        outer2 = self.calculate_outer2(t_0_index, a_index)
        # integral 3
        outer3 = self.calculate_outer3(t_0_index, a_index, M_max, M_length, d)

        return 1 + self.p() / d * (outer1 - outer2 + outer3)

    def calculate_outer1(self, t_0_index, a_index):
        f_old = self.f[-1]
        temp = np.zeros(self.N_length + 1)
        for j in range(0, min(t_0_index + 1, self.N_length)):
            temp[j] = (self.beta(self.a_array[j], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index - j, a_index + j] *
                       f_old[t_0_index - j, a_index + j])
        for j in range(min(t_0_index + 1, self.N_length), self.N_length + 1):
            if t_0_index != 0:
                t_0_index_fixed = t_0_index
                b_index_fixed = j % t_0_index
            else:  # move t_0 one period forward
                t_0_index_fixed = t_0_index + self.period_length
                b_index_fixed = int(j % t_0_index_fixed)
            temp[j] = (self.beta(self.a_array[j], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index_fixed - b_index_fixed,
                                      a_index + j] *
                       f_old[t_0_index_fixed - b_index_fixed, a_index + j])
        return np.trapz(temp)

    def calculate_outer2(self, t_0_index, a_index):
        f_old = self.f[-1]
        temp = np.zeros(self.N_length + 1)
        for j in range(0, min(t_0_index + 1, self.N_length)):
            temp[j] = (self.beta(self.a_array[j], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index - j, j] *
                       f_old[t_0_index - j, j])
        for j in range(min(t_0_index + 1, self.N_length), self.N_length + 1):
            if t_0_index != 0:
                t_0_index_fixed = t_0_index
                b_index_fixed = j % t_0_index
            else:  # move t_0 one period forward
                t_0_index_fixed = t_0_index + self.period_length
                b_index_fixed = int(j % t_0_index_fixed)
            temp[j] = (self.beta(self.a_array[j], self.t_0_array[t_0_index]) *
                       self.kappa_hat[t_0_index_fixed - b_index_fixed, j] *
                       f_old[t_0_index_fixed - b_index_fixed, j])
        return np.trapz(temp)

    def calculate_outer3(self, t_0_index, a_index, M_max, M_length, d):
        temp = np.zeros(self.N_length + 1)

        for j in range(0, self.N_length + 1):  # from 0 to b_length
            temp[j] = self.calculate_phi3(a_index, t_0_index, j, d, M_max,
                                          M_length)

        return np.trapz(temp)

    def calculate_phi3(self, a_index, t_0_index, b_index, d, M_max, M_length):
        f_old = self.f[-1]
        temp = np.zeros(M_length + 1)

        if b_index <= t_0_index:
            t_0_index_fixed = t_0_index
            b_index_fixed = b_index
        elif t_0_index != 0:
            t_0_index_fixed = t_0_index
            b_index_fixed = b_index % t_0_index
        else:  # move t_0 one period forward
            t_0_index_fixed = t_0_index + self.period_length
            b_index_fixed = int(b_index % t_0_index_fixed)

        for k in range(0, M_length + 1):
            temp[k] = (self.mu(self.a_array[k + b_index]) *
                       self.kappa_hat[t_0_index_fixed - b_index_fixed,
                                      k + b_index] *
                       f_old[t_0_index_fixed - b_index_fixed, k + b_index])

        inner = np.trapz(temp)
        return (self.beta(self.a_array[b_index], self.t_0_array[t_0_index]) *
                inner)


def recursive_fct_test(pars, filename, a_max=2, t_0_max=6):
    otfct = RecursiveFCT(pars, n_gen=5, trunc=10, a_max=a_max, t_0_max=t_0_max)
    t_0_array, a_array, kappa_plus = otfct.calculate_kappa_plus()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_n60_10', azim=-60,
                    my=0.5)
    return t_0_array, a_array, kappa_plus


def main3():
    T = 7  # days
    beta2 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    par = TestParameters1(beta2, p=1/3, h=0.25, period_time=T)
    t_0_array, a_array, kappa_plus = recursive_fct_test(
        par, '../../figures/periodic/fct_re_variable_p03', a_max=T,
        t_0_max=T)
    return t_0_array, a_array, kappa_plus


def main2():
    t_0_array, a_array, kappa_plus = recursive_fct_test(VariableParameters(
        p=1/3, h=0.5), '../../figures/periodic/fct_re_variable_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_plus


def main():
    print('Running simulation FCT with constant parameters and p=0.0')
    recursive_fct_test(ConstantParameters(p=0, h=0.5),
                       '../../figures/non_periodic/fct_re_constant_p0')
    print('Running simulation FCT with constant parameters and p=1/3')
    recursive_fct_test(ConstantParameters(p=1/3, h=0.5),
                       '../../figures/periodic/fct_re_constant_p03')
    print('Running simulation FCT with constant parameters and p=2/3')
    recursive_fct_test(ConstantParameters(p=2/3, h=0.5),
                       '../../figures/periodic/fct_re_constant_p06')
    print('Running simulation FCT with constant parameters and p=1')
    recursive_fct_test(ConstantParameters(p=1, h=0.5),
                       '../../figures/periodic/fct_re_constant_p1')

    print('Running simulation FCT with variable parameters and p=0.0')
    recursive_fct_test(VariableParameters(p=0, h=0.5),
                       '../../figures/non_periodic/fct_re_variable_p0')
    print('Running simulation FCT with variable parameters and p=1/3')
    recursive_fct_test(VariableParameters(p=1/3, h=0.5),
                       '../../figures/periodic/fct_re_variable_p03')
    print('Running simulation FCT with variable parameters and p=2/3')
    recursive_fct_test(VariableParameters(p=2/3, h=0.5),
                       '../../figures/periodic/fct_re_variable_p06')
    print('Running simulation FCT with variable parameters and p=1')
    recursive_fct_test(VariableParameters(p=1, h=0.5),
                       '../../figures/periodic/fct_re_variable_p1')


if __name__ == '__main__':
    t_0_array, a_array, kappa_plus = main3()
