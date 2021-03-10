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
from helper.exporter import Exporter


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
        self.trunc = trunc  # N in outer integral
        self.trunc_length = int(round(self.trunc / self.h(), 1))
        self.nct = NoCT(self.parameters, self.trunc * (self.n_gen - 1) +
                        self.a_max, self.t_0_max)
        self.bct = RecursiveBCT(self.parameters, self.trunc * (self.n_gen - 1)
                                + self.a_max, self.t_0_max)
        self.t_0_array, self.a_array, self.kappa_hat = self.nct.calculate_kappa_hat()
        self.f = []

    def calculate_f_0(self):
        _, _, kappa_minus = self.bct.calculate_kappa_minus()
        Exporter.save_variable(kappa_minus, 'kappa_re_bct')
        # kappa_minus = Exporter.load_variable('kappa_re_bct')
        f_0 = kappa_minus / self.kappa_hat
        # Exporter.save_variable(f_0, 'f_0_fct')
        # f_0 = Exporter.load_variable('f_0_fct')
        return f_0

    def calculate_kappa_plus(self):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.kappa_hat[:, 0:(self.a_length + 1)]
        f_plus = self.calculate_f_plus()
        return (self.t_0_array, self.a_array[0:(self.a_length + 1)],
                kappa_hat * f_plus)

    def calculate_d(self, t_0_index, N_length):
        f_old = self.f[-1]
        temp = np.zeros(N_length + 1)
        for i in range(0, N_length + 1):
            index = (t_0_index - i) % self.period_length
            temp[i] = (self.beta(self.a_array[i], self.t_0_array[t_0_index]) *
                       self.kappa_hat[index, i] * f_old[index, i])

        return np.trapz(temp)

    def calculate_f_plus(self):
        # Calculates f_plus_infinity: after convergence

        self.f.append(self.calculate_f_0())
        # Calculate first generation: sum_{b=0}^{N} sum_{a=0}^{M}
        for i in range(1, self.n_gen):  # from 1 to n_gen - 1

            # hasta donde calculo en gen i: f_i(a; t_0)
            M_length = self.a_length + (self.n_gen - i - 1) * self.trunc_length
            N_length = self.a_length + (self.n_gen - i) * self.trunc_length
            f_plus = np.ones((self.t_0_length + 1, M_length + 1))

            for t_0_index in range(0, self.t_0_length + 1):
                # Calculate d for t0[t_0_index]
                d = self.calculate_d(t_0_index, N_length)
                outer2 = self.calculate_outer2(t_0_index, M_length, N_length)
                for a_index in range(1, M_length + 1):
                    outer1 = self.calculate_outer1(t_0_index, a_index, N_length)
                    f_plus[t_0_index, a_index] = (1 - self.p() +
                                                  self.p() / d *
                                                  (outer1 - outer2))

            self.f.append(f_plus)
        return self.f[-1]

    def calculate_outer1(self, t_0_index, a_index, N_length):
        f_old = self.f[-1]
        temp = np.zeros(N_length + 1)

        for b_index in range(a_index, N_length - a_index):
            index = (t_0_index - b_index + a_index) % self.period_length
            temp[b_index] = (self.beta(self.a_array[b_index - a_index],
                                       self.t_0_array[t_0_index]) *
                             self.kappa_hat[index, b_index] *
                             f_old[index, b_index])

        return np.trapz(temp)

    def calculate_outer2(self, t_0_index, M_length, N_length):
        f_old = self.f[-1]

        tempi = np.zeros(M_length + 1)
        for a_index in range(0, M_length + 1):
            temp = np.zeros(N_length - a_index + 1)
            for b_index in range(0, N_length - a_index):  # from 0 to b_length
                index = (t_0_index - b_index) % self.period_length
                temp[b_index] = (self.beta(self.a_array[b_index],
                                           self.t_0_array[t_0_index]) *
                                 (self.mu(self.a_array[a_index + b_index]) *
                                 self.kappa_hat[index, a_index + b_index] *
                                 f_old[index, a_index + b_index]))
            tempi[a_index] = np.trapz(temp)

        return np.trapz(tempi)


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
        par, '../../figures/periodic/fct_re_variable_p03', a_max= T,
        t_0_max=2 * T)
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
