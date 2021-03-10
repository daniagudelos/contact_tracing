#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:54:35 2020

@author: saitel
"""
from scipy.integrate import trapz, solve_ivp
from parameters.parameters import ConstantParameters, VariableParameters
import numpy as np
from helper.plotter import Plotter


class RecursiveBCT:
    def __init__(self, parameters, a_max, t_0_max):
        self.beta = parameters.get_beta
        self.dbeta = parameters.get_dbeta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.a_max = a_max
        self.t_0_max = t_0_max
        self.t_0_length = int(round(t_0_max / self.h(), 1))
        self.a_length = int(round(a_max / self.h(), 1))
        self.t_0_array = np.linspace(0.0, self.t_0_max + self.a_max,
                                     self.t_0_length + self.a_length + 1)
        self.a_array = np.linspace(0.0, self.a_max, self.a_length + 1)
        self.kappa_minus = np.ones((self.t_0_length + self.a_length + 1,
                                    self.a_length + 1))
        self.dkappa_minus = np.ones_like(self.kappa_minus)

    def integrand(self, b_index, t_0_index, a_index):
        return (self.kappa_minus[t_0_index + a_index - b_index, b_index] * (
                self.dbeta(self.a_array[a_index - b_index],
                           self.t_0_array[t_0_index] +
                           self.a_array[a_index - b_index]) -
                self.beta(self.a_array[a_index - b_index],
                          self.t_0_array[t_0_index] +
                          self.a_array[a_index - b_index]) *
                self.mu(self.a_array[b_index])))

    def integral(self, a_upper, t_0_index):

        if self.p == 0:
            return 0

        if a_upper < self.h():
            return 0

        a_index = np.where(self.a_array <= a_upper)[0][-1]

        # y: array to integrate
        b_array = self.a_array[0:a_index + 1]
        y = np.zeros_like(b_array)  # from 0 to a_upper

        for i in range(0, len(b_array)):  # from 0 to a_upper
            y[i] = self.integrand(i, t_0_index, a_index)

        return trapz(y, b_array)

    def fun(self, a, kappa_minus, t_0_index):
        dkappa_minus = (- kappa_minus *
                        (self.mu(a) +
                         self.sigma(a, self.t_0_array[t_0_index] + a) -
                         self.p() * self.beta(0, self.t_0_array[t_0_index]) *
                         kappa_minus + self.p() *
                         self.beta(a, self.t_0_array[t_0_index] + a)  # * 1
                         + self.p() * self.integral(a, t_0_index)))
        return dkappa_minus

    def calculate_kappa_minus_for_cohort(self, t_0_index, a_index):
        a_array = self.a_array[0:a_index + 1]  # from 0 to a_index
        kappa0 = [1]  # must be a 1-d array!
        sol = solve_ivp(self.fun, [0, a_array[-1]], kappa0, method='Radau',
                        t_eval=a_array, dense_output=True, vectorized=True,
                        args=[t_0_index], rtol=1e-3, atol=1e-9)
        return sol

    def calculate_kappa_minus(self):
        # Calculate ghost cohorts: i represents t_0 + a index
        for t_0_index in range(len(self.t_0_array) - 2, self.t_0_length, -1):
            # From t_0 + a - h to t_0 + h
            # skip last t_0 ghost because kappa[-1,:] = 1
            a_index = self.t_0_length + self.a_length - t_0_index

            sol = self.calculate_kappa_minus_for_cohort(t_0_index, a_index)
            self.kappa_minus[t_0_index, 0: a_index + 1] = sol.y.reshape(-1)

        for t_0_index in range(self.t_0_length, -1, -1):  # From t_0  to 0
            a_index = self.a_length

            sol = self.calculate_kappa_minus_for_cohort(t_0_index, a_index)
            self.kappa_minus[t_0_index, 0: a_index + 1] = sol.y.reshape(-1)

        # Fix numerical errors:
        self.kappa_minus = np.where(self.kappa_minus < 1e-10, 0, self.kappa_minus)

        return self.t_0_array[0:(self.t_0_length + 1)], self.a_array,\
            self.kappa_minus[0:(self.t_0_length + 1), :]


def recursive_bct_test(pars, filename, a_max=2, t_0_max=6):
    otbct = RecursiveBCT(pars, a_max, t_0_max)
    t_0_array, a_array, kappa_minus = otbct.calculate_kappa_minus()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    Plotter.plot_3D(t_0, a, kappa_minus, filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa_minus, filename + '_n60_10', azim=-60,
                    my=0.5)
    return t_0_array, a_array, kappa_minus


def main2():
    t_0_array, a_array, kappa_minus = recursive_bct_test(VariableParameters(
        p=1/3, h=0.1), '../../figures/periodic/bct_re_variable_p03', 2, 2)
    return t_0_array, a_array, kappa_minus


def main():
    print('Running simulation BCT with constant parameters and p=0.0')
    recursive_bct_test(ConstantParameters(p=0, h=0.5),
                       '../../figures/non_periodic/bct_re_constant_p0')
    print('Running simulation BCT with constant parameters and p=1/3')
    recursive_bct_test(ConstantParameters(p=1/3, h=0.5),
                       '../../figures/periodic/bct_re_constant_p03')
    print('Running simulation BCT with constant parameters and p=2/3')
    recursive_bct_test(ConstantParameters(p=2/3, h=0.5),
                       '../../figures/periodic/bct_re_constant_p06')
    print('Running simulation BCT with constant parameters and p=1')
    recursive_bct_test(ConstantParameters(p=1, h=0.5),
                       '../../figures/periodic/bct_re_constant_p1')

    print('Running simulation BCT with variable parameters and p=0.0')
    recursive_bct_test(VariableParameters(p=0, h=0.5),
                       '../../figures/non_periodic/bct_re_variable_p0')
    print('Running simulation BCT with variable parameters and p=1/3')
    recursive_bct_test(VariableParameters(p=1/3, h=0.5),
                       '../../figures/periodic/bct_re_variable_p03')
    print('Running simulation BCT with variable parameters and p=2/3')
    recursive_bct_test(VariableParameters(p=2/3, h=0.5),
                       '../../figures/periodic/bct_re_variable_p06')
    print('Running simulation BCT with variable parameters and p=1')
    recursive_bct_test(VariableParameters(p=1, h=0.5),
                       '../../figures/periodic/bct_re_variable_p1')


if __name__ == '__main__':
    t_0_array, a_array, kappa_minus = main2()
