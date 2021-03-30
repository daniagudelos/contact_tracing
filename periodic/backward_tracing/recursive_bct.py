#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:54:35 2020

@author: saitel
"""
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simpson
from parameters.parameters import TestParameters1
import numpy as np
from helper.plotter import Plotter


class RecursiveBCT:
    def __init__(self, parameters, a_max, t_0_max):
        self.parameters = parameters
        self.period = self.parameters.get_period()  # Period in days
        self.period_length = self.parameters.get_period_length()
        self.beta = self.parameters.get_beta
        self.dbeta = self.parameters.get_dbeta
        self.mu = self.parameters.get_mu
        self.sigma = self.parameters.get_sigma
        self.p = self.parameters.get_p
        self.h = self.parameters.get_h

        self.a_max = a_max * self.period
        self.a_length = a_max * self.period_length
        self.a_array = np.linspace(0.0, self.a_max, self.a_length + 1)

        self.t_0_max = t_0_max * self.period
        self.t_0_length = t_0_max * self.period_length
        self.t_0_array = np.linspace(0.0, self.t_0_max + self.a_max,
                                     self.t_0_length + self.a_length + 1)

        self.kappa_minus = np.ones((self.t_0_length + self.a_length + 1,
                                    self.a_length + 1))

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

        return simpson(y, b_array)

    def fun(self, a, kappa_minus, t_0_index):
        dkappa_minus = (- kappa_minus *
                        (self.mu(a) +
                         self.sigma(a, self.t_0_array[t_0_index] + a) -
                         self.p() * self.beta(0, self.t_0_array[t_0_index]) *
                         kappa_minus + self.p() *
                         self.beta(a, self.t_0_array[t_0_index] + a)  # * 1
                         + self.p() * self.integral(a, t_0_index)))
        return dkappa_minus

    def calculate_kappa_minus_for_cohort(self, t_0_index, a_start, a_end):
        a_array = self.a_array[a_start:a_end + 1]  # from 0 to a_end
        kappa0 = [self.kappa_minus[t_0_index, a_start]]  # must be a 1-d array!
        sol = solve_ivp(self.fun, [a_array[0], a_array[-1]], kappa0,
                        method='LSODA', t_eval=a_array, dense_output=True,
                        vectorized=True, args=[t_0_index], rtol=1e-4,
                        atol=1e-9)
        return sol

    def calculate_kappa_minus(self):
        a_periods = int(self.a_max / self.period)
        # Extra periods for ghost cells
        t_0_periods = int(self.t_0_max / self.period) + a_periods

        # Calculate ghost cohorts: i represents t_0 + T
        start_ghost = self.period_length - 1
        # skip last t_0 ghost because kappa[-1,:] = 1 & not needed
        end_ghost = 2 * self.period_length - 1

        for period_index in range(1, a_periods + 1):
            a_start = self.period_length * (period_index - 1)
            a_end = self.period_length * (period_index - 1)

            # Calculate upper-left half of matrix second period
            for t_0_index in range(end_ghost, start_ghost, -1):
                a_end += 1
                # From 2 T to T + h
                sol = self.calculate_kappa_minus_for_cohort(t_0_index, a_start,
                                                            a_end)
                self.kappa_minus[t_0_index, a_start: a_end + 1] = (
                    sol.y.reshape(-1))

            # Copy values to first period in t_0 axis
            # a_end here should be = self.period_length * period_index
            self.kappa_minus[0: self.period_length, a_start:a_end + 1] = (
                self.kappa_minus[self.period_length: 2 * self.period_length,
                                 a_start:a_end + 1])

            # Calculate lower-right half of matrix period: From T  to 1
            for t_0_index in range(self.period_length - 1, 0, -1):
                a_start += 1
                sol = self.calculate_kappa_minus_for_cohort(t_0_index, a_start,
                                                            a_end)
                self.kappa_minus[t_0_index, a_start: a_end + 1] = (
                    sol.y.reshape(-1))

            a_start = self.period_length * (period_index - 1)
            a_end = self.period_length * (period_index)

            # Copy values to the rest of the periods in t_0-axis
            for i in range(1, t_0_periods):  # 1 : t_0_periods - 1
                t_0_start = self.period_length * i + 1
                t_0_end = self.period_length * (i + 1) + 1
                self.kappa_minus[t_0_start: t_0_end, a_start:a_end + 1] = (
                    self.kappa_minus[1: self.period_length + 1,
                                     a_start:a_end + 1])

        # Fix numerical errors:
        self.kappa_minus = np.where(self.kappa_minus < 0, 0, self.kappa_minus)

        return self.t_0_array[0:(self.t_0_length + 1)], self.a_array,\
            self.kappa_minus[0:(self.t_0_length + 1), :]


def recursive_bct_test(pars, filename, a_max=2, t_0_max=6):
    otbct = RecursiveBCT(pars, a_max, t_0_max)
    t_0_array, a_array, kappa_minus = otbct.calculate_kappa_minus()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    mx = round(t_0_max * pars.get_period() / 10)
    my = round(a_max * pars.get_period() / 10)
    Plotter.plot_3D(t_0, a, kappa_minus, filename + '_60_10', mx=mx, my=my)
    Plotter.plot_3D(t_0, a, kappa_minus, filename + '_n60_10', azim=-60,
                    mx=mx, my=my)
    return t_0_array, a_array, kappa_minus


def main():
    T = 7  # days
    beta2 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    par = TestParameters1(beta2, p=1/3, h=0.25, period_time=T)
    t_0_array, a_array, kappa_plus = recursive_bct_test(
        par, '../../figures/periodic/fct_re_variable_p03', a_max=2,
        t_0_max=3)
    return t_0_array, a_array, kappa_plus


if __name__ == '__main__':
    t_0_array, a_array, kappa_minus = main()
