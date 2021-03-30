#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:13:28 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time_bct import OneTimeBCT
from parameters.parameters import TestParameters1
import numpy as np
from timeit import default_timer as timer
from scipy.integrate import simps as simpson
from scipy import optimize
from joblib import Parallel, delayed
from helper.plotter import Plotter
from helper.exporter import Exporter


class OneTimeFCT:
    def __init__(self, parameters, n_gen, a_max, t_0_max, trunc=2):
        self.period = parameters.get_period()  # Period in days
        self.period_length = parameters.get_period_length()
        self.parameters = parameters
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.n_gen = n_gen
        # Number of a-periods asked by the user
        self.a_max = a_max * self.period
        self.a_length = a_max * self.period_length
        self.t_0_max = t_0_max * self.period
        self.t_0_length = t_0_max * self.period_length
        # Number of periods to approximate infinity
        self.trunc = max(a_max, trunc)
        self.inf_length = self.trunc * self.period_length

        self.nct = NoCT(parameters, self.trunc, t_0_max)
        self.bct = OneTimeBCT(parameters, self.trunc, t_0_max)
        self.t_0_array, self.a_array, self.kappa_hat = \
            self.nct.calculate_kappa_hat()
        Exporter.save_variable(self.kappa_hat, 'kappa_nct')
        Exporter.save_variable(self.t_0_array, 't_0_array')
        Exporter.save_variable(self.a_array, 'a_array')

        # self.t_0_array = Exporter.load_variable('t_0_array')
        # self.a_array = Exporter.load_variable('a_array')
        # self.kappa_hat = Exporter.load_variable('kappa_nct')
        self.it = 0
        self.f = []

    def calculate_f_0(self):
        _, _, kappa_minus = self.bct.calculate_kappa_minus()
        # kappa_minus = Exporter.load_variable('kappa_ot_bct')
        Exporter.save_variable(kappa_minus, 'kappa_ot_bct')
        f_0 = kappa_minus / self.kappa_hat
        return f_0

    def calculate_kappa_plus(self):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.kappa_hat[:, 0:(self.a_length + 1)]
        f_plus = self.calculate_f_plus()[:, 0:(self.a_length + 1)]
        return (self.t_0_array, self.a_array[0:(self.a_length + 1)],
                kappa_hat * f_plus)

    def calculate_d(self, t_0_index):
        f_old = self.f[-1]
        temp = np.zeros(self.inf_length + 1)
        for i in range(0, self.inf_length + 1):
            index = (t_0_index - i) % self.period_length
            temp[i] = (self.beta(self.a_array[i], self.t_0_array[t_0_index]) *
                       self.kappa_hat[index, i] * f_old[index, i])

        return simpson(temp)

    def calculate_f_plus(self):
        f_old = self.calculate_f_0()
        self.f.append(f_old)
        #f_plus = optimize.fixed_point(func=self.calculate_f, x0=f_old,
        #                              xtol=1e-02, method='del2')
        for i in range(0, self.n_gen + 1):
            f_plus = self.calculate_f(f_old)
            f_old = f_plus
        return f_plus

    def calculate_f(self, f_old):
        t_0_periods = int(self.t_0_max / self.period)

        f_plus = np.ones((self.t_0_length + 1, self.inf_length + 1))

        # from 0 to period
        f_plus[0: self.period_length + 1, :] = np.asarray(
            Parallel(n_jobs=-1)(delayed(self.calculate_f_plus_for_cohort)
                                (j) for j in range(0, self.period_length + 1)))

        # Copy values to the rest of the periods in t_0-axis
        for j in range(1, t_0_periods):  # 1 : t_0_periods - 1
            t_0_start = self.period_length * j + 1
            t_0_end = self.period_length * (j + 1) + 1
            f_plus[t_0_start: t_0_end, 0:self.inf_length + 1] = (
                f_plus[1: self.period_length + 1, 0:self.inf_length + 1])

        self.it += 1
        Exporter.save_variable(f_plus, 'f' + str(self.it))
        self.f.append(f_plus)

        return self.f[-1]

    def calculate_f_plus_for_cohort(self, t_0_index):
        print(t_0_index)
        f_cohort = np.zeros((self.inf_length + 1))
        # Calculate d for t0[t_0_index]
        d = self.calculate_d(t_0_index)

        f_old = self.f[-1]
        f_cohort[0] = 1

        # from 1 to a_max + b_max
        for a_index in range(1, self.inf_length + 1):
            # integral
            temp1 = np.zeros((self.inf_length + 1))
            temp2 = np.zeros((self.inf_length + 1))

            # From 0 to a
            for b_index in range(0, a_index):
                index = (t_0_index - b_index) % self.period_length
                temp1[b_index] = (self.beta_integral(b_index, t_0_index) *
                                  self.kappa_hat[index, b_index] *
                                  f_old[index, b_index] *
                                  self.sigma(self.a_array[b_index],
                                             self.t_0_array[t_0_index]))

            # From a to infity
            for b_index in range(a_index + 1, self.inf_length + 1):
                index = (t_0_index - b_index) % self.period_length
                index2 = (t_0_index - b_index + a_index) % self.period_length
                temp2[b_index] = (self.beta_integral(b_index, t_0_index) *
                                  self.kappa_hat[index, b_index] *
                                  f_old[index, b_index] *
                                  self.sigma(self.a_array[b_index],
                                             self.t_0_array[t_0_index]) -
                                  self.beta_integral(b_index - a_index,
                                                     t_0_index) *
                                  self.kappa_hat[index2, b_index] *
                                  f_old[index2, b_index] *
                                  self.sigma(self.a_array[b_index],
                                             self.t_0_array[t_0_index]
                                             + self.a_array[a_index]))

            # final f
            removal = self.p() / d * simpson(temp2 + temp1)
            f_cohort[a_index] = 1 - removal
        return f_cohort

    def beta_integral(self, a_index, t_0_index):
        start = 0
        end = a_index
        if(a_index < 0):
            start = a_index
            end = 0

        beta = np.zeros(abs(start - end) + 1)
        j = 0
        for i in range(start, end + 1):
            beta[j] = self.beta(self.a_array[i], self.t_0_array[t_0_index])
            j += 1
        return np.trapz(beta)


def one_time_fct_test(pars, filename, a_max=2, t_0_max=6):
    otfct = OneTimeFCT(pars, n_gen=3, a_max=a_max, t_0_max=t_0_max, trunc=2)
    t_0_array, a_array, kappa_plus = otfct.calculate_kappa_plus()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    mx = round(t_0_max * pars.get_period() / 10)
    my = round(a_max * pars.get_period() / 10)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_60_10', mx=mx, my=my)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_n60_10', azim=-60,
                    mx=mx, my=my)
    return t_0_array, a_array, kappa_plus


def main():
    T = 7  # days
    # beta2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    beta2 = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3.5, 3.5, 3.5, 3.5, 4, 4, 4, 4,
                      3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    par = TestParameters1(beta2, p=1/3, h=0.25, period_time=T)
    t_0_array, a_array, kappa_ot_fct = one_time_fct_test(
        par, '../../figures/periodic/fct_ot_variable_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_ot_fct


if __name__ == '__main__':
    t_0_array, a_array, kappa_ot_fct = main()
    Exporter.save_variable(kappa_ot_fct, 'kappa_ot_fct')
