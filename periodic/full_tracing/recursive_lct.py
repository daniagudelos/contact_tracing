#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:05:06 2021

@author: saitel
"""
from helper.exporter import Exporter
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.recursive_bct import RecursiveBCT
from periodic.forward_tracing.recursive_fct import RecursiveFCT
from parameters.parameters import ConstantParameters, VariableParameters, TestParameters1
import numpy as np
from helper.plotter import Plotter


class RecursiveLCT():
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
        # self.bct = RecursiveBCT(self.parameters, self.N_max * (self.n_gen - 1)
        #                        + self.a_max, self.t_0_max)
        self.t_0_array, self.a_array, self.kappa_hat = self.nct.calculate_kappa_hat()
        self.f = []

    # def calculate_kappa_plus(self):
    #    #kappa_plus = self.fct.calculate_f_plus()
    #    return kappa_plus

    def calculate_f_0(self):
        #_, _, kappa_minus = self.bct.calculate_kappa_minus()
        kappa_minus = Exporter.load_variable('kappa_re_bct')
        f_0 = self.kappa_minus / self.kappa_hat
        Exporter.save_variable(f_0, 'f_0')
        return f_0

    def calculate_kappa(self):
        """
            Returns an array with kappa_plus
        """
        kappa_plus = Exporter.load_variable('kappa_re_fct')
        # kappa_hat = self.kappa_hat[:, 0:(self.a_length + 1)]
        kappa_minus = Exporter.load_variable('kappa_re_bct')
        #kappa_minus = kappa_minus[:, 0:(self.a_length + 1)]
        kappa = kappa_minus * kappa_plus
        Exporter.save_variable(kappa, 'kappa_re_lct')
        return (self.t_0_array, self.a_array, #y[0:(self.a_length + 1)],
                kappa)


def recursive_lct_test(pars, filename, a_max=2, t_0_max=6):
    otlct = RecursiveLCT(pars, n_gen=5, trunc=10, a_max=a_max, t_0_max=t_0_max)
    t_0_array, a_array, kappa_plus = otlct.calculate_kappa()
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
    t_0_array, a_array, kappa_plus = recursive_lct_test(
        par, '../../figures/periodic/fct_re_variable_p03', a_max= T,
        t_0_max=2 * T)
    return t_0_array, a_array, kappa_plus


def main2():
    t_0_array, a_array, kappa_plus = recursive_lct_test(VariableParameters(
        p=1/3, h=0.5), '../../figures/periodic/lct_re_variable_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_plus


def main():
    print('Running simulation LCT with constant parameters and p=0.0')
    recursive_lct_test(ConstantParameters(p=0, h=0.5),
                       '../../figures/non_periodic/lct_re_constant_p0')
    print('Running simulation LCT with constant parameters and p=1/3')
    recursive_lct_test(ConstantParameters(p=1/3, h=0.5),
                       '../../figures/periodic/lct_re_constant_p03')
    print('Running simulation LCT with constant parameters and p=2/3')
    recursive_lct_test(ConstantParameters(p=2/3, h=0.5),
                       '../../figures/periodic/lct_re_constant_p06')
    print('Running simulation LCT with constant parameters and p=1')
    recursive_lct_test(ConstantParameters(p=1, h=0.5),
                       '../../figures/periodic/lct_re_constant_p1')

    print('Running simulation LCT with variable parameters and p=0.0')
    recursive_lct_test(VariableParameters(p=0, h=0.5),
                       '../../figures/non_periodic/lct_re_variable_p0')
    print('Running simulation LCT with variable parameters and p=1/3')
    recursive_lct_test(VariableParameters(p=1/3, h=0.5),
                       '../../figures/periodic/lct_re_variable_p03')
    print('Running simulation LCT with variable parameters and p=2/3')
    recursive_lct_test(VariableParameters(p=2/3, h=0.5),
                       '../../figures/periodic/lct_re_variable_p06')
    print('Running simulation LCT with variable parameters and p=1')
    recursive_lct_test(VariableParameters(p=1, h=0.5),
                       '../../figures/periodic/lct_re_variable_p1')


if __name__ == '__main__':
    t_0_array, a_array, kappa_plus = main3()
