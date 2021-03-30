#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:40:10 2021

@author: saitel
"""
from periodic.no_contact_tracing import NoCT
from periodic.backward_tracing.one_time_bct import OneTimeBCT
from parameters.parameters import ConstantParameters, VariableParameters
import numpy as np
from helper.plotter import Plotter


class OneTimeLCT:
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
        self.bct = OneTimeBCT(parameters, self.trunc, t_0_max)
        self.fct = OneTimeBCT(parameters, self.trunc, t_0_max)
        self.t_0_array, self.a_array, self.kappa_hat = self.nct.calculate_kappa_hat()
        self.f = []

    def calculate_f_0(self):
        _, _, kappa_minus = self.bct.calculate_kappa_minus()
        f_0 = kappa_minus / self.kappa_hat
        f_0 = np.where(f_0 > 1, 1, f_0)  # truncate ratio
        f_0 = np.where(f_0 == 0, 1, f_0)  # 0 / 0 ~ 1
        return f_0

    def calculate_kappa(self):
        """
            Returns an array with kappa_plus
        """
        kappa_hat = self.kappa_hat[:, 0:(self.a_length + 1)]
        f = self.calculate_f()
        return (self.t_0_array, self.a_array[0:(self.a_length + 1)],
                kappa_hat * f)

 


def one_time_lct_test(pars, filename, a_max=2, t_0_max=6):
    otlct = OneTimeLCT(pars, n_gen=5, trunc=10, a_max=a_max, t_0_max=t_0_max)
    t_0_array, a_array, kappa_plus = otlct.calculate_kappa()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa_plus, filename + '_n60_10', azim=-60,
                    my=0.5)
    return t_0_array, a_array, kappa_plus


def main2():
    t_0_array, a_array, kappa_plus = one_time_lct_test(VariableParameters(
        p=1/3, h=0.5), '../../figures/periodic/lct_ot_variable_p03', a_max=2,
        t_0_max=2)
    return t_0_array, a_array, kappa_plus


def main():
    print('Running simulation LCT with constant parameters and p=0.0')
    one_time_lct_test(ConstantParameters(p=0, h=0.5),
                      '../../figures/non_periodic/lct_ot_constant_p0')
    print('Running simulation LCT with constant parameters and p=1/3')
    one_time_lct_test(ConstantParameters(p=1/3, h=0.5),
                      '../../figures/periodic/lct_ot_constant_p03')
    print('Running simulation LCT with constant parameters and p=2/3')
    one_time_lct_test(ConstantParameters(p=2/3, h=0.5),
                      '../../figures/periodic/lct_ot_constant_p06')
    print('Running simulation LCT with constant parameters and p=1')
    one_time_lct_test(ConstantParameters(p=1, h=0.5),
                      '../../figures/periodic/lct_ot_constant_p1')

    print('Running simulation LCT with variable parameters and p=0.0')
    one_time_lct_test(VariableParameters(p=0, h=0.5),
                      '../../figures/non_periodic/lct_ot_variable_p0')
    print('Running simulation LCT with variable parameters and p=1/3')
    one_time_lct_test(VariableParameters(p=1/3, h=0.5),
                      '../../figures/periodic/lct_ot_variable_p03')
    print('Running simulation LCT with variable parameters and p=2/3')
    one_time_lct_test(VariableParameters(p=2/3, h=0.5),
                      '../../figures/periodic/lct_ot_variable_p06')
    print('Running simulation LCT with variable parameters and p=1')
    one_time_lct_test(VariableParameters(p=1, h=0.5),
                      '../../figures/periodic/lct_ot_variable_p1')


if __name__ == '__main__':
    t_0_array, a_array, kappa_plus = main2()
