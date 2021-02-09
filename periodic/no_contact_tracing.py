#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:18:07 2021

@author: saitel
"""
import numpy as np
import scipy.integrate as integrate
from math import exp
from parameters.parameters import ConstantParameters, VariableParameters
from helper.plotter import Plotter
from helper.exporter import Exporter


class NoCT:
    def __init__(self, parameters, a_max, t_0_max):
        self.beta = parameters.get_beta
        self.mu = parameters.get_mu
        self.sigma = parameters.get_sigma
        self.p = parameters.get_p
        self.h = parameters.get_h
        self.calculated = False
        self.d_calculated = False
        self.a_max = a_max
        self.t_0_max = t_0_max
        self.t_0_length = int(round(t_0_max / self.h(), 1))
        self.a_length = int(round(a_max / self.h(), 1))
        self.t_0_array = np.linspace(0.0, self.t_0_max, self.t_0_length + 1)
        self.a_array = np.linspace(0.0, self.a_max, self.a_length + 1)
        self.kappa_hat = np.zeros((self.t_0_length + 1, self.a_length + 1))
        self.dkappa_hat = np.zeros((self.t_0_length + 1, self.a_length + 1))

    def calculate_kappa_hat(self):
        for i in range(0, len(self.t_0_array)):
            for j in range(0, len(self.a_array)):
                self.kappa_hat[i, j] = self.calculate_kappa_hat_point(
                    self.a_array[j], self.t_0_array[i])
        self.calculated = True
        return self.t_0_array, self.a_array, self.kappa_hat

    def calculate_kappa_hat_point(self, a_v, t_0_v):
        """
        Returns kappa_hat(a;t_0)

        Parameters
        ----------
        a_v : Float. The value of current a.
        t_0_v : Float. The value of current t_0.

        Returns
        -------
        Float. Integral value

        """
        result = integrate.quad(lambda a: self.mu(a) +
                                self.sigma(a, t_0_v + a), 0, a_v)
        return exp(- result[0])

    def calculate_dkappa_hat(self):
        if(self.calculated is False):
            self.calculate_kappa_hat()

        for i in range(0, len(self.t_0_array)):
            for j in range(0, len(self.a_array)):
                self.dkappa_hat[i, j] = (self.kappa_hat(i, j) *
                                         (-self.mu(self.a_array[j]) -
                                          self.sigma(self.a_array[j],
                                                     self.t_0_array[i] +
                                                     self.a_array[j])))
        return self.dkappa_hat


def nct_test(pars, filename, a_max=2, t_0_max=6):
    nct = NoCT(pars, a_max, t_0_max)
    t_0_array, a_array, kappa_hat = nct.calculate_kappa_hat()
    # dkappa_hat = nct.calculate_dkappa_hat()
    a, t_0 = np.meshgrid(a_array, t_0_array)
    Plotter.plot_3D(t_0, a, kappa_hat, filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa_hat, filename + '_n60_10', azim=-60,
                    my=0.5)
    # Save data
    Exporter.save(t_0_array, a_array, kappa_hat, filename)
    return t_0, a, kappa_hat


def main():
    print('Running simulation NCT with constant parameters and p=0.0')
    nct_test(ConstantParameters(p=0, h=0.5),
             '../figures/non_periodic/NCT_constant_p0')
    print('Running simulation NCT with constant parameters and p=1/3')
    nct_test(ConstantParameters(p=1/3, h=0.5),
             '../figures/non_periodic/NCT_constant_p03')
    print('Running simulation NCT with constant parameters and p=2/3')
    nct_test(ConstantParameters(p=2/3, h=0.5),
             '../figures/non_periodic/NCT_constant_p06')
    print('Running simulation NCT with constant parameters and p=1')
    nct_test(ConstantParameters(p=1, h=0.5),
             '../figures/non_periodic/NCT_constant_p1')

    print('Running simulation NCT with variable parameters and p=0.0')
    nct_test(VariableParameters(p=0, h=0.5),
             '../figures/non_periodic/NCT_variable_p0')
    print('Running simulation NCT with variable parameters and p=1/3')
    nct_test(VariableParameters(p=1/3, h=0.5),
             '../figures/non_periodic/NCT_variable_p03')
    print('Running simulation NCT with variable parameters and p=2/3')
    nct_test(VariableParameters(p=2/3, h=0.5),
             '../figures/non_periodic/NCT_variable_p06')
    print('Running simulation NCT with variable parameters and p=1')
    nct_test(VariableParameters(p=1, h=0.5),
             '../figures/non_periodic/NCT_variable_p1')


if __name__ == '__main__':
    main()
