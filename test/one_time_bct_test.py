#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:38:14 2020

@author: saitel
"""
import numpy as np
from parameters import parameters
from periodic.backward_tracing import one_time as ot
from helper.plotter import Plotter
from helper.exporter import Exporter


def one_time_bct_test(pars, filename, a_max=2, t_0_max=6):
    """
    Test one time backward tracing using constant parameters

    Returns
    -------
    None.

    """
    bct = ot.OneTimeBCT(pars)

    t_0, a, kappa, dkappa = bct.calculate_kappa(a_max, t_0_max)

    # Save data
    Exporter.save(t_0, a, kappa, dkappa, '../data/periodicity/' + filename)

    # Plot data
    t_0, a = np.meshgrid(t_0, a)
    Plotter.plot_3D(t_0, a, kappa,
                    '../figures/periodicity/' + filename + '_60_10', my=0.5)
    Plotter.plot_3D(t_0, a, kappa,
                    '../figures/periodicity/' + filename + '_n60_10', azim=-60,
                    my=0.5)

    return t_0, a, kappa, dkappa


def main():

    print('Running simulation BCT with constant parameters and p=0.5')
    one_time_bct_test(parameters.ConstantParameters(), 'bct_ot_constant_p05')
    print('Running simulation BCT with constant parameters and p=0.0')
    one_time_bct_test(parameters.ConstantParameters(p=0), 'bct_ot_constant_p0')
    print('Running simulation BCT with constant parameters and p=1.0')
    one_time_bct_test(parameters.ConstantParameters(p=1), 'bct_ot_constant_p1')

    print('Running simulation BCT with variable parameters and p=0.5')
    one_time_bct_test(parameters.VariableParameters(), 'bct_ot_variable_p05')
    print('Running simulation BCT with variable parameters and p=0.0')
    one_time_bct_test(parameters.VariableParameters(p=0), 'bct_ot_variable_p0')
    print('Running simulation BCT with variable parameters and p=1.0')
    one_time_bct_test(parameters.VariableParameters(p=1), 'bct_ot_variable_p1')


if __name__ == '__main__':
    main()
